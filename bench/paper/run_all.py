"""Plan and execute every benchmark run, each in its own process.

    python bench/paper/run_all.py --data-dir bench_data --out RESULTS --datasets pbmc3k kang \
        --repeats 5 --thread-grid 1 2 4 8 16 32 64 --thread-dataset pbmc68k

Two modes:

* ``--mode pilot``: the untimed tuning pilot. For each tool it runs the core workflow once per
  (pilot dataset, thread count, settings variant) with ``--pilot-datasets``, picks the fastest variant
  per thread count and the best thread count per dataset (``protocol.choose_tuning``), and writes
  ``RESULTS/tuning.json``. Pilot results go to ``RESULTS/pilot/`` and never enter the report. Both tools get
  the same number of variants (the same tuning budget). The variants set ``NUMBA_THREADING_LAYER``,
  ``NUMBA_NUM_THREADS``, ``OMP`` / ``OPENBLAS`` / ``MKL`` / rayon threads and scanpy's ``n_jobs``.
* ``--mode main`` (default): the timed runs. ``--tuning RESULTS/tuning.json`` applies the pilot's choice
  per tool and thread count; the chosen settings are written into ``plan.json`` (key ``tuning``, and the
  resolved ``settings`` of every run).

Kinds of runs in ``--mode main`` (all in fresh processes, interleaved):

* main: the core workflow of every (dataset, tool) at each thread count of the grid
  (``--thread-grid``, default 1 2 4 8 16 32 64, clamped to the logical CPUs; ``--main-threads``
  overrides it), so the report can compare both tools at every matched thread count and at each
  tool's best. ``--grid-mode best`` runs only 1, the pilot's best and all CPUs per (tool, dataset);
* modules: one full-profile run per (dataset, tool) at all cores (--module-repeats times), for the
  optional modules; kept apart so their time and memory never mix into the headline. Scrublet is skipped
  above --scrublet-max-cells (its kNN cost grows ~cells^2);
* thread scan: the core workflow of --thread-dataset at each of --threads-scan;
* controls (``--controls``): never part of the headline. ``unpinned``: the same run without pinning, at
  ``--control-threads``; ``default-settings``: the untuned settings (every variable = the thread count) where
  the pilot chose something else. Both on ``--control-datasets`` (default: the thread-scan dataset).

Design choices for publishable numbers:

* every (dataset, tool, threads, repeat) is a fresh process, so no state, cache
  or allocator reuse leaks between runs;
* repeats are interleaved: repeat r of every configuration runs before repeat
  r + 1 of any, in a shuffled (seeded) order, so drift over the session
  (thermal throttling, background load) spreads over all tools instead of
  biasing one;
* the pinning policy is the same at every thread count (``protocol.PIN_POLICY``): each run is pinned to
  the first p logical CPUs, physical cores first, and the affinity is recorded;
* quiet host: the per-run median of foreign busy cores must stay below 1. A run that violates it is
  re-run once at the end of the pass; a cell that violates twice is kept, flagged, with its load in the
  report. The first attempt's files are kept under ``raw/_quiet/``;
* a run killed by the OS (out of memory) or by the timeout is recorded as a
  failed run with its reason, never silently dropped;
* the plan is resumable: runs whose JSON already exists with status ok are skipped;
* every attempt (ok, failed, timed out, skipped on resume) is appended to
  ``attempts.jsonl``, which is never truncated: a resume that succeeds on retry does not
  erase the earlier failure, and the report counts every attempt.

The expected number of runs and a rough wall time (from per-cell costs measured on the
32-core Threadripper in the 2026-10-02 run; other machines differ) are printed first.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from datasets import REGISTRY, SUBSETS, SYNTHETIC  # noqa: E402
import protocol  # noqa: E402
from protocol import THREAD_GRID, THREAD_VARS, VARIANTS  # noqa: E402,F401


def n_cells(name: str) -> int:
    sub = protocol.parse_subset_name(name)
    if sub:
        parent = REGISTRY.get(sub[0], {}).get("cells")
        return min(sub[1], parent) if parent else sub[1]
    return SYNTHETIC.get(name) or (SUBSETS[name][1] if name in SUBSETS else REGISTRY[name]["cells"])


def resolve_threads(a, ncpu: int) -> tuple[list[int], list[int]]:
    """(main thread counts, out-of-core thread counts): ``--main-threads`` if given, else the grid."""
    grid = protocol.clamp_grid(getattr(a, "thread_grid", None) or THREAD_GRID, ncpu)
    main = protocol.clamp_grid(a.main_threads, ncpu) if a.main_threads else grid
    ooc = protocol.clamp_grid(a.ooc_threads, ncpu) if a.ooc_threads else main
    return main, ooc


def _job(a, tuning, ncpu, dataset, tool, p, profile, save, control=None, variant=None, pin=None):
    """One planned run, with its resolved thread settings."""
    variant = variant or protocol.tuned_variant(tuning, tool, p)
    return {"dataset": dataset, "tool": tool, "threads": p, "profile": profile, "save": save,
            "pin": (not getattr(a, "no_pin", False)) if pin is None else pin, "variant": variant, "control": control,
            "settings": protocol.variant_settings(tool, variant, p)}


def plan(a) -> list[dict]:
    ncpu = os.cpu_count() or 1
    main_threads, ooc_threads = resolve_threads(a, ncpu)
    tuning = getattr(a, "tuning_data", None)
    best_mode = getattr(a, "grid_mode", "full") == "best" and tuning
    modules = a.profile == "full"
    base, seen = [], set()

    def add(job, reps):
        key = (job["dataset"], job["tool"], job["profile"], job["threads"], job["pin"], job["variant"], job["control"])
        if key not in seen:
            seen.add(key)
            base.append((job, reps))

    def threads_of(tool, ds, grid):
        if not best_mode:
            return grid
        best = (tuning.get("best_threads", {}).get(protocol.base_tool(tool), {}) or {}).get(ds)
        return sorted({p for p in (1, best, grid[-1]) if p is not None and p in grid}) if grid else grid

    for ds in a.datasets:
        for tool in a.tools:
            if protocol.base_tool(tool) == "scanpy" and a.scanpy_max_cells and n_cells(ds) > a.scanpy_max_cells:
                continue
            grid = ooc_threads if tool == "crest-ooc" else main_threads
            for p in threads_of(tool, ds, grid):
                # outputs for the accuracy comparison: from the module run if there is one
                save = (not modules) and p == grid[-1] and tool != "scanpy-lean"
                add(_job(a, tuning, ncpu, ds, tool, p, "core", save), a.repeats)
            if modules and tool != "scanpy-lean":
                add(_job(a, tuning, ncpu, ds, tool, ncpu, "full", True),
                    a.module_repeats if a.module_repeats is not None else a.repeats)
    thread_ds = getattr(a, "thread_dataset", None)
    if thread_ds and getattr(a, "threads_scan", None):
        for tool in ("crest", "scanpy"):
            if tool not in a.tools:
                continue
            for p in sorted(set(a.threads_scan) | {ncpu}):
                if p <= ncpu:
                    add(_job(a, tuning, ncpu, thread_ds, tool, p, "core", False), a.repeats)
    # controls: never part of the headline (the report keeps them out of every statistic)
    controls = getattr(a, "controls", None) or []
    ctl_ds = getattr(a, "control_datasets", None) or ([thread_ds] if thread_ds else [])
    ctl_threads = protocol.clamp_grid(getattr(a, "control_threads", None) or [8, ncpu], ncpu)
    for ds in ctl_ds:
        for tool in ("crest", "scanpy"):
            if tool not in a.tools or (protocol.base_tool(tool) == "scanpy" and a.scanpy_max_cells
                                       and n_cells(ds) > a.scanpy_max_cells):
                continue
            for p in ctl_threads:
                if "unpinned" in controls and p < ncpu and not getattr(a, "no_pin", False):
                    add(_job(a, tuning, ncpu, ds, tool, p, "core", False, control="unpinned", pin=False), a.repeats)
                if "default-settings" in controls and protocol.tuned_variant(tuning, tool, p) != protocol.DEFAULT_VARIANT:
                    add(_job(a, tuning, ncpu, ds, tool, p, "core", False, control="default-settings",
                             variant=protocol.DEFAULT_VARIANT), a.repeats)
    runs = []
    rng = random.Random(a.seed)
    for r in range(max(reps for _, reps in base) if base else 0):
        block = [dict(job, repeat=r) for job, reps in base if r < reps]
        rng.shuffle(block)
        runs += block
    return runs


def plan_pilot(a) -> list[dict]:
    """The untimed tuning pilot: one run per (pilot dataset, tool, thread count, variant), core profile. A variant
    whose resolved settings equal the default's at a thread count (a cap above the thread count, anything at one
    thread) is not run there."""
    ncpu = os.cpu_count() or 1
    main_threads, _ = resolve_threads(a, ncpu)
    runs = []
    for ds in a.pilot_datasets:
        for tool in ("crest", "scanpy"):
            if tool not in a.tools:
                continue
            for p in main_threads:
                default = protocol.variant_settings(tool, protocol.DEFAULT_VARIANT, p)
                for v in VARIANTS[tool]:
                    if v != protocol.DEFAULT_VARIANT and protocol.variant_settings(tool, v, p) == default:
                        continue   # the same settings as the default at this thread count: a second run would only add noise
                    j = _job(a, None, ncpu, ds, tool, p, "core", False, variant=v)
                    runs.append(dict(j, repeat=0, pilot_variant=v))
    random.Random(a.seed).shuffle(runs)
    return runs


# Rough cost model for the plan's time estimate: seconds per cell of the core workflow by
# thread count, from medians of the 2026-10-02 standard-tier run on the Threadripper PRO
# 3975WX (32 cores / 64 threads). Only a planning aid: other machines and versions differ.
_PER_CELL = {"crest": {1: 6.8e-4, 2: 3.8e-4, 4: 2.2e-4, 8: 1.4e-4, 16: 1.1e-4, 32: 1.0e-4, 64: 1.0e-4},
             "scanpy": {1: 2.8e-3, 2: 2.2e-3, 4: 1.8e-3, 8: 1.5e-3, 16: 1.8e-3, 32: 2.3e-3, 64: 2.9e-3}}
_PER_CELL["crest-ooc"] = {p: 1.15 * v for p, v in _PER_CELL["crest"].items()}
_PER_CELL["scanpy-lean"] = _PER_CELL["scanpy"]
_STARTUP = {"crest": 3.0, "crest-ooc": 4.0, "scanpy": 8.0, "scanpy-lean": 8.0}  # interpreter, imports, read


def estimate_seconds(job: dict) -> float:
    table = _PER_CELL[job["tool"]]
    p = min(table, key=lambda q: abs(q - job["threads"]))
    n = n_cells(job["dataset"])
    core = table[p] * n
    warm = table[p] * min(n, 20_000)
    return _STARTUP[job["tool"]] + warm + core * (2.5 if job["profile"] == "full" else 1.0)


def _log_attempt(a, rec: dict) -> None:
    with open(Path(a.out) / "attempts.jsonl", "a") as f:  # append-only: never truncated
        f.write(json.dumps(rec) + "\n")


def load_attempts(out: Path) -> list[dict]:
    f = Path(out) / "attempts.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []


def run(job: dict, a, log_dir: Path, raw: Path | None = None, force: bool = False) -> str:
    raw = raw or Path(a.out) / "raw"
    tag = protocol.job_tag(job)
    js = raw / f"{tag}.json"
    rec = {"tag": tag, **{k: job[k] for k in ("tool", "dataset", "profile", "threads", "repeat")},
           "variant": job.get("variant"), "control": job.get("control"),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if not force and js.exists() and json.loads(js.read_text()).get("status") == "ok":
        _log_attempt(a, {**rec, "status": "skipped", "reason": "already ok (resume)", "seconds": 0.0})
        return "skip"
    env = dict(os.environ)
    for v in (*THREAD_VARS, "NUMBA_THREADING_LAYER"):
        env.pop(v, None)
    env.update(protocol.settings_env(job["settings"]))
    cmd = [sys.executable, str(HERE / "run_one.py"), "--tool", job["tool"], "--data",
           str(Path(a.data_dir) / f"{job['dataset']}.h5"), "--out", str(raw), "--profile", job["profile"],
           "--threads", str(job["threads"]), "--repeat", str(job["repeat"]), "--variant", str(job.get("variant")),
           "--n-jobs", str(job["settings"]["n_jobs"]), "--sample-interval", str(getattr(a, "sample_interval", 0.05))]
    if job["pin"]:
        cmd.append("--pin")
    if job.get("control"):
        cmd += ["--control", job["control"]]
    if job.get("pilot_variant"):
        cmd.append("--pilot")
    if job["save"] and job["repeat"] == 0:
        cmd.append("--save-outputs")
    if job["profile"] == "full" and a.scrublet_max_cells:
        cmd += ["--scrublet-max-cells", str(a.scrublet_max_cells)]
    t0 = time.time()
    with open(log_dir / f"{tag}.log", "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            rc = p.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            rc = "timeout"
    el = time.time() - t0
    if rc == 0:
        own = json.loads(js.read_text()) if js.exists() else {}
        quiet = own.get("quiet") or {}
        if own:  # what the planner decided, next to what the process measured of itself
            own["harness"] = {"process_wall_seconds": el, "settings": job["settings"], "pin_planned": job["pin"],
                              "env": protocol.settings_env(job["settings"])}
            js.write_text(json.dumps(own, indent=1, default=float))
        _log_attempt(a, {**rec, "status": "ok", "returncode": 0, "seconds": el,
                         "median_foreign_busy_cores": quiet.get("median_foreign_busy_cores"),
                         "quiet_violation": bool(quiet.get("violated"))})
        return "ok"
    reason = "timeout" if rc == "timeout" else ("killed (likely out of memory)" if rc in (-9, 137) else f"exit {rc}")
    own = json.loads(js.read_text()) if js.exists() else None
    if own is not None and own.get("status") == "error" and own.get("error"):
        reason = f"{reason}: {own['error']}"  # run_one caught the exception and recorded it
    elif own is None or own.get("status") == "ok":  # no JSON from this attempt (killed), or a stale ok one
        js.write_text(json.dumps({"tool": job["tool"], "dataset": job["dataset"], "threads": job["threads"],
                                  "repeat": job["repeat"], "profile": job["profile"], "status": "failed",
                                  "control": job.get("control"), "variant": job.get("variant"),
                                  "error": reason, "total_seconds": el, "steps": {}}, indent=1))
    _log_attempt(a, {**rec, "status": "failed", "returncode": rc if rc != "timeout" else None, "reason": reason,
                     "seconds": el, "log": str(log_dir / f"{tag}.log")})
    return f"failed: {reason}"


def _archive_attempt(raw: Path, tag: str, n: int) -> None:
    """Move the files of attempt ``n`` of ``tag`` to ``raw/_quiet/`` (not read by the report's run loader)."""
    q = raw / "_quiet"
    q.mkdir(exist_ok=True)
    for f in raw.glob(f"{tag}.*"):
        f.rename(q / f.name.replace(tag, f"{tag}.attempt{n}", 1))


def quiet_pass(runs: list[dict], a, logs: Path, raw: Path) -> list[str]:
    """Apply the quiet-host rule to the finished runs: re-run a violating run once, flag one that violates
    again. Returns a line per run touched."""
    notes = []
    for j in runs:
        tag = protocol.job_tag(j)
        js = raw / f"{tag}.json"
        if not js.exists():
            continue
        rec = json.loads(js.read_text())
        if rec.get("status") != "ok":
            continue
        for _ in range(protocol.QUIET_MAX_RERUNS + 1):
            q = rec.get("quiet") or {}
            n = sum(1 for x in load_attempts(a.out) if x.get("tag") == tag and x.get("status") == "ok")
            decision = protocol.quiet_decision(bool(q.get("violated")), n)
            if decision == "accept":
                break
            if decision == "flag":
                if not q.get("flag"):
                    q["flag"] = f"median foreign busy cores {q.get('median_foreign_busy_cores'):.2f} >= " \
                                f"{q.get('threshold')} in {n} attempts"
                    rec["quiet"] = q
                    js.write_text(json.dumps(rec, indent=1, default=float))
                    _log_attempt(a, {"tag": tag, "status": "flagged", "reason": q["flag"]})
                notes.append(f"flagged {tag}: {q['flag']}")
                break
            _archive_attempt(raw, tag, n)
            status = run(j, a, logs, raw, force=True)
            notes.append(f"re-ran {tag} (quiet-host violation, attempt {n}): {status}")
            if not js.exists() or json.loads(js.read_text()).get("status") != "ok":
                break
            rec = json.loads(js.read_text())
    return notes


def _progress(i, n, j, status, t0, t_all):
    print(f"[{i:4d}/{n}] {j['tool']:11s} {j['dataset']:16s} {j['profile']:4s} t={j['threads']:<3d} r={j['repeat']}  "
          f"{(j.get('control') or ''):16s}{(j.get('pilot_variant') or j.get('variant') or ''):11s}{status:28s} "
          f"{time.time() - t0:8.1f}s   (elapsed {(time.time() - t_all) / 60:.1f} min)", flush=True)


def pilot(a, out: Path) -> None:
    runs = plan_pilot(a)
    raw = out / "pilot" / "raw"
    logs = out / "pilot" / "logs"
    raw.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    est = sum(estimate_seconds(j) for j in runs)
    (out / "pilot" / "plan.json").write_text(json.dumps({"args": vars(a), "runs": runs, "estimated_hours": est / 3600}, indent=1))
    per_tool = {t: sum(1 for j in runs if j["tool"] == t) for t in ("crest", "scanpy")}
    print(f"tuning pilot: {len(runs)} untimed runs {per_tool} (same budget per tool: "
          f"{protocol.tuning_budget_is_equal()}), rough wall time {est / 3600:.1f} h")
    if a.dry_run:
        for j in runs:
            print(j)
        return
    t_all = time.time()
    for i, j in enumerate(runs, 1):
        t0 = time.time()
        _progress(i, len(runs), j, run(j, a, logs, raw), t0, t_all)
    records = []
    for j in runs:
        f = raw / f"{protocol.job_tag(j)}.json"
        r = json.loads(f.read_text()) if f.exists() else {}
        ok = r.get("status") == "ok"
        records.append({"tool": j["tool"], "dataset": j["dataset"], "threads": j["threads"], "variant": j["variant"],
                        "seconds": r.get("total_seconds") if ok else None, "status": r.get("status"),
                        "numba_threading_layer": (r.get("env") or {}).get("numba_threading_layer")})
    tuning = protocol.choose_tuning(records)
    tuning["records"] = records
    tuning["pilot_datasets"] = list(a.pilot_datasets)
    (out / "tuning.json").write_text(json.dumps(tuning, indent=1))
    print(f"chosen variants: {tuning['variant']}\nbest threads per dataset: {tuning['best_threads']}\n"
          f"written: {out / 'tuning.json'}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--tools", nargs="+", default=["crest", "crest-ooc", "scanpy"],
                    help="crest, crest-ooc, scanpy, scanpy-lean (the memory-lean profile of scanpy: core runs only)")
    ap.add_argument("--mode", choices=["main", "pilot"], default="main",
                    help="main = the timed runs; pilot = the untimed tuning pilot (writes tuning.json)")
    ap.add_argument("--profile", choices=["core", "full"], default="full",
                    help="full = also run the optional modules (separate runs); core = headline only")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--thread-grid", nargs="*", type=int, default=list(THREAD_GRID),
                    help="thread counts of the main runs, clamped to the logical CPUs (default: 1 2 4 8 16 32 64)")
    ap.add_argument("--main-threads", nargs="*", type=int, default=[],
                    help="override --thread-grid for the main core-workflow runs of CREST and scanpy")
    ap.add_argument("--ooc-threads", nargs="*", type=int, default=[],
                    help="thread counts of the main runs of out-of-core CREST (default: the main thread counts)")
    ap.add_argument("--tuning", default=None, help="tuning.json from a pilot: apply its per-thread-count variants")
    ap.add_argument("--grid-mode", choices=["full", "best"], default="full",
                    help="full = every grid point; best = 1, the pilot's best point per (tool, dataset) and all CPUs")
    ap.add_argument("--pilot-datasets", nargs="*", default=[], help="datasets of the tuning pilot (--mode pilot)")
    ap.add_argument("--controls", nargs="*", choices=list(protocol.CONTROLS), default=[],
                    help="control runs kept out of every statistic: unpinned, default-settings")
    ap.add_argument("--control-datasets", nargs="*", default=[], help="default: the --thread-dataset")
    ap.add_argument("--control-threads", nargs="*", type=int, default=[], help="default: 8 and all CPUs")
    ap.add_argument("--module-repeats", type=int, default=None, help="repeats of the module runs (default: --repeats)")
    ap.add_argument("--scrublet-max-cells", type=int, default=150_000,
                    help="skip Scrublet in module runs above this many cells (0 = never skip)")
    ap.add_argument("--threads-scan", nargs="*", type=int, default=[])
    ap.add_argument("--thread-dataset", default=None)
    ap.add_argument("--scanpy-max-cells", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--sample-interval", type=float, default=0.05, help="memory and load sampling interval (s)")
    ap.add_argument("--no-quiet-rerun", action="store_true", help="do not re-run runs that violate the quiet-host rule")
    ap.add_argument("--no-pin", action="store_true", help="no pinning for any run")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main():
    a = build_parser().parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.mode == "pilot":
        if not a.pilot_datasets:
            sys.exit("--mode pilot needs --pilot-datasets")
        return pilot(a, out)
    a.tuning_data = json.loads(Path(a.tuning).read_text()) if a.tuning else None
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    logs = out / "logs" / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    runs = plan(a)
    est = sum(estimate_seconds(j) for j in runs)
    ncpu = os.cpu_count() or 1
    main_threads, ooc_threads = resolve_threads(a, ncpu)
    args = {k: v for k, v in vars(a).items() if k != "tuning_data"}
    (out / "plan.json").write_text(json.dumps({
        "args": args, "runs": runs, "estimated_hours": est / 3600, "ncpu": ncpu, "main_threads": main_threads,
        "ooc_threads": ooc_threads, "tuning": a.tuning_data, "pin_policy": protocol.PIN_POLICY,
        "quiet": {"max_foreign_busy_cores": protocol.QUIET_MAX_FOREIGN_CORES, "max_reruns": protocol.QUIET_MAX_RERUNS},
        "variants": {t: list(v) for t, v in VARIANTS.items()}}, indent=1))
    kinds = {}
    for j in runs:
        k = "controls" if j.get("control") else ("core (main + thread scan)" if j["profile"] == "core" else "modules (full profile)")
        kinds[(k, j["tool"])] = kinds.get((k, j["tool"]), 0) + 1
    print(f"{len(runs)} runs planned ({a.repeats} repeats, interleaved, seed {a.seed}); "
          f"rough wall time {est / 3600:.1f} h (+-50%; calibrated on a 32-core Threadripper)")
    for (k, t), c in sorted(kinds.items()):
        print(f"  {c:5d}  {t:11s} {k}")
    if a.dry_run:
        for j in runs:
            print(j)
        return
    t_all = time.time()
    for i, j in enumerate(runs, 1):
        t0 = time.time()
        _progress(i, len(runs), j, run(j, a, logs, raw), t0, t_all)
    if not a.no_quiet_rerun:
        for line in quiet_pass(runs, a, logs, raw):
            print("  quiet host:", line, flush=True)


if __name__ == "__main__":
    main()
