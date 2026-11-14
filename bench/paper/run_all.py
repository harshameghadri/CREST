"""Plan and execute every benchmark run, each in its own process.

    python bench/paper/run_all.py --data-dir bench_data --out RESULTS --datasets pbmc3k kang \
        --repeats 5 --main-threads 8 16 32 64 --ooc-threads 8 64 --threads-scan 1 2 4 8 --thread-dataset pbmc68k

Three kinds of runs (all in fresh processes, interleaved):

* main: the core workflow of every (dataset, tool) at each of --main-threads (CREST and
  scanpy) or --ooc-threads (CREST out-of-core), so the report can compare both at every
  matched thread count and at each tool's best measured thread count;
* modules: one full-profile run per (dataset, tool) at all cores (--module-repeats times),
  for the optional modules; kept apart so their time and memory never mix into the
  headline. Scrublet is skipped above --scrublet-max-cells (its kNN cost grows ~cells^2);
* thread scan: the core workflow of --thread-dataset at each of --threads-scan, pinned.

Design choices for publishable numbers:

* every (dataset, tool, threads, repeat) is a fresh process, so no state, cache
  or allocator reuse leaks between runs;
* repeats are interleaved: repeat r of every configuration runs before repeat
  r + 1 of any, in a shuffled (seeded) order, so drift over the session
  (thermal throttling, background load) spreads over all tools instead of
  biasing one;
* thread counts are fixed through RAYON/OMP/OPENBLAS/MKL/NUMBA env variables and
  (Linux) CPU affinity to the first p cores;
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

THREAD_VARS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def n_cells(name: str) -> int:
    return SYNTHETIC.get(name) or (SUBSETS[name][1] if name in SUBSETS else REGISTRY[name]["cells"])


def plan(a) -> list[dict]:
    ncpu = os.cpu_count() or 1
    main_threads = sorted({min(p, ncpu) for p in (a.main_threads or [ncpu])})
    ooc_threads = sorted({min(p, ncpu) for p in (a.ooc_threads or main_threads)})
    modules = a.profile == "full"
    base, seen = [], set()

    def add(job, reps):
        key = (job["dataset"], job["tool"], job["profile"], job["threads"])
        if key not in seen:
            seen.add(key)
            base.append((job, reps))

    for ds in a.datasets:
        for tool in a.tools:
            if tool == "scanpy" and a.scanpy_max_cells and n_cells(ds) > a.scanpy_max_cells:
                continue
            for p in (ooc_threads if tool == "crest-ooc" else main_threads):
                # outputs for the accuracy comparison: from the module run if there is one
                save = (not modules) and p == main_threads[-1]
                add({"dataset": ds, "tool": tool, "threads": p, "pin": p < ncpu and not a.no_pin,
                     "profile": "core", "save": save}, a.repeats)
            if modules:
                add({"dataset": ds, "tool": tool, "threads": ncpu, "pin": False, "profile": "full", "save": True},
                    a.module_repeats if a.module_repeats is not None else a.repeats)
    if a.thread_dataset and a.threads_scan:
        for tool in ("crest", "scanpy"):
            if tool not in a.tools:
                continue
            for p in sorted(set(a.threads_scan) | {ncpu}):
                if p <= ncpu:
                    add({"dataset": a.thread_dataset, "tool": tool, "threads": p, "pin": p < ncpu and not a.no_pin,
                         "profile": "core", "save": False}, a.repeats)
    runs = []
    rng = random.Random(a.seed)
    for r in range(max(reps for _, reps in base) if base else 0):
        block = [dict(job, repeat=r) for job, reps in base if r < reps]
        rng.shuffle(block)
        runs += block
    return runs


# Rough cost model for the plan's time estimate: seconds per cell of the core workflow by
# thread count, from medians of the 2026-10-02 standard-tier run on the Threadripper PRO
# 3975WX (32 cores / 64 threads). Only a planning aid: other machines and versions differ.
_PER_CELL = {"crest": {1: 6.8e-4, 2: 3.8e-4, 4: 2.2e-4, 8: 1.4e-4, 16: 1.1e-4, 32: 1.0e-4, 64: 1.0e-4},
             "scanpy": {1: 2.8e-3, 2: 2.2e-3, 4: 1.8e-3, 8: 1.5e-3, 16: 1.8e-3, 32: 2.3e-3, 64: 2.9e-3}}
_PER_CELL["crest-ooc"] = {p: 1.15 * v for p, v in _PER_CELL["crest"].items()}
_STARTUP = {"crest": 3.0, "crest-ooc": 4.0, "scanpy": 8.0}  # interpreter, imports, read


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


def run(job: dict, a, log_dir: Path) -> str:
    raw = Path(a.out) / "raw"
    tag = f"{job['tool']}__{job['dataset']}__{job['profile']}__t{job['threads']}__r{job['repeat']}"
    js = raw / f"{tag}.json"
    rec = {"tag": tag, **{k: job[k] for k in ("tool", "dataset", "profile", "threads", "repeat")},
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    if js.exists() and json.loads(js.read_text()).get("status") == "ok":
        _log_attempt(a, {**rec, "status": "skipped", "reason": "already ok (resume)", "seconds": 0.0})
        return "skip"
    env = dict(os.environ)
    for v in THREAD_VARS:
        env[v] = str(job["threads"])
    cmd = [sys.executable, str(HERE / "run_one.py"), "--tool", job["tool"], "--data",
           str(Path(a.data_dir) / f"{job['dataset']}.h5"), "--out", str(raw), "--profile", job["profile"],
           "--threads", str(job["threads"]), "--repeat", str(job["repeat"])]
    if job["pin"]:
        cmd.append("--pin")
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
        _log_attempt(a, {**rec, "status": "ok", "returncode": 0, "seconds": el})
        return "ok"
    reason = "timeout" if rc == "timeout" else ("killed (likely out of memory)" if rc in (-9, 137) else f"exit {rc}")
    own = json.loads(js.read_text()) if js.exists() else None
    if own is not None and own.get("status") == "error" and own.get("error"):
        reason = f"{reason}: {own['error']}"  # run_one caught the exception and recorded it
    elif own is None or own.get("status") == "ok":  # no JSON from this attempt (killed), or a stale ok one
        js.write_text(json.dumps({"tool": job["tool"], "dataset": job["dataset"], "threads": job["threads"],
                                  "repeat": job["repeat"], "profile": job["profile"], "status": "failed",
                                  "error": reason, "total_seconds": el, "steps": {}}, indent=1))
    _log_attempt(a, {**rec, "status": "failed", "returncode": rc if rc != "timeout" else None, "reason": reason,
                     "seconds": el, "log": str(log_dir / f"{tag}.log")})
    return f"failed: {reason}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--tools", nargs="+", default=["crest", "crest-ooc", "scanpy"])
    ap.add_argument("--profile", choices=["core", "full"], default="full",
                    help="full = also run the optional modules (separate runs); core = headline only")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--main-threads", nargs="*", type=int, default=[],
                    help="thread counts of the main core-workflow runs of CREST and scanpy (default: all cores)")
    ap.add_argument("--ooc-threads", nargs="*", type=int, default=[],
                    help="thread counts of the main runs of out-of-core CREST (default: --main-threads)")
    ap.add_argument("--module-repeats", type=int, default=None, help="repeats of the module runs (default: --repeats)")
    ap.add_argument("--scrublet-max-cells", type=int, default=150_000,
                    help="skip Scrublet in module runs above this many cells (0 = never skip)")
    ap.add_argument("--threads-scan", nargs="*", type=int, default=[])
    ap.add_argument("--thread-dataset", default=None)
    ap.add_argument("--scanpy-max-cells", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    logs = out / "logs" / "runs"
    logs.mkdir(parents=True, exist_ok=True)
    runs = plan(a)
    est = sum(estimate_seconds(j) for j in runs)
    (out / "plan.json").write_text(json.dumps({"args": vars(a), "runs": runs, "estimated_hours": est / 3600}, indent=1))
    kinds = {}
    for j in runs:
        k = "core (main + thread scan)" if j["profile"] == "core" else "modules (full profile)"
        kinds[(k, j["tool"])] = kinds.get((k, j["tool"]), 0) + 1
    print(f"{len(runs)} runs planned ({a.repeats} repeats, interleaved, seed {a.seed}); "
          f"rough wall time {est / 3600:.1f} h (+-50%; calibrated on a 32-core Threadripper)")
    for (k, t), c in sorted(kinds.items()):
        print(f"  {c:5d}  {t:9s} {k}")
    if a.dry_run:
        for j in runs:
            print(j)
        return
    t_all = time.time()
    for i, j in enumerate(runs, 1):
        t0 = time.time()
        status = run(j, a, logs)
        el = time.time() - t_all
        print(f"[{i:4d}/{len(runs)}] {j['tool']:9s} {j['dataset']:16s} {j['profile']:4s} t={j['threads']:<3d} r={j['repeat']}  "
              f"{status:28s} {time.time() - t0:8.1f}s   (elapsed {el / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
