"""Plan and execute every benchmark run, each in its own process.

    python bench/paper/run_all.py --data-dir bench_data --out RESULTS --datasets pbmc3k kang \
        --repeats 5 --main-threads 8 64 --threads-scan 1 2 4 8 --thread-dataset pbmc68k

Three kinds of runs (all in fresh processes, interleaved):

* main: the core workflow of every (dataset, tool) at each of --main-threads, so the
  report can compare both at matched threads and at each tool's best thread count;
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
* the plan is resumable: runs whose JSON already exists with status ok are skipped.
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
            for p in main_threads:
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


def run(job: dict, a, log_dir: Path) -> str:
    raw = Path(a.out) / "raw"
    tag = f"{job['tool']}__{job['dataset']}__{job['profile']}__t{job['threads']}__r{job['repeat']}"
    js = raw / f"{tag}.json"
    if js.exists() and json.loads(js.read_text()).get("status") == "ok":
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
    if rc == 0:
        return "ok"
    reason = "timeout" if rc == "timeout" else ("killed (likely out of memory)" if rc in (-9, 137) else f"exit {rc}")
    if not js.exists() or json.loads(js.read_text()).get("status") == "ok":
        js.write_text(json.dumps({"tool": job["tool"], "dataset": job["dataset"], "threads": job["threads"],
                                  "repeat": job["repeat"], "profile": job["profile"], "status": "failed",
                                  "error": reason, "total_seconds": time.time() - t0, "steps": {}}, indent=1))
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
                    help="thread counts for the main core-workflow runs (default: all cores)")
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
    (out / "plan.json").write_text(json.dumps({"args": vars(a), "runs": runs}, indent=1))
    print(f"{len(runs)} runs planned ({a.repeats} repeats, interleaved, seed {a.seed})")
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
