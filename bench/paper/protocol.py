"""Benchmark protocol rules, as pure functions (no benchmark data, no tool imports).

Everything the pre-registration fixes that can be expressed as a rule lives here, so that the
run planner (``run_all.py``), the single-run script (``run_one.py``) and the report
(``summarize.py``) apply the same rule and the unit tests can pin it:

* the thread grid and the CPU pinning policy (physical cores first, every run pinned);
* the per-tool thread / threading-layer settings that the untuned and the tuned runs use, and
  the untimed tuning pilot's choice among them;
* the run tag and the list of result files each run must leave behind;
* the quiet-host rule (foreign load during a run) and what to do about a violation.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# --------------------------------------------------------------------------- thread grid
THREAD_GRID = (1, 2, 4, 8, 16, 32, 64)
THREAD_VARS = ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")

PIN_POLICY = ("every run is pinned (sched_setaffinity) to the first p logical CPUs in physical-core-first order: "
              "one thread context of every core before any second SMT sibling; with p equal to the number of "
              "logical CPUs the mask is all CPUs; --no-pin switches pinning off for every run alike; a separate "
              "unpinned control at selected thread counts measures the effect of pinning")


def clamp_grid(requested, ncpu: int) -> list[int]:
    """Sorted distinct thread counts, each clamped to the number of logical CPUs."""
    return sorted({max(1, min(int(p), ncpu)) for p in requested})


# --------------------------------------------------------------------------- pinning
def parse_cpu_list(text: str) -> list[int]:
    """Linux cpu list syntax: ``"0,32"``, ``"0-3"``, ``"0-1,8-9"``."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out += range(int(lo), int(hi) + 1)
        else:
            out.append(int(part))
    return out


def read_smt_siblings(root: str = "/sys/devices/system/cpu") -> dict[int, tuple[int, ...]]:
    """{cpu: its SMT siblings (itself included)} from sysfs ``thread_siblings_list``; empty where unavailable."""
    out: dict[int, tuple[int, ...]] = {}
    for p in sorted(Path(root).glob("cpu[0-9]*"), key=lambda q: int(q.name[3:])):
        f = p / "topology" / "thread_siblings_list"
        try:
            out[int(p.name[3:])] = tuple(parse_cpu_list(f.read_text()))
        except (OSError, ValueError):
            continue
    return out


def pin_order(cpus, siblings: dict | None = None) -> list[int]:
    """The CPUs in the order in which a p-thread run takes them: the lowest-numbered thread context of
    every physical core first (cores in order of that CPU), then the second context of every core, and
    so on. Without topology information the CPUs are taken in numerical order."""
    cpus = sorted(cpus)
    if not siblings:
        return cpus
    have = set(cpus)

    def key(c):
        group = sorted(s for s in siblings.get(c, (c,)) if s in have) or [c]
        return group.index(c), group[0]

    return sorted(cpus, key=lambda c: (key(c), c))


def pin_cpus(p: int, affinity, siblings: dict | None = None) -> list[int]:
    """The CPUs a run with ``p`` threads is pinned to."""
    return pin_order(affinity, siblings)[: max(1, int(p))]


# --------------------------------------------------------------------------- thread settings
_BASE_TOOL = {"crest": "crest", "crest-ooc": "crest", "scanpy": "scanpy", "scanpy-lean": "scanpy"}


def base_tool(tool: str) -> str:
    return _BASE_TOOL[tool]


# A variant changes which thread settings a tool gets at p threads. "default" is what a user who only
# sets the thread count gets (every variable = p). The others separate the causes of a slow high-thread
# regime: BLAS / OpenMP oversubscription, too many numba or rayon workers, the numba threading layer.
# Both tools get the same number of variants, i.e. the same tuning budget.
VARIANTS = {
    "scanpy": {
        "default": {},
        "blas1": {"blas_threads": 1, "omp_threads": 1},
        "numba_cap8": {"numba_threads": 8},
        "workqueue": {"numba_layer": "workqueue"},
    },
    "crest": {
        "default": {},
        "blas1": {"blas_threads": 1, "omp_threads": 1},
        "rayon_cap8": {"rayon_threads": 8},
        "rayon_half": {"rayon_threads": "half"},
    },
}
DEFAULT_VARIANT = "default"
# The pilot keeps the default unless another variant is at least this much faster (a larger margin
# than the run-to-run noise of one pilot repeat).
TUNING_MARGIN = 0.05


def variant_settings(tool: str, variant: str, p: int) -> dict:
    """Resolved thread settings of ``tool`` at ``p`` threads under ``variant``."""
    base = base_tool(tool)
    if variant not in VARIANTS[base]:
        raise KeyError(f"unknown variant {variant!r} for {tool}")
    over = VARIANTS[base][variant]
    rayon = over.get("rayon_threads", p)
    rayon = max(1, p // 2) if rayon == "half" else min(p, rayon)
    s = {"threads": p, "rayon_threads": rayon, "omp_threads": over.get("omp_threads", p),
         "blas_threads": over.get("blas_threads", p), "numba_threads": min(p, over.get("numba_threads", p)),
         "numba_layer": over.get("numba_layer"), "n_jobs": p}
    return s


def settings_env(s: dict) -> dict[str, str]:
    """Environment variables for a resolved settings dict."""
    env = {"RAYON_NUM_THREADS": str(s["rayon_threads"]), "OMP_NUM_THREADS": str(s["omp_threads"]),
           "OPENBLAS_NUM_THREADS": str(s["blas_threads"]), "MKL_NUM_THREADS": str(s["blas_threads"]),
           "VECLIB_MAXIMUM_THREADS": str(s["blas_threads"]), "NUMBA_NUM_THREADS": str(s["numba_threads"])}
    if s.get("numba_layer"):
        env["NUMBA_THREADING_LAYER"] = s["numba_layer"]
    return env


def tuned_variant(tuning: dict | None, tool: str, p: int) -> str:
    """The variant the tuning pilot chose for ``tool`` at ``p`` threads (default when there is none)."""
    if not tuning:
        return DEFAULT_VARIANT
    return (tuning.get("variant", {}).get(base_tool(tool), {}) or {}).get(str(p), DEFAULT_VARIANT)


def choose_tuning(records: list[dict], margin: float = TUNING_MARGIN) -> dict:
    """Pick, from untimed pilot runs, each tool's variant per thread count and best thread count per dataset.

    ``records``: dicts with tool, dataset, threads, variant, seconds (None = failed). A variant is eligible
    at (tool, p) only if it completed on every pilot dataset that has any completed run at that point. Its
    score is the mean over datasets of seconds / (fastest eligible variant's seconds on that dataset). The
    lowest score wins, but the default keeps the place unless the winner is faster by more than ``margin``
    (score ratio). ``best_threads[tool][dataset]`` is the thread count with the lowest time under the
    chosen variants.
    """
    by: dict = {}
    for r in records:
        if r.get("seconds") is None:
            continue
        by.setdefault((base_tool(r["tool"]), int(r["threads"])), {}).setdefault(r["variant"], {})[r["dataset"]] = float(r["seconds"])
    chosen: dict = {}
    scores: dict = {}
    for (tool, p), per_var in by.items():
        datasets = sorted({d for v in per_var.values() for d in v})
        eligible = {v: t for v, t in per_var.items() if all(d in t for d in datasets)}
        if not eligible:
            continue
        fastest = {d: min(t[d] for t in eligible.values()) for d in datasets}
        score = {v: sum(t[d] / fastest[d] for d in datasets) / len(datasets) for v, t in eligible.items()}
        best = min(score, key=lambda v: (score[v], v != DEFAULT_VARIANT))
        pick = DEFAULT_VARIANT if DEFAULT_VARIANT in score and score[DEFAULT_VARIANT] <= (1 + margin) * score[best] else best
        chosen.setdefault(tool, {})[str(p)] = pick
        scores[(tool, p)] = {v: round(s, 4) for v, s in sorted(score.items())}
    best_threads: dict = {}
    for tool, per_p in chosen.items():
        times: dict = {}
        for p_s, v in per_p.items():
            for d, t in by[(tool, int(p_s))][v].items():
                times.setdefault(d, {})[int(p_s)] = t
        for d, tp in times.items():
            best_threads.setdefault(tool, {})[d] = min(tp, key=lambda q: (tp[q], q))
    runs_per_tool: dict = {}
    for r in records:
        runs_per_tool[base_tool(r["tool"])] = runs_per_tool.get(base_tool(r["tool"]), 0) + 1
    return {"variant": chosen, "best_threads": best_threads, "scores": {f"{t}@{p}": s for (t, p), s in scores.items()},
            "pilot_runs_per_tool": runs_per_tool, "margin": margin}


def tuning_budget_is_equal() -> bool:
    """Both tools get the same number of variants (the same tuning budget)."""
    return len({len(v) for v in VARIANTS.values()}) == 1


# --------------------------------------------------------------------------- run tags and result files
CONTROLS = ("unpinned", "default-settings")


def run_tag(tool: str, dataset: str, profile: str, threads: int, repeat: int, control: str | None = None,
            pilot_variant: str | None = None) -> str:
    """File stem of a run's result files; control runs (never part of the headline) and tuning-pilot runs
    carry a suffix."""
    tag = f"{tool}__{dataset}__{profile}__t{threads}__r{repeat}"
    if control:
        tag += f"__ctl-{control}"
    if pilot_variant:
        tag += f"__pilot-{pilot_variant}"
    return tag


def job_tag(job: dict) -> str:
    return run_tag(job["tool"], job["dataset"], job["profile"], job["threads"], job["repeat"], job.get("control"),
                   job.get("pilot_variant"))


def required_files(job: dict, status: str | None = "ok") -> list[str]:
    """Result files (relative to ``raw/``) that must exist for a planned run: its JSON always (a failed
    run leaves one saying why); for a run that finished ok also its timeline, and for the run that saves
    outputs the per-cell outputs."""
    tag = job_tag(job)
    files = [f"{tag}.json"]
    if status == "ok":
        files.append(f"{tag}.timeline.csv")
        if job.get("save") and job["repeat"] == 0:
            files.append(f"{tag}.outputs.npz")
    return files


def missing_result_files(res: Path) -> list[str]:
    """Required result files of ``res/plan.json``'s runs that are not on disk. ``plan.json`` itself missing
    is reported as one entry (the expectation cannot be checked)."""
    plan = Path(res) / "plan.json"
    if not plan.exists():
        return ["plan.json"]
    raw = Path(res) / "raw"
    missing = []
    for job in json.loads(plan.read_text()).get("runs", []):
        jf = raw / f"{job_tag(job)}.json"
        status = None
        if jf.exists():
            try:
                status = json.loads(jf.read_text()).get("status")
            except ValueError:
                status = None
        for f in required_files(job, status):
            if not (raw / f).exists():
                missing.append(f)
    return missing


# --------------------------------------------------------------------------- quiet host
QUIET_MAX_FOREIGN_CORES = 1.0   # per-run median of foreign busy cores must stay below this
QUIET_MAX_RERUNS = 1            # a violating run is re-run once


def foreign_busy_cores(core_util_percent, proc_cores: float) -> float:
    """Busy cores not due to this process: the sum of every logical core's utilisation (percent / 100)
    minus the cores the process itself used in the same interval, floored at 0."""
    return max(0.0, float(sum(core_util_percent)) / 100.0 - float(proc_cores))


def quiet_decision(violated: bool, n_attempts: int, max_reruns: int = QUIET_MAX_RERUNS) -> str:
    """``accept`` a quiet run; ``rerun`` a violating one while re-runs are left; otherwise ``flag`` it.
    ``n_attempts`` counts the completed attempts of this cell including the one just judged."""
    if not violated:
        return "accept"
    return "rerun" if n_attempts <= max_reruns else "flag"


def end_to_end_seconds(startup_seconds: float | None, total_seconds: float | None) -> float | None:
    """Primary endpoint: process start to the last pipeline step (imports and the timed pipeline, GC and
    result extraction included; the in-process JIT warm-up and the untimed accuracy extras excluded)."""
    if startup_seconds is None or total_seconds is None:
        return None
    return float(startup_seconds) + float(total_seconds)


def parse_subset_name(name: str):
    """``("<parent>", n_cells)`` for a real-subsample dataset name ``<parent>_sub<N>`` (N: digits with an
    optional k / m suffix), else None."""
    m = re.fullmatch(r"(?P<parent>.+)_sub(?P<n>\d+)(?P<u>[km]?)", name)
    if not m:
        return None
    return m["parent"], int(m["n"]) * {"": 1, "k": 1000, "m": 1_000_000}[m["u"]]


def subset_name(parent: str, n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{parent}_sub{n // 1_000_000}m"
    if n % 1000 == 0:
        return f"{parent}_sub{n // 1000}k"
    return f"{parent}_sub{n}"
