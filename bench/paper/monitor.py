"""Resource monitoring for one benchmark process.

A background thread samples, every ``interval`` seconds:

* process RSS (including child processes),
* process CPU time (user + system) -> instantaneous cores in use,
* system-wide utilisation of every logical core,
* every core's clock frequency (Linux sysfs / psutil; empty where unavailable),
* package energy from Intel/AMD RAPL (``/sys/class/powercap``) when readable,

and tags each sample with the pipeline step running at that moment. Steps get
wall time, CPU time, effective parallelism (CPU / wall), peak and delta RSS,
context switches, I/O bytes and energy. Everything is plain JSON/CSV.
"""

from __future__ import annotations

import gc
import os
import resource
import threading
import time
from pathlib import Path

import psutil

RAPL = [p for p in Path("/sys/class/powercap").glob("intel-rapl:*") if (p / "energy_uj").exists()
        and ":" not in p.name.split("intel-rapl:")[1]] if Path("/sys/class/powercap").exists() else []


def _energy_uj() -> float | None:
    tot = 0.0
    try:
        for p in RAPL:
            tot += float((p / "energy_uj").read_text())
        return tot if RAPL else None
    except OSError:
        return None


def _freqs() -> list[float]:
    try:
        fr = psutil.cpu_freq(percpu=True)
        return [round(f.current, 1) for f in fr] if fr else []
    except Exception:  # noqa: BLE001
        return []


class Monitor:
    def __init__(self, interval: float = 0.05, freq_every: int = 4):
        self.proc = psutil.Process(os.getpid())
        self.interval = interval
        self.freq_every = freq_every
        self.samples: list[dict] = []
        self.step = "setup"
        self.steps: dict[str, dict] = {}
        self._stop = threading.Event()
        self._peak = 0
        psutil.cpu_percent(percpu=True)  # prime
        self.t0 = time.perf_counter()
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()

    def _rss(self) -> int:
        r = self.proc.memory_info().rss
        for c in self.proc.children(recursive=True):
            try:
                r += c.memory_info().rss
            except psutil.Error:
                pass
        return r

    def _run(self):
        last_cpu = sum(self.proc.cpu_times()[:2])
        last_t = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            time.sleep(self.interval)
            now = time.perf_counter()
            ct = self.proc.cpu_times()
            cpu = ct.user + ct.system
            rss = self._rss()
            self._peak = max(self._peak, rss)
            s = {"t": round(now - self.t0, 4), "step": self.step, "rss_gb": round(rss / 1e9, 4),
                 "proc_cores": round((cpu - last_cpu) / max(now - last_t, 1e-9), 3),
                 "core_util": psutil.cpu_percent(percpu=True)}
            if k % self.freq_every == 0:
                s["core_mhz"] = _freqs()
            self.samples.append(s)
            last_cpu, last_t, k = cpu, now, k + 1

    # ------------------------------------------------------------------ steps
    class _Step:
        def __init__(self, mon: "Monitor", name: str):
            self.m, self.name = mon, name

        def __enter__(self):
            gc.collect()
            m = self.m
            m.step = self.name
            m._peak = m._rss()
            self.rss0 = m._peak
            self.ct0 = m.proc.cpu_times()
            self.cs0 = m.proc.num_ctx_switches()
            try:
                self.io0 = m.proc.io_counters()
            except (psutil.Error, AttributeError):
                self.io0 = None
            self.e0 = _energy_uj()
            self.t = time.perf_counter()
            return self

        def __exit__(self, et, ev, tb):
            m = self.m
            wall = time.perf_counter() - self.t
            ct = m.proc.cpu_times()
            cs = m.proc.num_ctx_switches()
            rec = {"seconds": wall, "cpu_user_s": ct.user - self.ct0.user, "cpu_sys_s": ct.system - self.ct0.system,
                   "peak_rss_gb": max(m._peak, m._rss()) / 1e9, "start_rss_gb": self.rss0 / 1e9,
                   "ctx_voluntary": cs.voluntary - self.cs0.voluntary,
                   "ctx_involuntary": cs.involuntary - self.cs0.involuntary, "ok": et is None}
            rec["parallelism"] = (rec["cpu_user_s"] + rec["cpu_sys_s"]) / wall if wall > 0 else None
            if self.io0 is not None:
                io = m.proc.io_counters()
                rec["read_mb"] = (io.read_bytes - self.io0.read_bytes) / 1e6
                rec["write_mb"] = (io.write_bytes - self.io0.write_bytes) / 1e6
            e1 = _energy_uj()
            if self.e0 is not None and e1 is not None and e1 >= self.e0:
                rec["energy_j"] = (e1 - self.e0) / 1e6
            if et is not None:
                rec["error"] = f"{et.__name__}: {ev}"
            m.steps[self.name] = rec
            m.step = "between"
            return False

    def track(self, name: str) -> "Monitor._Step":
        return Monitor._Step(self, name)

    def stop(self) -> dict:
        self._stop.set()
        self._th.join()
        ru = resource.getrusage(resource.RUSAGE_SELF)
        maxrss = ru.ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)
        return {"ru_maxrss_gb": maxrss / 1e9, "ru_utime_s": ru.ru_utime, "ru_stime_s": ru.ru_stime,
                "ru_majflt": ru.ru_majflt, "ru_minflt": ru.ru_minflt, "ru_nvcsw": ru.ru_nvcsw,
                "ru_nivcsw": ru.ru_nivcsw, "wall_s": time.perf_counter() - self.t0}

    def write_timeline(self, path: Path) -> None:
        import csv
        n_cores = max((len(s["core_util"]) for s in self.samples), default=0)
        n_freq = max((len(s.get("core_mhz", [])) for s in self.samples), default=0)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "step", "rss_gb", "proc_cores"] + [f"util_{i}" for i in range(n_cores)]
                       + [f"mhz_{i}" for i in range(n_freq)])
            for s in self.samples:
                mhz = s.get("core_mhz", [])
                w.writerow([s["t"], s["step"], s["rss_gb"], s["proc_cores"]] + s["core_util"]
                           + (mhz + [""] * (n_freq - len(mhz)) if mhz else [""] * n_freq))
