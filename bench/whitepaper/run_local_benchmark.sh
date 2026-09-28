#!/usr/bin/env bash
# CREST vs scanpy white-paper benchmark - run everything on this machine.
#
#   bash bench/whitepaper/run_local_benchmark.sh                       # defaults
#   bash bench/whitepaper/run_local_benchmark.sh --sizes "100000 200000 500000" --repeats 3
#
# Steps: build CREST from this checkout into a fresh venv, run the test suite
# (incl. scanpy parity), download the real 10x data, build the scaled-up
# datasets, then run every (dataset x tool x repeat) in its own process and
# write tables + figures to $OUT/report. A run that dies (e.g. the OS
# out-of-memory killer) is recorded as a failure and the benchmark continues.
#
# Options:
#   --data-dir DIR      where datasets live           (default: ./bench_data)
#   --out DIR           results directory             (default: ./bench_results/<host>-<date>)
#   --sizes "N ..."     synthetic dataset sizes       (default: "100000 200000")
#   --tools "T ..."     crest crest-ooc scanpy        (default: all three)
#   --repeats K         repeats per run               (default: 3)
#   --scanpy-max-cells N  skip scanpy above N cells (useful on macOS, which swaps instead of OOM-killing)
#   --skip-install      reuse the existing venv
#   --skip-tests        do not run pytest first

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="$REPO/bench_data"
OUT="$REPO/bench_results/$(hostname -s)-$(date +%Y%m%d-%H%M)"
SIZES="100000 200000"
TOOLS="crest crest-ooc scanpy"
REPEATS=3
SCANPY_MAX=0
SKIP_INSTALL=0
SKIP_TESTS=0
VENV="$REPO/.venv-bench"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --sizes) SIZES="$2"; shift 2 ;;
    --tools) TOOLS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --scanpy-max-cells) SCANPY_MAX="$2"; shift 2 ;;
    --skip-install) SKIP_INSTALL=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option $1"; exit 2 ;;
  esac
done

log() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
mkdir -p "$DATA_DIR" "$OUT"
cd "$REPO"

# ---------------------------------------------------------------- environment
if [[ $SKIP_INSTALL -eq 0 ]]; then
  log "Checking toolchain"
  command -v python3 >/dev/null || { echo "python3 (>= 3.10) is required"; exit 1; }
  python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"'
  if ! command -v cargo >/dev/null; then
    echo "Rust is required to build CREST. Install it with:"
    echo "  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && source \$HOME/.cargo/env"
    exit 1
  fi
  log "Creating venv $VENV"
  if command -v uv >/dev/null; then
    uv venv "$VENV" -q --python python3
    PIP="uv pip install -q --python $VENV/bin/python"
  else
    python3 -m venv "$VENV"
    PIP="$VENV/bin/python -m pip install -q"
    $PIP --upgrade pip
  fi
  $PIP maturin
  log "Building CREST (release) and benchmark dependencies"
  (source "$VENV/bin/activate" && maturin develop --release -E bench,test)
fi
source "$VENV/bin/activate"
PY="$VENV/bin/python"

# ---------------------------------------------------------------- sanity
if [[ $SKIP_TESTS -eq 0 ]]; then
  log "Test suite (Rust + Python, incl. scanpy parity)"
  cargo test --release -q 2>&1 | tail -3
  $PY -m pytest -q tests 2>&1 | tail -3
fi

# ---------------------------------------------------------------- machine info
$PY - "$OUT/machine.json" <<'EOF'
import json, os, platform, subprocess, sys, psutil
info = {"host": platform.node(), "os": platform.platform(), "python": platform.python_version(),
        "cores_logical": os.cpu_count(), "cores_physical": psutil.cpu_count(logical=False),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1)}
for cmd in (["lscpu"], ["sysctl", "-n", "machdep.cpu.brand_string"]):
    try:
        info["cpu"] = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()[:2000]
        break
    except Exception:
        pass
import crest, polars, numpy
info["versions"] = {"crest": crest.__version__, "polars": polars.__version__, "numpy": numpy.__version__}
try:
    import scanpy, anndata
    info["versions"].update(scanpy=scanpy.__version__, anndata=anndata.__version__)
except Exception:
    pass
open(sys.argv[1], "w").write(json.dumps(info, indent=2))
print(json.dumps({k: v for k, v in info.items() if k != "cpu"}, indent=2))
EOF

# ---------------------------------------------------------------- data
log "Datasets in $DATA_DIR"
$PY bench/whitepaper/fetch_data.py "$DATA_DIR"
DATASETS="pbmc68k"
for n in $SIZES; do
  name="pbmc_$((n / 1000))k"
  DATASETS="$DATASETS $name"
  if [[ ! -s "$DATA_DIR/$name.h5" ]]; then
    $PY bench/whitepaper/make_dataset.py --src "$DATA_DIR/pbmc10k_v3.h5" --cells "$n" --out "$DATA_DIR/$name.h5"
  fi
done

# ---------------------------------------------------------------- runs
cells_of() { case "$1" in pbmc68k) echo 68579 ;; pbmc_*k) echo $(( ${1//[!0-9]/} * 1000 )) ;; esac; }
for rep in $(seq 1 "$REPEATS"); do
  for d in $DATASETS; do
    for t in $TOOLS; do
      if [[ "$t" == "scanpy" && $SCANPY_MAX -gt 0 && $(cells_of "$d") -gt $SCANPY_MAX ]]; then
        printf '%s.h5\t%s\tskipped (above --scanpy-max-cells %s)\n' "$d" "$t" "$SCANPY_MAX" > "$OUT/FAILED_${t}_${d}.txt"
        continue
      fi
      log "repeat $rep/$REPEATS  $t  $d  $(date +%T)"
      set +e
      $PY bench/whitepaper/run_pipeline.py --tool "$t" --data "$DATA_DIR/$d.h5" --out "$OUT/rep$rep" --repeat "$rep" \
        > "$OUT/log_${t}_${d}_r${rep}.txt" 2>&1
      code=$?
      set -e
      grep -E "^\{|^  " "$OUT/log_${t}_${d}_r${rep}.txt" || true
      if [[ $code -ne 0 ]]; then
        why="exit code $code"
        [[ $code -eq 137 ]] && why="killed by the OS (out of memory)"
        echo "  FAILED: $why (see $OUT/log_${t}_${d}_r${rep}.txt)"
        printf '%s.h5\t%s\t%s\n' "$d" "$t" "$why" > "$OUT/FAILED_${t}_${d}.txt"
      fi
      rm -rf "$OUT/rep$rep/crest_parquet"   # out-of-core scratch copy
    done
  done
done

# ---------------------------------------------------------------- report
log "Report"
$PY bench/whitepaper/summarize.py "$OUT" --out "$OUT/report"
echo
echo "Done. Report: $OUT/report/report.md   Figures: $OUT/report/fig_*.png"
