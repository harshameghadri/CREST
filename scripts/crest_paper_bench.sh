#!/usr/bin/env bash
# CREST paper benchmark - the whole thing, on your own machine, in one command.
#
#   cd /mnt/scratch                                # any roomy disk: everything is written below the current directory
#   curl -LO https://raw.githubusercontent.com/harshameghadri/CREST/dev/scripts/crest_paper_bench.sh
#   bash crest_paper_bench.sh                      # standard tier (~10-20 h on 64 threads; run count + estimate printed first)
#   bash crest_paper_bench.sh --tier quick         # 3 small datasets, 2 repeats, short thread scan (~1 h)
#   bash crest_paper_bench.sh --tier full          # adds the 647k COVID atlas, 1.3M neurons, 500k/1M synthetic
#   bash crest_paper_bench.sh --tier smoke         # PBMC 3k + a tiny synthetic series: checks the harness (minutes)
#
# Nothing is written to $HOME: the source, venv, data, results and every cache (uv packages and
# Pythons, cargo registry, numba, matplotlib, temp files) live under --workdir, which defaults to
# ./crest-bench in the directory you run the script from.
#
# What it does (every step logged under $WORKDIR/results/<host>-<date>/):
#   1. preflight: git, curl, uv (installs it with --install-uv), Rust (installs with --install-rust), disk, RAM
#   2. clones harshameghadri/CREST at --ref into a throwaway uv venv, builds CREST in release mode, and
#      installs the pinned benchmark environment (bench/paper/requirements-bench.txt: scanpy 1.11.x etc.)
#   3. runs the test suites (cargo test, pytest incl. scanpy and R-DESeq2 parity); stops if they fail
#   4. records the machine: CPU model/cache/governor/turbo, cores, RAM, OS, compilers, BLAS, every package
#   5. downloads and converts the datasets (resumable, size-checked, SHA-256 recorded)
#   6. runs every (dataset x tool x repeat) in its own process, interleaved and shuffled: the core
#      workflow of every dataset at every --main-threads count (CREST and scanpy; matched and best-thread
#      comparisons), out-of-core CREST at --ooc-threads, separate runs of the optional modules, and a
#      full thread scan of one dataset; per step: wall and CPU time, parallelism, peak (absolute and
#      minus baseline) and timeline memory, per-core utilisation and clock, context switches, I/O, energy
#      (RAPL, if readable); OOM/timeouts recorded as failures; every attempt appended to attempts.jsonl
#   7. agreement with scanpy (ARI/NMI, the two tools' actual kNN graphs, kNN recall vs brute force,
#      same-graph Leiden + igraph modularity, PCA subspace, HVG, DE markers, UMAP trustworthiness, ...)
#   8. module benchmarks: DESeq2 vs R/pydeseq2 (thread-matched), Harmony vs harmonypy, Scrublet vs scanpy
#      (demuxlet truth), label transfer vs scanpy.tl.ingest, 20-seed Leiden modularity vs leidenalg
#   9. statistics (medians, min-max, IQR, speedups at every matched thread count and best vs best,
#      Mann-Whitney, Hodges-Lehmann, scaling exponents on the synthetic series, Amdahl fits up to the
#      peak), tables (CSV + LaTeX), figures (PDF + PNG), report.md, and a .tar.gz
#  10. copies the report, tables, figures, module results and environment record into
#      <workdir>/CREST/bench/paper/results/<host>-<date>/ (with commit, host and versions), ready to
#      commit; see bench/paper/results/README.md
#
# Options:
#   --workdir DIR          everything goes here                      (default: ./crest-bench)
#   --ref REF              branch/tag/commit to benchmark            (default: dev)
#   --repo URL             repository                                (default: https://github.com/harshameghadri/CREST)
#   --tier T               smoke | quick | standard | full           (default: standard)
#   --datasets "a b"       explicit dataset list (overrides --tier; see --list-datasets)
#   --repeats K            repeats per configuration                 (default: 5; 2 in the quick/smoke tiers)
#   --threads "1 2 4 8"    thread counts for the strong-scaling scan (default: powers of two up to all cores;
#                          quick tier: 1, 8 and all cores; smoke: 1, 2, 4)
#   --thread-dataset NAME  dataset for the thread scan               (default: pbmc68k; pbmc10k in the quick tier)
#   --main-threads "8 64"  thread counts of the main core-workflow runs of CREST and scanpy, on EVERY dataset
#                          (default: standard/full 8 16 32 64 (those <= all cores, plus all cores);
#                          quick: 8 and all cores; smoke: 2 and 4)
#   --ooc-threads "8 64"   thread counts of the out-of-core CREST runs (default: 8 and all cores)
#   --module-repeats K     repeats of the optional-module runs       (default: 3; 2 in the quick/smoke tiers)
#   --scrublet-max-cells N skip Scrublet in module runs above N cells (default: 150000; 0 = never skip)
#   --scanpy-max-cells N   skip scanpy above N cells (0 = never; default: from RAM, ~12.5k cells per GB)
#   --timeout SEC          per-run limit                             (default: 7200)
#   --python X.Y           Python for the venv                       (default: 3.11)
#   --profile P            full = core workflow + separate module runs; core = core only (default: full)
#   --no-pin               do not pin thread-scan runs to cores
#   --skip-tests / --skip-modules / --skip-thread-scan / --skip-build
#   --install-uv           install uv if missing (https://astral.sh/uv)
#   --install-rust         install Rust via rustup if missing
#   --list-datasets        print the dataset registry and exit
#   --dry-run              print the run plan and exit
#
# Parse PBMCs (--datasets "parse_pbmc_1m parse_pbmc", or --tier full; 9.7M cells, and a 1M random
# subset for the scanpy comparison): put the h5ad from
# https://figshare.com/articles/dataset/pbmc_parse/28589774 in <workdir>/data/ (as
# Parse_1M_adata_for_cellflow_datasets_with_embeddings.h5ad or its original name); it is used from
# there. Its layout can be checked with: python bench/paper/datasets.py --inspect FILE.h5ad
#
# Requirements: Linux or macOS, git, curl, a C toolchain (for Rust), ~10 GB disk (standard) or ~60 GB (full;
# +~20 GB for the converted Parse data).
# Optional: R with DESeq2 (for the R comparison): R -e 'BiocManager::install("DESeq2")' + jsonlite.

set -euo pipefail

WORKDIR="$PWD/crest-bench"
REF="dev"
REPO="https://github.com/harshameghadri/CREST"
TIER="standard"
DATASETS=""
REPEATS=""
THREADS=""
THREAD_DS=""
SCANPY_MAX=""
TIMEOUT=7200
PYVER="3.11"
PROFILE="full"
MAIN_THREADS=""
OOC_THREADS=""
MODULE_REPEATS=""
SCRUBLET_MAX=150000
NO_PIN=0; SKIP_TESTS=0; SKIP_MODULES=0; SKIP_SCAN=0; SKIP_BUILD=0; INSTALL_UV=0; INSTALL_RUST=0
LIST=0; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workdir) WORKDIR="$2"; shift 2 ;;
    --ref) REF="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --tier) TIER="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --thread-dataset) THREAD_DS="$2"; shift 2 ;;
    --scanpy-max-cells) SCANPY_MAX="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --python) PYVER="$2"; shift 2 ;;
    --profile) PROFILE="$2"; shift 2 ;;
    --main-threads) MAIN_THREADS="$2"; shift 2 ;;
    --ooc-threads) OOC_THREADS="$2"; shift 2 ;;
    --module-repeats) MODULE_REPEATS="$2"; shift 2 ;;
    --scrublet-max-cells) SCRUBLET_MAX="$2"; shift 2 ;;
    --no-pin) NO_PIN=1; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --skip-modules) SKIP_MODULES=1; shift ;;
    --skip-thread-scan) SKIP_SCAN=1; shift ;;
    --skip-build) SKIP_BUILD=1; shift ;;
    --install-uv) INSTALL_UV=1; shift ;;
    --install-rust) INSTALL_RUST=1; shift ;;
    --list-datasets) LIST=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) echo "unknown option: $1 (see --help)"; exit 2 ;;
  esac
done

# Keep every cache and install under the work directory (the home disk may be small).
if ! mkdir -p "$WORKDIR" 2>/dev/null || [[ ! -w "$WORKDIR" ]]; then
  printf '\033[31mERROR: cannot write to %s\033[0m\n' "$WORKDIR" >&2
  echo "  Run the script from a directory you own on a big disk, or pass --workdir DIR." >&2
  echo "  Here: $(df -h "$(dirname "$WORKDIR")" 2>/dev/null | awk 'NR==2 {print $1" mounted on "$6", "$4" free"}')" >&2
  echo "  (if this is a mount point, check that the disk is actually mounted: findmnt $(dirname "$WORKDIR"))" >&2
  exit 1
fi
WORKDIR="$(cd "$WORKDIR" && pwd)"
C="$WORKDIR/cache"
mkdir -p "$C"/{uv,uv-python,cargo,numba,mpl,xdg,tmp}
export UV_CACHE_DIR="$C/uv" UV_PYTHON_INSTALL_DIR="$C/uv-python" UV_INSTALL_DIR="$WORKDIR/bin"
export NUMBA_CACHE_DIR="$C/numba" MPLCONFIGDIR="$C/mpl" XDG_CACHE_HOME="$C/xdg" TMPDIR="$C/tmp"
export PATH="$WORKDIR/bin:$PATH"

bold() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
die() { printf '\033[31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
OS="$(uname -s)"
NCPU="$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.logicalcpu )"
if [[ "$OS" == "Darwin" ]]; then RAM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 )); else RAM_GB=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); fi

# ------------------------------------------------------------------ 1. preflight
bold "1. Preflight ($OS, $NCPU logical cores, ${RAM_GB} GB RAM)"
command -v git >/dev/null || die "git is required"
command -v curl >/dev/null || die "curl is required"
if ! command -v uv >/dev/null; then
  if [[ $INSTALL_UV -eq 1 ]]; then
    curl -LsSf https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh   # into $WORKDIR/bin
  else
    die "uv is required: curl -LsSf https://astral.sh/uv/install.sh | sh   (or re-run with --install-uv)"
  fi
fi
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
[[ -x "$WORKDIR/rust/cargo/bin/cargo" ]] && export RUSTUP_HOME="$WORKDIR/rust/rustup" PATH="$WORKDIR/rust/cargo/bin:$PATH"
if ! command -v cargo >/dev/null; then
  if [[ $INSTALL_RUST -eq 1 ]]; then
    export RUSTUP_HOME="$WORKDIR/rust/rustup" CARGO_HOME="$WORKDIR/rust/cargo"   # not in $HOME
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --no-modify-path
    export PATH="$CARGO_HOME/bin:$PATH"
  else
    die "Rust is required: curl https://sh.rustup.rs -sSf | sh   (or re-run with --install-rust)"
  fi
fi
# crate downloads for the build go under the work directory too (the toolchain stays where it is)
export CARGO_HOME="$C/cargo"
FREE_GB=$(df -Pk "$WORKDIR" | awk 'NR==2 {print int($4/1048576)}')
NEED_GB=$([[ "$TIER" == "full" ]] && echo 60 || ([[ "$TIER" == "quick" || "$TIER" == "smoke" ]] && echo 3 || echo 12))
echo "  free disk in $WORKDIR: ${FREE_GB} GB (tier '$TIER' needs ~${NEED_GB} GB)"
[[ $FREE_GB -ge $NEED_GB ]] || die "not enough disk space"
if [[ -z "$SCANPY_MAX" ]]; then SCANPY_MAX=$(( RAM_GB * 12500 )); fi
SMALL=0; [[ "$TIER" == "quick" || "$TIER" == "smoke" ]] && SMALL=1
if [[ -z "$REPEATS" ]]; then REPEATS=$([[ $SMALL -eq 1 ]] && echo 2 || echo 5); fi
if [[ -z "$MODULE_REPEATS" ]]; then MODULE_REPEATS=$([[ $SMALL -eq 1 ]] && echo 2 || echo 3); fi
clip() { echo "$@" | tr ' ' '\n' | awk -v n="$NCPU" 'NF && $1<=n' | sort -nu | xargs; }
if [[ -z "$MAIN_THREADS" ]]; then
  case "$TIER" in
    smoke) MAIN_THREADS="$(clip 2 4)" ;;
    quick) MAIN_THREADS="$(clip 8 "$NCPU")" ;;
    *)     MAIN_THREADS="$(clip 8 16 32 64 "$NCPU")" ;;
  esac
fi
if [[ -z "$OOC_THREADS" ]]; then
  if [[ "$TIER" == "smoke" ]]; then OOC_THREADS="$(clip 4)"; else OOC_THREADS="$(clip 8 "$NCPU")"; fi
fi
echo "  scanpy is skipped above $SCANPY_MAX cells (--scanpy-max-cells 0 disables the cap)"

STAMP="$(hostname -s 2>/dev/null || hostname)-$(date +%Y%m%d-%H%M)"
OUT="$WORKDIR/results/$STAMP"
mkdir -p "$OUT"/{env,logs,modules}
exec > >(tee -a "$OUT/logs/driver.log") 2>&1
echo "  results: $OUT"

# ------------------------------------------------------------------ 2. code + environment
bold "2. CREST source ($REPO @ $REF) and venv"
SRC="$WORKDIR/CREST"
if [[ -d "$SRC/.git" ]]; then
  git -C "$SRC" fetch --tags origin
else
  git clone "$REPO" "$SRC"
fi
git -C "$SRC" checkout -q "$REF"
git -C "$SRC" pull -q --ff-only origin "$REF" 2>/dev/null || true
COMMIT="$(git -C "$SRC" rev-parse HEAD)"
echo "  commit $COMMIT"
VENV="$WORKDIR/venv-$(echo "$COMMIT" | cut -c1-10)"
PY="$VENV/bin/python"
if [[ $SKIP_BUILD -eq 0 || ! -x "$PY" ]]; then
  rm -rf "$VENV"   # rebuild from scratch (uv refuses to reuse an existing venv)
  uv venv -q --python "$PYVER" "$VENV"
  echo "  building CREST in release mode (a few minutes; log: logs/build.log)"
  uv pip install -q --python "$PY" maturin
  ( cd "$SRC" && VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" maturin develop --release -E test,bench ) \
    > "$OUT/logs/build.log" 2>&1 || { tail -30 "$OUT/logs/build.log"; die "build failed (see logs/build.log)"; }
fi
PINS="$SRC/bench/paper/requirements-bench.txt"
if [[ -f "$PINS" ]]; then
  echo "  pinning the benchmark environment ($PINS)"
  uv pip install -q --python "$PY" -r "$PINS" >> "$OUT/logs/build.log" 2>&1 || die "could not install the pinned versions (logs/build.log)"
  grep -E '^[A-Za-z]' "$PINS" | sed 's/==.*//' | while read -r pkg; do
    want="$(grep -E "^$pkg==" "$PINS" | cut -d= -f3)"
    have="$("$PY" -c "from importlib.metadata import version; print(version('$pkg'))" 2>/dev/null || echo missing)"
    [[ "$want" == "$have" ]] || die "$pkg is $have, pinned $want"
  done
else
  echo "  WARNING: $PINS not found at $REF: the benchmark environment is NOT pinned"
fi
echo "  built: $("$PY" -c 'import crest; print(crest.__file__)')"

# ------------------------------------------------------------------ 3. tests
if [[ $SKIP_TESTS -eq 0 ]]; then
  bold "3. Test suites"
  ( cd "$SRC" && cargo test --release ) > "$OUT/logs/cargo_test.log" 2>&1 || die "cargo test failed (logs/cargo_test.log)"
  ( cd "$SRC" && "$PY" -m pytest -q ) > "$OUT/logs/pytest.log" 2>&1 || die "pytest failed (logs/pytest.log)"
  grep -h "test result\|passed\|failed" "$OUT/logs/cargo_test.log" "$OUT/logs/pytest.log" | tail -3
fi

# ------------------------------------------------------------------ 4. machine record
bold "4. Recording the machine"
E="$OUT/env"
uname -a > "$E/uname.txt"
{ [[ -f /etc/os-release ]] && cat /etc/os-release; sw_vers 2>/dev/null || true; } > "$E/os.txt"
if [[ "$OS" == "Linux" ]]; then
  lscpu > "$E/lscpu.txt" 2>/dev/null || true
  cp /proc/cpuinfo "$E/cpuinfo.txt"; cp /proc/meminfo "$E/meminfo.txt"
  cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor > "$E/governor.txt" 2>/dev/null || echo "n/a" > "$E/governor.txt"
  { cat /sys/devices/system/cpu/intel_pstate/no_turbo 2>/dev/null && echo "(intel no_turbo)"; \
    cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null && echo "(boost)"; } > "$E/turbo.txt" || true
  lsblk -o NAME,ROTA,SIZE,MODEL > "$E/disks.txt" 2>/dev/null || true
else
  sysctl -a 2>/dev/null | grep -E '^(machdep\.cpu|hw\.)' > "$E/sysctl.txt" || true
fi
{ df -h "$WORKDIR"; findmnt -T "$WORKDIR" -o SOURCE,FSTYPE,OPTIONS 2>/dev/null || true; } > "$E/disk_free.txt"
{ rustc -V; cargo -V; "$PY" -V; uv --version; } > "$E/toolchain.txt" 2>&1
uv pip freeze --python "$PY" > "$E/pip_freeze.txt"
echo "$COMMIT" > "$E/crest_commit.txt"
"$PY" - > "$E/python_env.json" <<'PYEOF'
import json, numpy, platform, psutil
from importlib import metadata
from threadpoolctl import threadpool_info
import crest, scanpy  # noqa: F401  (load both tools' thread pools before listing them)
vers = {}
for d in ("crest-sc", "scanpy", "anndata", "numpy", "scipy", "pandas", "polars", "h5py", "scikit-learn", "pynndescent",
          "umap-learn", "numba", "llvmlite", "igraph", "leidenalg", "harmonypy", "pydeseq2", "threadpoolctl"):
    try:
        vers[d] = metadata.version(d)
    except metadata.PackageNotFoundError:
        vers[d] = None
print(json.dumps({"python": platform.python_version(), "numpy": numpy.__version__, "versions": vers,
                  "logical_cores": psutil.cpu_count(), "physical_cores": psutil.cpu_count(logical=False),
                  "ram_gb": psutil.virtual_memory().total / 1e9, "threadpools": threadpool_info()}, indent=1, default=str))
PYEOF
CPU_MODEL="$( (grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2) || sysctl -n machdep.cpu.brand_string)"
cat > "$E/summary.md" <<EOF
**Machine** $(echo $CPU_MODEL | xargs), $NCPU logical cores, ${RAM_GB} GB RAM, $OS ($(uname -r)); governor: $(cat "$E/governor.txt" 2>/dev/null || echo n/a).
**Software** CREST $COMMIT ($REF), Python $("$PY" -c 'import platform;print(platform.python_version())'), $(rustc -V).
**Protocol** core workflow of every dataset at $MAIN_THREADS threads (CREST, scanpy; out-of-core CREST at $OOC_THREADS), $REPEATS repeats per configuration; optional modules in separate runs ($([[ "$PROFILE" == "full" ]] && echo "$MODULE_REPEATS repeats, Scrublet skipped above $SCRUBLET_MAX cells" || echo "not run")); interleaved in shuffled order, fresh process per run, JIT warm-up excluded.
EOF
cat "$E/summary.md"

# ------------------------------------------------------------------ 5. data
bold "5. Datasets"
DATA="$WORKDIR/data"
cd "$SRC"
if [[ $LIST -eq 1 ]]; then "$PY" bench/paper/datasets.py --list; exit 0; fi
if [[ -z "$DATASETS" ]]; then
  DATASETS="$("$PY" -c "import sys; sys.path.insert(0,'bench/paper'); from datasets import tier_datasets; print(' '.join(tier_datasets('$TIER')))")"
fi
echo "  datasets: $DATASETS"
[[ $DRY -eq 1 ]] || "$PY" bench/paper/datasets.py --data-dir "$DATA" --datasets $DATASETS 2>&1 | tee "$OUT/logs/datasets.log"
cp "$DATA/checksums.json" "$E/data_checksums.json" 2>/dev/null || true

# ------------------------------------------------------------------ 6. runs
bold "6. Benchmark runs"
if [[ -z "$THREAD_DS" ]]; then
  if [[ " $DATASETS " == *" pbmc68k "* ]]; then THREAD_DS=pbmc68k
  elif [[ " $DATASETS " == *" pbmc10k "* ]]; then THREAD_DS=pbmc10k
  elif [[ " $DATASETS " == *" pbmc3k "* ]]; then THREAD_DS=pbmc3k
  else THREAD_DS="$(echo $DATASETS | awk '{print $NF}')"; fi
fi
if [[ -z "$THREADS" ]]; then
  if [[ "$TIER" == "smoke" ]]; then THREADS="$(clip 1 2 4)"
  elif [[ "$TIER" == "quick" ]]; then THREADS="1 $(( NCPU < 8 ? NCPU : 8 ))"
  else p=1; THREADS=""; while [[ $p -lt $NCPU ]]; do THREADS="$THREADS $p"; p=$((p*2)); done; fi
fi
echo "  thread scan on $THREAD_DS: threads $(echo $THREADS $NCPU | tr ' ' '\n' | sort -nu | xargs); $REPEATS repeats"
SCAN_ARGS=(--threads-scan $THREADS --thread-dataset "$THREAD_DS")
[[ $SKIP_SCAN -eq 1 ]] && SCAN_ARGS=()
PIN_ARGS=(); [[ $NO_PIN -eq 1 ]] && PIN_ARGS=(--no-pin)
"$PY" bench/paper/run_all.py --data-dir "$DATA" --out "$OUT" --datasets $DATASETS --repeats "$REPEATS" \
  --profile "$PROFILE" --main-threads $MAIN_THREADS --ooc-threads $OOC_THREADS --module-repeats "$MODULE_REPEATS" \
  --scrublet-max-cells "$SCRUBLET_MAX" \
  --scanpy-max-cells "$SCANPY_MAX" --timeout "$TIMEOUT" ${SCAN_ARGS[@]+"${SCAN_ARGS[@]}"} ${PIN_ARGS[@]+"${PIN_ARGS[@]}"} \
  $([[ $DRY -eq 1 ]] && echo --dry-run)
[[ $DRY -eq 1 ]] && exit 0

# ------------------------------------------------------------------ 7. accuracy
bold "7. Agreement with scanpy"
"$PY" bench/paper/accuracy.py "$OUT/raw" --out "$OUT/accuracy.json" 2>&1 | tee "$OUT/logs/accuracy.log"

# ------------------------------------------------------------------ 8. modules
if [[ $SKIP_MODULES -eq 0 ]]; then
  bold "8. Module benchmarks"
  M="$OUT/modules"
  KANG="$DATA/src/kang"
  run_mod() {  # name, command...
    local name="$1"; shift
    echo "  - $name"
    if "$@" > "$OUT/logs/module_$name.log" 2>&1; then echo "    ok"; else echo "    FAILED (logs/module_$name.log)"; fi
  }
  R_ARGS=(); command -v Rscript >/dev/null && Rscript -e 'suppressMessages(library(DESeq2))' 2>/dev/null || R_ARGS=(--skip-r)
  [[ ${#R_ARGS[@]} -gt 0 ]] && echo "  (R DESeq2 not found: DESeq2 is compared with pydeseq2 only)"
  # thread-matched DESeq2: 1, 8 and 64 threads (all cores instead of 64 on smaller machines)
  DE_THREADS="$(clip 1 8 64 $([[ $NCPU -lt 64 ]] && echo "$NCPU"))"
  run_mod deseq2_kang "$PY" bench/deseq2/kang_pseudobulk.py --data "$KANG" --out "$M/deseq2" --threads $DE_THREADS ${R_ARGS[@]+"${R_ARGS[@]}"}
  if [[ ${#R_ARGS[@]} -eq 0 ]]; then run_mod deseq2_simulated "$PY" bench/deseq2/compare_r.py --out "$M/deseq2"; fi
  run_mod harmony "$PY" bench/harmony/compare_harmonypy.py --data "$KANG" --out "$M/harmony"
  run_mod scrublet "$PY" bench/doublets/compare_scrublet.py --data "$KANG" --out "$M/doublets"
  run_mod ingest "$PY" bench/ingest/compare_scanpy_ingest.py --data "$KANG" --out "$M/ingest"
  # 20-seed Leiden modularity vs leidenalg on one graph (the docs' "+0.0015 over 20 seeds" check)
  SWEEP_DS=pbmc68k; [[ -f "$DATA/$SWEEP_DS.h5" ]] || SWEEP_DS="$THREAD_DS"
  run_mod modularity_sweep "$PY" bench/paper/modularity_sweep.py --data "$DATA/$SWEEP_DS.h5" --out "$M/modularity" --seeds 20
  for f in "$M"/*/*.md; do [[ -f "$f" ]] && cp "$f" "$M/$(basename "$(dirname "$f")")_$(basename "$f")"; done
fi

# ------------------------------------------------------------------ 9. statistics, figures, archive
bold "9. Statistics, tables, figures"
"$PY" bench/paper/summarize.py "$OUT" 2>&1 | tee "$OUT/logs/summarize.log"
# ------------------------------------------------------------------ 10. committable copy
bold "10. Committable results"
KEEP="$SRC/bench/paper/results/$STAMP"
mkdir -p "$KEEP"/{tables,figures,modules,environment}
cp "$OUT/report.md" "$KEEP/" 2>/dev/null || true
for f in accuracy.json plan.json attempts.jsonl; do [[ -f "$OUT/$f" ]] && cp "$OUT/$f" "$KEEP/"; done
cp "$OUT"/tables/*.csv "$KEEP/tables/" 2>/dev/null || true
cp "$OUT"/figures/*.png "$KEEP/figures/" 2>/dev/null || true
if [[ -d "$OUT/modules" ]]; then
  ( cd "$OUT/modules" && find . -type f \( -name '*.md' -o -name '*.csv' -o -name '*.json' \) -size -5M \
      -exec sh -c 'mkdir -p "$1/$(dirname "$2")" && cp "$2" "$1/$2"' _ "$KEEP/modules" {} \; )
fi
for f in summary.md python_env.json pip_freeze.txt toolchain.txt crest_commit.txt uname.txt os.txt governor.txt \
         turbo.txt lscpu.txt data_checksums.json; do [[ -f "$E/$f" ]] && cp "$E/$f" "$KEEP/environment/"; done
"$PY" - "$KEEP/meta.json" <<PYEOF
import json, sys
json.dump({"stamp": "$STAMP", "host": "$(hostname -s 2>/dev/null || hostname)", "commit": "$COMMIT", "ref": "$REF",
           "repo": "$REPO", "tier": "$TIER", "datasets": "$DATASETS".split(), "repeats": $REPEATS,
           "main_threads": "$MAIN_THREADS".split(), "ooc_threads": "$OOC_THREADS".split(),
           "thread_scan": "$THREADS".split(), "thread_dataset": "$THREAD_DS", "profile": "$PROFILE",
           "versions": json.load(open("$E/python_env.json")).get("versions")}, open(sys.argv[1], "w"), indent=1)
PYEOF
echo "  $KEEP  ($(du -sh "$KEEP" | cut -f1); commit it as described in bench/paper/results/README.md)"
TAR="$WORKDIR/results/crest-bench-$STAMP.tar.gz"
tar -czf "$TAR" -C "$WORKDIR/results" --exclude='*.outputs.npz' --exclude='_ooc_*' "$STAMP"
bold "Done"
echo "  report:  $OUT/report.md"
echo "  figures: $OUT/figures/"
echo "  archive: $TAR  (send this back; it has everything except the large per-cell outputs)"
echo "  to commit: cd $SRC && git checkout -b bench/results-$STAMP && git add bench/paper/results/$STAMP && git commit -m 'bench: results $STAMP'"
