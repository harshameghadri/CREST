# Committed benchmark results

Each subdirectory `<host>-<YYYYMMDD-HHMM>/` is one run of `scripts/crest_paper_bench.sh`,
copied here by the script's last step so the numbers quoted in the docs and the paper can be
traced to a commit. It holds:

| file | content |
|---|---|
| `meta.json` | commit, ref, host, tier, datasets, repeats, thread counts, package versions |
| `report.md` | the full report (headline table, scaling, thread scaling, agreement, modules, failures) |
| `tables/*.csv` | every table of the report; `runs.csv` has one row per run (all repeats) |
| `figures/*.png` | the report's figures (the PDFs are in the run's `.tar.gz`) |
| `accuracy.json` | agreement with scanpy, per dataset |
| `attempts.jsonl` | every run attempt, including failed ones that a resume later retried |
| `plan.json` | the planned runs, in execution order |
| `modules/` | the module benchmarks (DESeq2 vs R / pydeseq2, Harmony, Scrublet, ingest, modularity sweep) |
| `environment/` | machine and software record (`summary.md`, `python_env.json`, `pip_freeze.txt`, `lscpu.txt`, ...) |

Raw per-run JSON, timelines and per-cell outputs are not committed (too large); they are in
the `crest-bench-<stamp>.tar.gz` archive the script writes next to the results.

## How to commit a run

The script copies the run into the CREST checkout it benchmarked, under
`<workdir>/CREST/bench/paper/results/<stamp>/` (its last lines print the exact path).
That checkout is at the benchmarked commit, so start a branch from there and open a PR into
`dev` (never push to `main`):

```bash
cd crest-bench/CREST                      # the checkout the script used
git checkout -b bench/results-<stamp>
git add bench/paper/results/<stamp>
git commit -m "bench: paper benchmark results <stamp>"
git push -u origin bench/results-<stamp>
gh pr create --base dev --title "bench: paper benchmark results <stamp>"
```

Check the size first (`du -sh bench/paper/results/<stamp>`; a standard-tier run is a few MB).
If the docs quote numbers from a run, cite its directory name.
