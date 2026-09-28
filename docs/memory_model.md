# CREST memory model

What CREST keeps in memory, compared with the standard scanpy workflow. Numbers are
from the 100k-cell benchmark (97,723 cells after QC, 191M non-zeros, 2,000 HVGs,
4 cores / 16.9 GB; `bench/whitepaper/results/`).

## The one large object: raw counts, stored once

| | scanpy | CREST in-memory | CREST out-of-core |
|---|---|---|---|
| raw counts | CSR in RAM (float32 + int32 = 8 B/non-zero ≈ 1.5 GB) | same CSR, 8 B/non-zero ≈ 1.5 GB | Parquet parts on disk (zstd); ~8M non-zeros in RAM at a time (~0.1 GB) |
| normalised + log1p matrix | replaces X in place; `adata.raw` keeps the log-normalised full matrix for DE | never stored: recomputed per cell inside each kernel | same |
| HVG subset | copied (`adata[:, hvg].copy()`) | never stored: genes selected by an index map | same |
| scaled matrix (`sc.pp.scale`) | **dense** cells × 2,000 (plus float64 temporaries) | never formed: scaling is folded into the PCA Gram matrix exactly | same |
| peak RSS (measured) | **10.3 GB** | **2.5 GB** | **1.1 GB** |

Everything CREST adds on top of the raw counts is small and scales with cells or
genes, never with cells × genes:

| structure | size at 100k cells |
|---|---|
| working chunk (per kernel call) | ≤ 16.8M entries, bounded |
| gene × gene Gram matrix (PCA) | 2,000² × 8 B = 32 MB |
| PCA scores | 97.7k × 50 × 4 B = 20 MB |
| kNN indices + distances | 97.7k × 14 × 8 B = 11 MB |
| fuzzy graph (Leiden / UMAP) | ~2M edges ≈ 30 MB |
| DE accumulators | groups × genes × 3 × 8 B (38 × 21k ≈ 19 MB) |
| Wilcoxon gene block | ≤ `memory_budget_gb` (default 0.25 GB) |

## Why the results are unchanged

Every kernel sees exactly the values scanpy would hold in memory: per cell it
applies the gene filter, divides by the cell's total over kept genes, multiplies by
`target_sum`, then `log1p`. Nothing is approximated; only *when* the arithmetic
happens changes. PCA with `scale(max_value)` uses the identity
`Z = 1 bᵀ + S` (every implicit zero of gene *j* maps to the constant
`b_j = clip(−μ_j/σ_j)`; `S` is sparse with entries `clip((x−μ)/σ) − b_j`); centring
removes `1 bᵀ`, so the PCA of the dense scaled matrix equals the PCA of the sparse
`S`, which is accumulated one chunk at a time as a Gram matrix. `tests/test_crest.py`
checks that results do not depend on chunk size and that the scaled PCA equals a
dense reference.

## The cost

* Each step is one (or two, for PCA) pass over the raw counts, recomputing the
  normalisation. The fused kernels make this cheaper than scanpy's single
  materialised pass: every step is still 3–17× faster.
* Out-of-core mode re-reads Parquet from disk on every pass (≈ 10 passes for the
  full pipeline, more for Wilcoxon, which processes genes in memory-bounded
  blocks). That is the time you trade for flat ~1 GB memory: 109 s vs 57 s
  in-memory at 100k, 283 s vs 120 s at 200k.
* There is no stored normalised matrix to hand to other tools; `to_scipy()` /
  `to_anndata(transform=True)` materialise it on demand.
