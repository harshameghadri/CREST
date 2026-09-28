# CREST vs scanpy benchmark

### pbmc68k.h5 (68,551 cells after QC)

| step | scanpy | CREST (in-memory) | CREST (out-of-core) | speed-up (in-memory) |
|---|---|---|---|---|
| read | 2.10 s | 0.19 s | 1.90 s | 11.2x |
| QC filter | 0.94 s | 0.12 s | 1.18 s | 7.6x |
| normalize + log1p | 0.17 s | 0.00 s | 0.00 s | lazy |
| HVG | 1.66 s | 0.25 s | 0.99 s | 6.6x |
| scale + PCA | 33.53 s | 1.09 s | 2.01 s | 30.7x |
| neighbors | 5.30 s | 1.74 s | 1.65 s | 3.0x |
| Leiden | 2.31 s | 0.64 s | 0.66 s | 3.6x |
| UMAP | 36.07 s | 9.68 s | 9.44 s | 3.7x |
| DE t-test | 8.85 s | 0.88 s | 1.41 s | 10.0x |
| DE Wilcoxon | 40.96 s | 4.12 s | 5.01 s | 9.9x |
| **total** | 133.7 s | 18.8 s | 24.4 s | 7.1x |

Peak RSS: scanpy 5.20 GB / CREST (in-memory) 1.21 GB / CREST (out-of-core) 0.99 GB

### pbmc_100k.h5 (97,723 cells after QC)

| step | scanpy | CREST (in-memory) | CREST (out-of-core) | speed-up (in-memory) |
|---|---|---|---|---|
| read | 24.53 s | 12.84 s | 16.78 s | 1.9x |
| QC filter | 8.03 s | 0.47 s | 5.61 s | 17.2x |
| normalize + log1p | 0.82 s | 0.00 s | 0.00 s | lazy |
| HVG | 9.21 s | 1.22 s | 3.75 s | 7.6x |
| scale + PCA | 38.47 s | 2.56 s | 7.80 s | 15.0x |
| neighbors | 7.70 s | 2.26 s | 2.22 s | 3.4x |
| Leiden | 3.44 s | 0.99 s | 0.94 s | 3.5x |
| UMAP | 59.85 s | 12.31 s | 12.08 s | 4.9x |
| DE t-test | 32.68 s | 3.51 s | 5.31 s | 9.3x |
| DE Wilcoxon | 159.57 s | 20.54 s | 54.88 s | 7.8x |
| **total** | 346.4 s | 56.8 s | 109.4 s | 6.1x |

Peak RSS: scanpy 10.33 GB / CREST (in-memory) 2.49 GB / CREST (out-of-core) 1.14 GB

### pbmc_200k.h5 (195,479 cells after QC)

**scanpy: out of memory (16.9 GB RAM).**

| step | CREST (in-memory) | CREST (out-of-core) |
|---|---|---|
| read | 22.64 s | 33.96 s |
| QC filter | 1.16 s | 10.80 s |
| normalize + log1p | 0.00 s | 0.00 s |
| HVG | 2.68 s | 7.86 s |
| scale + PCA | 4.90 s | 14.95 s |
| neighbors | 4.90 s | 4.65 s |
| Leiden | 2.49 s | 2.44 s |
| UMAP | 25.10 s | 26.57 s |
| DE t-test | 6.41 s | 12.98 s |
| DE Wilcoxon | 49.14 s | 168.55 s |
| **total** | 119.6 s | 282.9 s |

Peak RSS: CREST (in-memory) 4.11 GB / CREST (out-of-core) 1.03 GB

## Agreement with scanpy

- pbmc68k.h5 / CREST (in-memory): Leiden ARI vs scanpy 0.871 (46 vs 47 clusters, 68,551 shared cells)
- pbmc68k.h5 / CREST (out-of-core): Leiden ARI vs scanpy 0.871 (46 vs 47 clusters, 68,551 shared cells)
- pbmc_100k.h5 / CREST (in-memory): Leiden ARI vs scanpy 0.894 (38 vs 38 clusters, 97,723 shared cells)
- pbmc_100k.h5 / CREST (out-of-core): Leiden ARI vs scanpy 0.894 (38 vs 38 clusters, 97,723 shared cells)

Machine: 4 cores, 17 GB RAM, Python 3.11.15.
