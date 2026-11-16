# Known issues

Limitations of the current release that are understood and documented rather than fixed.
Planned improvements are tracked in the 0.5.1 milestone:
<https://github.com/harshameghadri/CREST/milestones>.

## Performance

* **`view()` scans the chunk's cell and gene maps on every call.** The cost of one call grows
  with the chunk size (it is O(chunk) per call). This matters only for very large chunks
  combined with many repeated calls.

## Numerical behaviour

* **Variance computed from raw power sums loses precision when the mean is very large
  relative to the standard deviation.** Per-gene variance is derived from accumulated sums of
  x and x². Typical count data is not affected; the loss shows up in stress cases such as a
  gene whose values are a very large constant plus small noise.
* **Leiden uses absolute epsilons.** Edge weights must not be rescaled to tiny values: the
  comparisons inside the algorithm are not relative to the weight scale. Weights produced by
  `pp.neighbors` are on a suitable scale.
* **Parallel UMAP lowers global-layout consistency as the number of threads increases.**
  The local neighbourhood structure of the embedding is unaffected; the relative arrangement
  of distant clusters is less consistent from run to run with more threads.

## Compatibility

* **A `numpy.core` deprecation warning appears under NumPy 2.** The Rust binding library used
  by this release (rust-numpy 0.21) triggers it when NumPy 2 is installed. It does not affect
  results. The supported range is declared as `numpy>=1.24,<3`.
