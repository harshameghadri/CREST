- Changed: `harmony` accepts `sigma` as an array with one value per cluster, as harmonypy 2.1.0 does; a wrong
  length, or a value that is not finite and positive, raises `ValueError`.
- Fixed: the native Harmony binding's `max_iter_kmeans` default is now 4, the same as `crest.tl.harmony` and
  harmonypy (it was 20 when the binding was called directly).
- Changed: the documentation names harmonypy 2.1.0, the release CREST is compared with.
- Added: tests that pin the connectivities against umap-learn, Leiden modularity against leidenalg, UMAP
  trustworthiness against umap-learn, and the fit of the UMAP curve parameters `a` and `b`.
