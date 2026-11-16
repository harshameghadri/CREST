"""UMAP must not leave flush-to-zero / denormals-are-zero set on the caller's thread.

On x86-64 the UMAP SGD turns on FTZ|DAZ for speed. On the serial path (one
thread, or fewer than 1000 cells) that runs on the Python thread itself, so the
mode has to be restored afterwards or numpy would flush subnormals for the rest
of the process. On other architectures this test is trivially true.
"""

import numpy as np

from crest import crest as _native


def test_umap_does_not_flush_denormals_afterwards():
    rng = np.random.default_rng(0)
    n = 200  # < 1000: serial path, runs on this thread
    rows, cols = [], []
    for i in range(n):
        for j in rng.choice(n, 8, replace=False):
            if i != j:
                rows.append(i)
                cols.append(int(j))
    rows = np.asarray(rows, np.uint32)
    cols = np.asarray(cols, np.uint32)
    w = rng.uniform(0.1, 1.0, len(rows)).astype(np.float32)
    tiny = np.array([1e-40, 3e-39], np.float32)  # float32 subnormals
    assert (tiny * np.float32(1.0) != 0).all()
    emb = _native.umap(n, rows, cols, w, 2, 0.5, 1.0, 50, 100, 0)
    assert np.isfinite(emb).all()
    # with DAZ left on, subnormal inputs read as 0; with FTZ, subnormal results become 0
    assert (tiny * np.float32(1.0) != 0).all()
    assert np.float32(1e-30) * np.float32(1e-10) != 0
