"""One-predictor loess with the netlib ``dloess`` defaults used by R ``loess`` and
``skmisc.loess``: ``surface="interpolate"`` (k-d tree, local fits at the cell
vertices, cubic Hermite interpolation in between), tricube weights, no
robustness iterations (``family="gaussian"``).

Only what ``highly_variable_genes(flavor="seurat_v3")`` needs; validated against
``skmisc.loess`` in the tests. Scaling ``x`` (``normalize=True``) does not change a
one-predictor fit, so it is skipped.
"""

from __future__ import annotations

import numpy as np


def _local_fit(xs: np.ndarray, ys: np.ndarray, q: float, nf: int, span: float, degree: int):
    """Value and slope of the local polynomial at ``q`` (netlib ehg127/ehg136)."""
    d2 = (xs - q) ** 2
    part = np.argpartition(d2, nf - 1)[:nf]
    rho = d2[part].max() * max(1.0, span)
    if rho <= 0:
        return float(ys[part].mean()), 0.0
    near = part[d2[part] < rho]
    w = (1.0 - np.sqrt(d2[near] / rho) ** 3) ** 3
    t = xs[near] - q
    X = np.vander(t, degree + 1, increasing=True)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], ys[near] * sw, rcond=None)
    return float(coef[0]), float(coef[1]) if degree >= 1 else 0.0


def _kd_vertices(xs: np.ndarray, fc: int) -> np.ndarray:
    """Cell boundaries of the 1-d k-d tree (netlib ehg126/ehg124)."""
    lo, hi = float(xs.min()), float(xs.max())
    mu = 0.005 * max(hi - lo, 1e-10 * max(abs(lo), abs(hi)) + 1e-30)
    lo, hi = lo - mu, hi + mu
    xsort = np.sort(xs)
    verts = {lo, hi}
    stack = [(0, len(xs) - 1, lo, hi)]  # 0-based inclusive point range, cell bounds
    while stack:
        l, u, vlo, vhi = stack.pop()
        if u - l + 1 <= fc:
            continue
        m = (l + u + 2) // 2 - 1  # Fortran m = (l+u)/2 on 1-based indices
        # move the split off a run of ties: offsets 0, 1, -1, 2, -2, ...
        off = 0
        while l <= m + off < u:
            if xsort[m + off] == xsort[m + off + 1]:
                off = -off
                if off >= 0:
                    off += 1
            else:
                m = m + off
                break
        xi = float(xsort[m])
        if xi == vlo or xi == vhi:
            continue
        verts.add(xi)
        stack.append((l, m, vlo, xi))
        stack.append((m + 1, u, xi, vhi))
    return np.array(sorted(verts))


def loess_fit(x: np.ndarray, y: np.ndarray, span: float = 0.75, degree: int = 2, cell: float = 0.2) -> np.ndarray:
    """Fitted values of ``loess(y ~ x, span, degree)`` at the data points."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    n = len(x)
    if n == 0:
        return np.zeros(0)
    nf = min(n, int(np.floor(n * span + 1e-5)))
    if nf <= 0:
        raise ValueError("span is too small")
    fc = int(np.floor(n * span * cell))
    v = _kd_vertices(x, fc)
    fits = np.array([_local_fit(x, y, q, nf, span, degree) for q in v])
    val, slope = fits[:, 0], fits[:, 1]
    j = np.clip(np.searchsorted(v, x, side="left") - 1, 0, len(v) - 2)
    x0, x1 = v[j], v[j + 1]
    h = x1 - x0
    u = (x - x0) / h
    phi0 = (1 - u) ** 2 * (1 + 2 * u)
    phi1 = u ** 2 * (3 - 2 * u)
    psi0 = u * (1 - u) ** 2
    psi1 = -(u ** 2) * (1 - u)
    return phi0 * val[j] + phi1 * val[j + 1] + (psi0 * slope[j] + psi1 * slope[j + 1]) * h
