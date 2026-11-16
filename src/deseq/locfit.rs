//! R `locfit` for the one call DESeq2 makes in `localDispersionFit`:
//!
//! ```r
//! locfit(logDisps ~ logMeans, weights = means)   # alpha = 0.7, deg = 2, kern = "tricube", ev = rbox(), family "qgaussian"
//! exp(predict(fit, data.frame(logMeans = log(means))))
//! ```
//!
//! One predictor only. The port follows locfit 1.5-9.12 (`startlf.c`, `ev_atree.c`, `ev_interp.c`, `lf_nbhd.c`,
//! `weight.c`, `locfit.c`, `procv.c`, `m_jacob.c`, `m_eigen.c`):
//!
//! * the bounding box is the range of the data; the evaluation structure is locfit's adaptive tree: a cell is
//!   split at its midpoint while `cut (0.8) < width / hmin`, where `hmin` is the smaller of the bandwidths at its
//!   two vertices; a midpoint whose cell is narrower than `cut * min(h)` is a "pseudo vertex" whose value and
//!   slope are interpolated from its parents, every other vertex gets its own local fit; at most 19 vertices
//!   (the allocation of `atree_guessnv` for `alpha = 0.7`, `maxk = 100`);
//! * a local fit at a vertex: the bandwidth is the distance to the `floor(0.7 n + 1e-12)`-th nearest data point,
//!   weights are `tricube(|x - x0| / h)` times the prior weights, the local quadratic is
//!   `a0 + a1 (x - x0) + a2 (x - x0)^2 / 2`, and the normal equations are solved with locfit's one Newton step
//!   (`max_nr` on a Gaussian likelihood stops after its first step) through the eigen-decomposition of the
//!   diagonally scaled matrix with eigenvalues below `1e-8 * max` dropped; the sums are formed in the order of the C
//!   code, because the decomposition amplifies roundoff;
//! * locfit's parametric component: a global weighted quadratic fit (centred at the weighted mean of `x`) is
//!   subtracted from the vertex values and slopes and added back at every evaluation, so the curve beyond the range
//!   of `x` is quadratic (not linear) with the curvature of that global fit;
//! * a prediction descends the tree to a terminal cell and interpolates the vertex values and slopes by a cubic
//!   Hermite polynomial (the end value and slope continue linearly outside the box, before the parametric component
//!   is added back).

const NN: f64 = 0.7;
const CUT: f64 = 0.8;
const E_TOL: f64 = 1.0e-8;
const E_MAXIT: usize = 20;

/// `atree_guessnv` for d = 1, alpha = 0.7, cut = 0.8, maxk = 100.
fn max_vertices() -> usize {
    let a0 = if NN > 1.0 { 1.0 } else { 1.0 / NN };
    let cu = CUT.min(1.0);
    ((5.0 * a0 / cu + 1.0) * 2.0) as usize
}

/// locfit's `eig_dec`: Jacobi rotations on the symmetric `d x d` row-major matrix `x`; returns the eigenvectors
/// (columns of `p`, row-major) and leaves the eigenvalues on the diagonal of `x`.
fn eig_dec(x: &mut [f64], p: &mut [f64], d: usize) {
    for i in 0..d {
        for j in 0..d {
            p[i * d + j] = if i == j { 1.0 } else { 0.0 };
        }
    }
    for _ in 0..E_MAXIT {
        let mut ms = false;
        for i in 0..d {
            for j in (i + 1)..d {
                if x[i * d + j] * x[i * d + j] > 1.0e-15 * (x[i * d + i] * x[j * d + j]).abs() {
                    let mut c = (x[j * d + j] - x[i * d + i]) / 2.0;
                    let s0 = -x[i * d + j];
                    let r = (c * c + s0 * s0).sqrt();
                    c /= r;
                    let s = ((1.0 - c) / 2.0).sqrt() * (if s0 > 0.0 { 1.0 } else { -1.0 });
                    let c = ((1.0 + c) / 2.0).sqrt();
                    for k in 0..d {
                        let (u, v) = (x[i * d + k], x[j * d + k]);
                        x[i * d + k] = u * c + v * s;
                        x[j * d + k] = v * c - u * s;
                    }
                    for k in 0..d {
                        let (u, v) = (x[k * d + i], x[k * d + j]);
                        x[k * d + i] = u * c + v * s;
                        x[k * d + j] = v * c - u * s;
                    }
                    x[i * d + j] = 0.0;
                    x[j * d + i] = 0.0;
                    for k in 0..d {
                        let (u, v) = (p[k * d + i], p[k * d + j]);
                        p[k * d + i] = u * c + v * s;
                        p[k * d + j] = v * c - u * s;
                    }
                    ms = true;
                }
            }
        }
        if !ms {
            return;
        }
    }
}

/// locfit's `jacob_solve` with the default method `JAC_EIGD`: `(X' W X)^-1 v` through the eigen-decomposition of
/// the diagonally scaled matrix, dropping eigenvalues below `1e-8 * max`. Returns the rank.
fn jacob_solve(z: &[f64; 9], v: &mut [f64; 3]) -> usize {
    let d = 3;
    let mut zz = *z;
    let mut dg = [0.0; 3];
    for i in 0..d {
        dg[i] = if zz[i * (d + 1)] <= 0.0 { 0.0 } else { 1.0 / zz[i * (d + 1)].sqrt() };
    }
    for i in 0..d {
        for j in 0..d {
            zz[i * d + j] *= dg[i] * dg[j];
        }
    }
    let mut q = [0.0; 9];
    eig_dec(&mut zz, &mut q, d);
    for i in 0..d {
        v[i] *= dg[i];
    }
    let mut mx = zz[0];
    for i in 1..d {
        if zz[i * (d + 1)] > mx {
            mx = zz[i * (d + 1)];
        }
    }
    let tol = E_TOL * mx;
    let mut w = [0.0; 3];
    let mut rank = 0;
    for i in 0..d {
        w[i] = 0.0;
        for j in 0..d {
            w[i] += q[j * d + i] * v[j];
        }
    }
    for i in 0..d {
        if zz[i * (d + 1)] > tol {
            w[i] /= zz[i * (d + 1)];
            rank += 1;
        }
    }
    for i in 0..d {
        v[i] = 0.0;
        for j in 0..d {
            v[i] += q[i * d + j] * w[j];
        }
    }
    for i in 0..d {
        v[i] *= dg[i];
    }
    rank
}

fn tricube(u: f64) -> f64 {
    let u = u.abs();
    if u > 1.0 {
        0.0
    } else {
        let t = 1.0 - u * u * u;
        t * t * t
    }
}

/// locfit's `hermite2`.
fn hermite2(x: f64, z: f64) -> [f64; 4] {
    if z == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let h = x / z;
    if h < 0.0 {
        return [1.0, 0.0, h, 0.0];
    }
    if h > 1.0 {
        return [0.0, 1.0, 0.0, h - 1.0];
    }
    let p1 = h * h * (3.0 - 2.0 * h);
    [1.0 - p1, p1, h * (1.0 - h) * (1.0 - h), h * h * (h - 1.0)]
}

/// `exvvalpv`: value and slope of the pseudo vertex at the midpoint of a cell of width `dl` from its end vertices.
fn pseudo_values(vl: [f64; 2], vr: [f64; 2], dl: f64) -> [f64; 2] {
    let f0 = (vl[0] + vr[0]) / 2.0 + dl * (vl[1] - vr[1]) / 8.0;
    let f1 = 1.5 * (vr[0] - vl[0]) / dl - (vl[1] + vr[1]) / 4.0;
    [f0, f1]
}

/// One observation of a local fit: kernel weight, prior weight, `x - x0`, response.
struct Row {
    kw: f64,
    prior: f64,
    dx: f64,
    y: f64,
}

/// The kernel weight: locfit's tricube of `|x - x0| / h`, or 1 at `x0` itself where the bandwidth is 0.
fn kernel(di: f64, h: f64) -> f64 {
    if h == 0.0 {
        if di == 0.0 { 1.0 } else { 0.0 }
    } else {
        tricube(di / h)
    }
}

/// The weighted quadratic fit of the rows in the basis `[1, dx, dx^2 / 2]`: locfit's start value (`reginit`) followed by one
/// Newton step (`max_nr` on the Gaussian likelihood stops after its first step). The sums are formed in the order of locfit's
/// C code (`reginit`, `likereg`, `addouter`) so that the roundoff is the same. `None` where no row has weight.
fn quad_solve(rows: &[Row]) -> Option<[f64; 3]> {
    let (mut s0, mut s1) = (0.0f64, 0.0f64);
    for r in rows {
        s1 += r.kw * (r.prior * r.y);
        s0 += r.kw * r.prior;
    }
    if rows.is_empty() || s0 == 0.0 {
        return None;
    }
    let cf0 = (s1 - 0.0) / s0;
    let mut z = [0.0f64; 9];
    let mut f1 = [0.0f64; 3];
    for r in rows {
        let xr = [1.0, r.dx, r.dx * r.dx / 2.0];
        let zdll = r.prior * (r.y - cf0);
        let c = r.kw * r.prior;
        for a in 0..3 {
            f1[a] += xr[a] * r.kw * zdll;
            for b in 0..3 {
                z[a * 3 + b] += c * xr[a] * xr[b];
            }
        }
    }
    let mut delta = f1;
    let rank = jacob_solve(&z, &mut delta);
    if rank == 0 {
        delta = [0.0; 3];
    }
    Some([cf0 + delta[0], delta[1], delta[2]])
}

/// locfit's parametric component (`compparcomp`): a global weighted quadratic fit, centred at the weighted mean of `x`.
/// It is subtracted from the vertex values and added back at every evaluation, which makes the extrapolation beyond the
/// range of `x` quadratic instead of linear.
fn parametric_component(x: &[f64], y: &[f64], w: &[f64]) -> Result<[f64; 4], String> {
    let mut sw = 0.0;
    let mut sxw = 0.0;
    for i in 0..x.len() {
        sw += w[i];
        sxw += x[i] * w[i];
    }
    let xbar = sxw / sw;
    let rows: Vec<Row> = (0..x.len()).map(|i| Row { kw: 1.0, prior: w[i], dx: x[i] - xbar, y: y[i] }).collect();
    let c = quad_solve(&rows).ok_or_else(|| "locfit: no points with non-zero weight".to_string())?;
    Ok([xbar, c[0], c[1], c[2]])
}

fn pc_value(pc: &[f64; 4], x: f64) -> f64 {
    let dx = x - pc[0];
    pc[1] + pc[2] * dx + pc[3] * dx * dx / 2.0
}

fn pc_slope(pc: &[f64; 4], x: f64) -> f64 {
    pc[2] + pc[3] * (x - pc[0])
}

enum Node {
    Leaf,
    Split { mid: usize, left: Box<Node>, right: Box<Node> },
}

/// A fitted locfit model (one predictor).
pub struct LocalTrend {
    xs: Vec<f64>,
    val: Vec<f64>,
    slope: Vec<f64>,
    pseudo: Vec<bool>,
    pc: [f64; 4],
    root: Node,
}

struct Builder<'a> {
    x: &'a [f64],
    y: &'a [f64],
    w: &'a [f64],
    xs: Vec<f64>,
    val: Vec<f64>,
    slope: Vec<f64>,
    h: Vec<f64>,
    pseudo: Vec<bool>,
    nvm: usize,
    box_width: f64,
    pc: [f64; 4],
    ordered: bool,
}

impl<'a> Builder<'a> {
    /// A local fit at `x0` (locfit's `procv`): value, slope (both net of the parametric component) and bandwidth.
    fn fit_vertex(&self, x0: f64) -> Result<(f64, f64, f64), String> {
        let n = self.x.len();
        let k = (n as f64 * NN + 1e-12) as usize;
        let mut rows: Vec<Row> = Vec::new();
        let h;
        if self.ordered && k <= n {
            // locfit's `nbhd1`: x is sorted; the window of k points around the nearest one gives the bandwidth, and the
            // points are visited from the nearest one to the left, then to the right, until the weight is zero
            let xd = self.x;
            let z0: usize = if x0 <= xd[0] {
                0
            } else if x0 >= xd[n - 1] {
                n - 1
            } else {
                let (mut l, mut r) = (0usize, n - 1);
                while r - l > 1 {
                    let z = (r + l) / 2;
                    if xd[z] > x0 {
                        r = z;
                    } else {
                        l = z;
                    }
                }
                if (x0 - xd[l]) > (xd[r] - x0) { r } else { l }
            };
            let mut hh = 0.0;
            if k > 0 {
                let (mut l, mut r) = (z0, z0);
                if l == 0 {
                    r = k - 1;
                }
                if r == n - 1 {
                    l = n - k;
                }
                while r - l < k - 1 {
                    if (x0 - xd[l - 1]) < (xd[r + 1] - x0) {
                        l -= 1;
                    } else {
                        r += 1;
                    }
                    if l == 0 {
                        r = k - 1;
                    }
                    if r == n - 1 {
                        l = n - k;
                    }
                }
                hh = x0 - xd[l];
                if hh < xd[r] - x0 {
                    hh = xd[r] - x0;
                }
            }
            h = hh;
            let mut z = z0 as isize;
            if xd[z0] > x0 {
                z -= 1;
            }
            let mut i = z;
            while i >= 0 {
                let di = x0 - xd[i as usize];
                let kw = kernel(di, h);
                if kw > 0.0 {
                    rows.push(Row { kw, prior: self.w[i as usize], dx: xd[i as usize] - x0, y: self.y[i as usize] });
                    i -= 1;
                } else {
                    break;
                }
            }
            let mut i = (z + 1) as usize;
            while i < n {
                let di = xd[i] - x0;
                let kw = kernel(di, h);
                if kw > 0.0 {
                    rows.push(Row { kw, prior: self.w[i], dx: xd[i] - x0, y: self.y[i] });
                    i += 1;
                } else {
                    break;
                }
            }
        } else {
            let di: Vec<f64> = self.x.iter().map(|&xi| (xi - x0).abs()).collect();
            h = if k == 0 {
                0.0
            } else if k < n {
                let mut s = di.clone();
                s.sort_by(|a, b| a.partial_cmp(b).unwrap());
                s[k - 1]
            } else {
                di.iter().cloned().fold(0.0, f64::max) * (k as f64 / n as f64)
            };
            for i in 0..n {
                let kw = kernel(di[i], h);
                if kw > 0.0 {
                    rows.push(Row { kw, prior: self.w[i], dx: self.x[i] - x0, y: self.y[i] });
                }
            }
        }
        let c = quad_solve(&rows).ok_or_else(|| "locfit: no points with non-zero weight".to_string())?;
        Ok((c[0] - pc_value(&self.pc, x0), c[1] - pc_slope(&self.pc, x0), h))
    }

    fn add_real(&mut self, x0: f64) -> Result<usize, String> {
        let (v, s, h) = self.fit_vertex(x0)?;
        self.xs.push(x0);
        self.val.push(v);
        self.slope.push(s);
        self.h.push(h);
        self.pseudo.push(false);
        Ok(self.xs.len() - 1)
    }

    fn grow(&mut self, lv: usize, rv: usize, ll: f64, ur: f64) -> Result<Node, String> {
        let mut hmin = 0.0;
        for &i in &[lv, rv] {
            let h = self.h[i];
            if h > 0.0 && (hmin == 0.0 || h < hmin) {
                hmin = h;
            }
        }
        let le = ur - ll;
        let score = if hmin == 0.0 { 2.0 * (ur - ll) / self.box_width } else { le / hmin };
        if !(CUT < score) {
            return Ok(Node::Leaf);
        }
        if self.xs.len() == self.nvm {
            return Err("locfit: out of vertex space".into());
        }
        let pv = le < CUT * self.h[lv].min(self.h[rv]);
        let xm = (self.xs[rv] + self.xs[lv]) / 2.0;
        let mid = if pv {
            self.xs.push(xm);
            self.val.push(f64::NAN);
            self.slope.push(f64::NAN);
            self.h.push((self.h[lv] + self.h[rv]) / 2.0);
            self.pseudo.push(true);
            self.xs.len() - 1
        } else {
            self.add_real(xm)?
        };
        let left = self.grow(lv, mid, ll, xm)?;
        let right = self.grow(mid, rv, xm, ur)?;
        Ok(Node::Split { mid, left: Box::new(left), right: Box::new(right) })
    }
}

impl LocalTrend {
    /// `locfit(y ~ x, weights = w)` with locfit's defaults; `x`, `y`, `w` must have the same length (at least 1).
    pub fn fit(x: &[f64], y: &[f64], w: &[f64]) -> Result<LocalTrend, String> {
        if x.is_empty() || x.len() != y.len() || x.len() != w.len() {
            return Err("locfit: empty or mismatched input".into());
        }
        if x.iter().chain(y).chain(w).any(|v| !v.is_finite()) {
            return Err("locfit: non-finite input".into());
        }
        let fl = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let fr = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut b = Builder {
            x, y, w, xs: vec![], val: vec![], slope: vec![], h: vec![], pseudo: vec![], nvm: max_vertices(),
            box_width: fr - fl,
            pc: parametric_component(x, y, w)?,
            ordered: x.windows(2).all(|p| p[1] >= p[0]),
        };
        let v0 = b.add_real(fl)?;
        let v1 = b.add_real(fr)?;
        let root = b.grow(v0, v1, fl, fr)?;
        Ok(LocalTrend { xs: b.xs, val: b.val, slope: b.slope, pseudo: b.pseudo, pc: b.pc, root })
    }

    /// The fitted curve at `x` (locfit's `atree_int`).
    pub fn predict(&self, x: f64) -> f64 {
        let (mut lv, mut rv) = (0usize, 1usize);
        let mut vl = [self.val[0], self.slope[0]];
        let mut vr = [self.val[1], self.slope[1]];
        let mut node = &self.root;
        while let Node::Split { mid, left, right } = node {
            let (ll, ur) = (self.xs[lv], self.xs[rv]);
            let h = ur - ll;
            let lo = (2.0 * (x - ll)) < h;
            let vm = if self.pseudo[*mid] { pseudo_values(vl, vr, h) } else { [self.val[*mid], self.slope[*mid]] };
            if lo {
                vr = vm;
                rv = *mid;
                node = left;
            } else {
                vl = vm;
                lv = *mid;
                node = right;
            }
        }
        let (ll, ur) = (self.xs[lv], self.xs[rv]);
        let mut phi = hermite2(x - ll, ur - ll);
        phi[2] *= ur - ll;
        phi[3] *= ur - ll;
        phi[0] * vl[0] + phi[1] * vr[0] + phi[2] * vl[1] + phi[3] * vr[1] + pc_value(&self.pc, x)
    }

    /// Number of vertices (fitted and pseudo).
    #[cfg(test)]
    pub fn n_vertices(&self) -> usize {
        self.xs.len()
    }
}

/// DESeq2 `localDispersionFit`: the trend is `exp(locfit(log(disps) ~ log(means), weights = means))`.
pub fn local_dispersion_fit(means: &[f64], disps: &[f64]) -> Result<LocalTrend, String> {
    let x: Vec<f64> = means.iter().map(|m| m.ln()).collect();
    let y: Vec<f64> = disps.iter().map(|d| d.ln()).collect();
    LocalTrend::fit(&x, &y, means)
}

impl LocalTrend {
    /// `exp(predict(fit, log(mean)))`.
    pub fn disp_at(&self, mean: f64) -> f64 {
        self.predict(mean.ln()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproduces_a_straight_line() {
        let x: Vec<f64> = (0..200).map(|i| i as f64 / 20.0).collect();
        let y: Vec<f64> = x.iter().map(|v| 0.5 * v - 1.0).collect();
        let w = vec![1.0; 200];
        let f = LocalTrend::fit(&x, &y, &w).unwrap();
        for &t in &[0.0, 0.3, 4.7, 9.95, 12.0, -1.0] {
            assert!((f.predict(t) - (0.5 * t - 1.0)).abs() < 1e-6, "{t}: {}", f.predict(t));
        }
    }

    #[test]
    fn tied_x_and_few_points_do_not_fail() {
        let x = [1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 4.0];
        let y = [0.1, 0.2, 0.15, 0.4, 0.5, 0.9, 1.2, 1.1];
        let w = [1.0, 2.0, 3.0, 1.0, 1.0, 5.0, 1.0, 2.0];
        let f = LocalTrend::fit(&x, &y, &w).unwrap();
        assert!(f.predict(2.5).is_finite() && f.n_vertices() >= 2);
    }
}
