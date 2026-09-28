//! Small dense linear algebra for per-gene GLM fits (p is the number of model
//! coefficients, typically < 20). Row-major, allocation-light, no BLAS.

/// LU decomposition with partial pivoting of a row-major p × p matrix.
pub struct Lu {
    a: Vec<f64>,
    piv: Vec<usize>,
    p: usize,
}

impl Lu {
    /// `None` if the matrix is (numerically) singular.
    pub fn new(mut a: Vec<f64>, p: usize) -> Option<Lu> {
        let mut piv: Vec<usize> = (0..p).collect();
        let scale = a.iter().fold(0.0f64, |s, &v| s.max(v.abs()));
        if !(scale > 0.0) || !scale.is_finite() {
            return None;
        }
        for k in 0..p {
            let (mut best, mut bv) = (k, a[k * p + k].abs());
            for i in k + 1..p {
                let v = a[i * p + k].abs();
                if v > bv {
                    best = i;
                    bv = v;
                }
            }
            if bv <= scale * 1e-14 {
                return None;
            }
            if best != k {
                for j in 0..p {
                    a.swap(k * p + j, best * p + j);
                }
                piv.swap(k, best);
            }
            let d = a[k * p + k];
            for i in k + 1..p {
                let f = a[i * p + k] / d;
                a[i * p + k] = f;
                for j in k + 1..p {
                    a[i * p + j] -= f * a[k * p + j];
                }
            }
        }
        Some(Lu { a, piv, p })
    }

    pub fn log_abs_det(&self) -> f64 {
        (0..self.p).map(|i| self.a[i * self.p + i].abs().ln()).sum()
    }

    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let p = self.p;
        let mut x: Vec<f64> = self.piv.iter().map(|&i| b[i]).collect();
        for i in 0..p {
            for j in 0..i {
                x[i] -= self.a[i * p + j] * x[j];
            }
        }
        for i in (0..p).rev() {
            for j in i + 1..p {
                x[i] -= self.a[i * p + j] * x[j];
            }
            x[i] /= self.a[i * p + i];
        }
        x
    }

    /// Row-major inverse.
    pub fn inverse(&self) -> Vec<f64> {
        let p = self.p;
        let mut inv = vec![0.0; p * p];
        let mut e = vec![0.0; p];
        for j in 0..p {
            e.iter_mut().for_each(|v| *v = 0.0);
            e[j] = 1.0;
            let col = self.solve(&e);
            for i in 0..p {
                inv[i * p + j] = col[i];
            }
        }
        inv
    }
}

/// Xᵀ diag(w) X for row-major X (m × p).
pub fn xtwx(x: &[f64], m: usize, p: usize, w: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..m {
        let r = &x[i * p..(i + 1) * p];
        let wi = w[i];
        for a in 0..p {
            let ra = r[a] * wi;
            for b in a..p {
                out[a * p + b] += ra * r[b];
            }
        }
    }
    for a in 0..p {
        for b in 0..a {
            out[a * p + b] = out[b * p + a];
        }
    }
    out
}

/// Xᵀ diag(w) z.
pub fn xtwz(x: &[f64], m: usize, p: usize, w: &[f64], z: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; p];
    for i in 0..m {
        let wz = w[i] * z[i];
        for a in 0..p {
            out[a] += x[i * p + a] * wz;
        }
    }
    out
}

/// X β for row-major X (m × p).
#[inline]
pub fn xb(x: &[f64], m: usize, p: usize, beta: &[f64], out: &mut [f64]) {
    for i in 0..m {
        out[i] = x[i * p..(i + 1) * p].iter().zip(beta).map(|(a, b)| a * b).sum();
    }
}

/// A B for row-major p × p matrices.
pub fn matmul(a: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut out = vec![0.0; p * p];
    for i in 0..p {
        for k in 0..p {
            let aik = a[i * p + k];
            for j in 0..p {
                out[i * p + j] += aik * b[k * p + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_and_inverts() {
        let a = vec![4.0, 1.0, 2.0, 1.0, 3.0, 0.5, 2.0, 0.5, 5.0];
        let lu = Lu::new(a.clone(), 3).unwrap();
        let inv = lu.inverse();
        let id = matmul(&a, &inv, 3);
        for i in 0..3 {
            for j in 0..3 {
                assert!((id[i * 3 + j] - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
            }
        }
        // det = 4(15-0.25) - 1(5-1) + 2(0.5-6) = 59 - 4 - 11 = 44
        assert!((lu.log_abs_det() - 44f64.ln()).abs() < 1e-12);
        assert!(Lu::new(vec![1.0, 2.0, 2.0, 4.0], 2).is_none());
    }
}
