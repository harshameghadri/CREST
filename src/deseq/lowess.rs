//! R's `lowess()` (Cleveland 1979; port of `clowess` in R's stats/src/lowess.c),
//! used by DESeq2's independent filtering. `x` must be sorted ascending.

fn fcube(x: f64) -> f64 {
    x * x * x
}

#[allow(clippy::too_many_arguments)]
fn lowest(x: &[f64], y: &[f64], xs: f64, nleft: usize, nright: usize, w: &mut [f64], rw: Option<&[f64]>) -> Option<f64> {
    let n = x.len();
    let range = x[n - 1] - x[0];
    let h = (xs - x[nleft]).max(x[nright] - xs);
    let (h9, h1) = (0.999 * h, 0.001 * h);
    let mut a = 0.0;
    let mut j = nleft;
    while j < n {
        w[j] = 0.0;
        let r = (x[j] - xs).abs();
        if r <= h9 {
            w[j] = if r <= h1 { 1.0 } else { fcube(1.0 - fcube(r / h)) };
            if let Some(rw) = rw {
                w[j] *= rw[j];
            }
            a += w[j];
        } else if x[j] > xs {
            break;
        }
        j += 1;
    }
    let nrt = j - 1; // rightmost point (may exceed nright because of ties)
    if a <= 0.0 {
        return None;
    }
    for wj in &mut w[nleft..=nrt] {
        *wj /= a;
    }
    if h > 0.0 {
        let a: f64 = (nleft..=nrt).map(|j| w[j] * x[j]).sum();
        let mut b = xs - a;
        let c: f64 = (nleft..=nrt).map(|j| w[j] * (x[j] - a) * (x[j] - a)).sum();
        if c.sqrt() > 0.001 * range {
            b /= c;
            for j in nleft..=nrt {
                w[j] *= b * (x[j] - a) + 1.0;
            }
        }
    }
    Some((nleft..=nrt).map(|j| w[j] * y[j]).sum())
}

/// Smoothed values at `x` (R: `lowess(x, y, f, iter, delta)`; R's default delta is `0.01 * diff(range(x))`).
pub fn lowess(x: &[f64], y: &[f64], f: f64, nsteps: usize, delta: f64) -> Vec<f64> {
    let n = x.len();
    if n < 2 {
        return y.to_vec();
    }
    let ns = 2.max(n.min((f * n as f64 + 1e-7) as usize));
    let mut ys = vec![0.0; n];
    let mut res = vec![0.0; n];
    let mut rw = vec![0.0; n];
    let mut w = vec![0.0; n];
    let mut iter = 1;
    while iter <= nsteps + 1 {
        let (mut nleft, mut nright) = (0usize, ns - 1);
        let mut last: isize = -1;
        let mut i = 0usize;
        loop {
            if nright < n - 1 {
                let d1 = x[i] - x[nleft];
                let d2 = x[nright + 1] - x[i];
                if d1 > d2 {
                    nleft += 1;
                    nright += 1;
                    continue;
                }
            }
            let fit = lowest(x, y, x[i], nleft, nright, &mut w, if iter > 1 { Some(&rw) } else { None });
            ys[i] = fit.unwrap_or(y[i]);
            if last < i as isize - 1 {
                let l = last as usize; // last >= 0 here: i > 0 whenever a gap exists
                let denom = x[i] - x[l];
                for j in l + 1..i {
                    let alpha = (x[j] - x[l]) / denom;
                    ys[j] = alpha * ys[i] + (1.0 - alpha) * ys[l];
                }
            }
            last = i as isize;
            let lu = last as usize;
            let cut = x[lu] + delta;
            let mut k = lu + 1;
            let mut lastu = lu;
            while k < n {
                if x[k] > cut {
                    break;
                }
                if x[k] == x[lastu] {
                    ys[k] = ys[lastu];
                    lastu = k;
                }
                k += 1;
            }
            last = lastu as isize;
            i = (lastu + 1).max(k - 1);
            if lastu >= n - 1 {
                break;
            }
        }
        for k in 0..n {
            res[k] = y[k] - ys[k];
        }
        let sc = res.iter().map(|r| r.abs()).sum::<f64>() / n as f64;
        if iter > nsteps {
            break;
        }
        for k in 0..n {
            rw[k] = res[k].abs();
        }
        let mut sorted = rw.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let m1 = n / 2;
        let cmad = if n % 2 == 0 { 3.0 * (sorted[m1] + sorted[n - m1 - 1]) } else { 6.0 * sorted[m1] };
        if cmad < 1e-7 * sc {
            break;
        }
        let (c9, c1) = (0.999 * cmad, 0.001 * cmad);
        for k in 0..n {
            let r = res[k].abs();
            rw[k] = if r <= c1 {
                1.0
            } else if r <= c9 {
                let u = 1.0 - (r / cmad) * (r / cmad);
                u * u
            } else {
                0.0
            };
        }
        iter += 1;
    }
    ys
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowess_matches_r() {
        // R: lowess(1:10, c(2,4,3,6,5,8,7,10,9,20), f=1/2)$y
        let x: Vec<f64> = (1..=10).map(|v| v as f64).collect();
        let y = vec![2.0, 4.0, 3.0, 6.0, 5.0, 8.0, 7.0, 10.0, 9.0, 20.0];
        let r = [2.287396884, 3.169721026, 4.152120654, 4.854933645, 6.145245957,
                 6.854889276, 8.154400363, 9.526068707, 14.803207661, 19.483632229];
        let got = lowess(&x, &y, 0.5, 3, 0.01 * 9.0);
        for (g, e) in got.iter().zip(r.iter()) {
            assert!((g - e).abs() < 1e-8, "{:?}", got);
        }
        // ties and a small span: lowess(x, y, f=1/5)$y
        let x2 = [1.0, 2.0, 2.0, 3.0, 5.0, 8.0, 8.0, 9.0, 12.0, 13.0, 20.0];
        let y2 = [1.0, 3.0, 2.0, 5.0, 4.0, 9.0, 7.0, 8.0, 15.0, 11.0, 30.0];
        let r2 = [1.0, 2.5, 2.5, 5.0, 4.0, 8.0, 8.0, 8.0, 15.0, 11.0, 30.0];
        for (g, e) in lowess(&x2, &y2, 0.2, 3, 0.19).iter().zip(r2.iter()) {
            assert!((g - e).abs() < 1e-8);
        }
        // a straight line is reproduced exactly
        let yl: Vec<f64> = x.iter().map(|v| 3.0 * v - 1.0).collect();
        for (g, e) in lowess(&x, &yl, 0.2, 3, 0.09).iter().zip(&yl) {
            assert!((g - e).abs() < 1e-9);
        }
    }
}
