//! Utility functions for UMAP

/// Fits the UMAP curve 1 / (1 + a * x^(2b)) to the target
/// y = 1 (x < min_dist), exp(-(x - min_dist) / spread) otherwise, on 300 points
/// in [0, 3 * spread] — the same least-squares problem umap-learn solves with
/// scipy's `curve_fit`, solved here by Levenberg–Marquardt in f64.
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    let (spread, min_dist) = (spread as f64, min_dist as f64);
    let n_pts = 300;
    let xs: Vec<f64> = (0..n_pts).map(|i| i as f64 * 3.0 * spread / (n_pts - 1) as f64).collect();
    let ys: Vec<f64> = xs.iter().map(|&x| if x < min_dist { 1.0 } else { (-(x - min_dist) / spread).exp() }).collect();

    let sse = |a: f64, b: f64| -> f64 {
        xs.iter().zip(&ys).map(|(&x, &y)| {
            let f = 1.0 / (1.0 + a * x.powf(2.0 * b));
            (f - y) * (f - y)
        }).sum()
    };
    let (mut a, mut b) = (1.0f64, 1.0f64);
    let mut lambda = 1e-3;
    let mut cur = sse(a, b);
    for _ in 0..500 {
        // J^T J and J^T r for residual r = f - y
        let (mut jaa, mut jab, mut jbb, mut ga, mut gb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (&x, &y) in xs.iter().zip(&ys) {
            if x <= 0.0 {
                let r = 1.0 - y; // f(0) = 1, derivatives vanish
                let _ = r;
                continue;
            }
            let x2b = x.powf(2.0 * b);
            let den = 1.0 + a * x2b;
            let f = 1.0 / den;
            let r = f - y;
            let da = -x2b / (den * den);
            let db = -2.0 * a * x2b * x.ln() / (den * den);
            jaa += da * da;
            jab += da * db;
            jbb += db * db;
            ga += da * r;
            gb += db * r;
        }
        let (m_aa, m_bb) = (jaa * (1.0 + lambda), jbb * (1.0 + lambda));
        let det = m_aa * m_bb - jab * jab;
        if det.abs() < 1e-300 {
            break;
        }
        let step_a = -(m_bb * ga - jab * gb) / det;
        let step_b = -(-jab * ga + m_aa * gb) / det;
        let (na, nb) = (a + step_a, b + step_b);
        let new = if na > 0.0 && nb > 0.0 { sse(na, nb) } else { f64::INFINITY };
        if new < cur {
            let done = (cur - new) < 1e-15 * cur.max(1e-300);
            a = na;
            b = nb;
            cur = new;
            lambda = (lambda / 10.0).max(1e-12);
            if done {
                break;
            }
        } else {
            lambda *= 10.0;
            if lambda > 1e12 {
                break;
            }
        }
    }
    (a as f32, b as f32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ab_params() {
        // reference values from umap-learn (scipy curve_fit)
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!((a - 1.5769).abs() < 2e-3 && (b - 0.8951).abs() < 2e-3, "{} {}", a, b);
        let (a, b) = find_ab_params(1.0, 0.5);
        assert!((a - 0.5830).abs() < 2e-3 && (b - 1.3342).abs() < 2e-3, "{} {}", a, b);
    }

    /// Largest absolute difference allowed against umap-learn's `find_ab_params` (scipy `curve_fit`).
    const AB_GRID_TOL: f64 = 1.3e-5;

    #[test]
    fn find_ab_params_matches_umap_learn_on_a_grid() {
        // (spread, min_dist, a, b) from `umap.umap_.find_ab_params` (umap-learn 0.5.12 and 0.6.0
        // agree to the digits printed here), 3 spreads x 6 min_dist values.
        let grid: [(f32, f32, f64, f64); 18] = [
        (0.5, 0.001, 5.776298973, 0.792515306),
        (0.5, 0.05, 5.453765169, 0.895060801),
        (0.5, 0.1, 5.069309328, 1.003005423),
        (0.5, 0.25, 3.706290205, 1.334167248),
        (0.5, 0.4, 2.386754026, 1.681234620),
        (0.5, 0.5, 1.667716869, 1.929240438),
        (1.0, 0.001, 1.929073395, 0.791504533),
        (1.0, 0.01, 1.895605866, 0.800637844),
        (1.0, 0.1, 1.576943460, 0.895060878),
        (1.0, 0.25, 1.121436343, 1.057499877),
        (1.0, 0.5, 0.583030021, 1.334166992),
        (1.0, 0.9, 0.164903885, 1.803038468),
        (2.0, 0.01, 0.635344408, 0.795552890),
        (2.0, 0.1, 0.544660541, 0.842055426),
        (2.0, 0.5, 0.258878741, 1.057499700),
        (2.0, 1.0, 0.091715509, 1.334166562),
        (2.0, 1.5, 0.028874204, 1.621700558),
        (2.0, 2.0, 0.007926773, 1.929231525),
        ];
        let mut worst = 0.0f64;
        for &(spread, min_dist, a_ref, b_ref) in grid.iter() {
            let (a, b) = find_ab_params(spread, min_dist);
            let d = (a as f64 - a_ref).abs().max((b as f64 - b_ref).abs());
            worst = worst.max(d);
            assert!(d <= AB_GRID_TOL, "spread {} min_dist {}: a {} (ref {}), b {} (ref {})", spread, min_dist, a, a_ref, b, b_ref);
        }
        println!("max |difference| in (a, b) over the grid: {:e}", worst);
    }
}
