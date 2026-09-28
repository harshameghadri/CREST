//! DESeq2 (Love, Huber & Anders 2014, *Genome Biology* 15:550) in Rust, for
//! pseudobulk and bulk count matrices.
//!
//! This follows DESeq2 1.42's `DESeq()` + `results()` step by step (the R and
//! C++ sources were used as the reference), so results agree with R:
//!
//! 1. size factors: median-of-ratios (`type = "ratio"` or `"poscounts"`);
//! 2. gene-wise dispersions: moments/rough initial value, then a line search
//!    on the Cox-Reid adjusted profile likelihood of log(alpha) (`fitDisp`),
//!    with a grid search for rows that do not converge;
//! 3. dispersion trend: parametric `a + b / mean` (Gamma-family GLM with
//!    outlier trimming); falls back to the mean if that fit fails;
//! 4. MAP dispersions: log-normal prior centred on the trend, prior variance
//!    from the residual spread minus its sampling variance;
//! 5. Wald test: negative-binomial GLM by IRLS with a tiny ridge, Newton
//!    refinement (DESeq2 uses L-BFGS-B) for rows that do not converge;
//! 6. Cook's distances, and outlier replacement + refit when a design cell
//!    has >= 7 replicates.
//!
//! `results()` (contrasts, Cook's filtering, independent filtering) lives in
//! the Python layer, which calls [`lowess`] from here.
//!
//! Every gene is independent after steps 1, 3 and 4, so each step is one
//! parallel pass over genes.

mod linalg;
pub mod lowess;

use linalg::{xb, xtwx, xtwz, Lu};
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, FisherSnedecor, Normal};
use statrs::function::gamma::{digamma, ln_gamma};

const LN2: f64 = std::f64::consts::LN_2;

// ------------------------------------------------------------------ numerics

/// Stirling-series remainder: lgamma(x) - [(x - 1/2) ln x - x + ln(2π)/2], x >= 15.
fn stirlerr(x: f64) -> f64 {
    let x2 = 1.0 / (x * x);
    (1.0 / x) * (1.0 / 12.0 - x2 * (1.0 / 360.0 - x2 * (1.0 / 1260.0 - x2 / 1680.0)))
}

/// lgamma(y + r) - lgamma(r) - y ln(mu + r) - r ln(1 + mu / r): the part of the
/// NB log-likelihood (size r = 1/alpha) that depends on r, evaluated without the
/// cancellation that the direct formula suffers when r is large.
#[inline]
fn nb_core(y: f64, mu: f64, r: f64) -> f64 {
    if r < 15.0 {
        ln_gamma(y + r) - ln_gamma(r) - y * (mu + r).ln() - r * (mu / r).ln_1p()
    } else {
        (r + y - 0.5) * (y / r).ln_1p() - (r + y) * (mu / r).ln_1p() - y + stirlerr(r + y) - stirlerr(r)
    }
}

/// log dnbinom(y; mu, size r).
#[inline]
fn dnbinom_log(y: f64, mu: f64, r: f64) -> f64 {
    nb_core(y, mu, r) + if y > 0.0 { y * mu.ln() } else { 0.0 } - ln_gamma(y + 1.0)
}

fn digamma_tail(x: f64) -> f64 {
    // psi(x) - ln(x), asymptotic, x >= 15
    let x2 = 1.0 / (x * x);
    -0.5 / x - x2 * (1.0 / 12.0 - x2 * (1.0 / 120.0 - x2 * (1.0 / 252.0 - x2 / 240.0)))
}

/// r² · [psi(r) + ln(1+mu/r) - mu/(r+mu) - psi(y+r) + y/(mu+r)], the per-sample
/// term of d loglik / d alpha (DESeq2's `dlog_posterior`), stably.
#[inline]
fn dll_term(y: f64, mu: f64, r: f64) -> f64 {
    if r > 1e6 * (1.0 + y + mu) {
        return 0.5 * ((y - mu) * (y - mu) - y); // Poisson limit of the bracket × r²
    }
    let dpsi = if r >= 15.0 { -(y / r).ln_1p() + digamma_tail(r) - digamma_tail(r + y) } else { digamma(r) - digamma(y + r) };
    r * r * (dpsi + (mu / r).ln_1p() - mu / (r + mu) + y / (mu + r))
}

pub fn trigamma(mut x: f64) -> f64 {
    let mut r = 0.0;
    while x < 15.0 {
        r += 1.0 / (x * x);
        x += 1.0;
    }
    let x2 = 1.0 / (x * x);
    r + 1.0 / x + x2 / 2.0 + (1.0 / x) * x2 * (1.0 / 6.0 - x2 * (1.0 / 30.0 - x2 * (1.0 / 42.0 - x2 / 30.0)))
}

fn median(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// R's `mean(x, trim)`.
fn trimmed_mean(v: &[f64], trim: f64) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    if trim <= 0.0 {
        return v.iter().sum::<f64>() / n as f64;
    }
    let mut s = v.to_vec();
    if trim >= 0.5 {
        return median(&mut s);
    }
    let lo = (n as f64 * trim).floor() as usize; // 0-based: R's lo - 1
    let hi = n - lo;
    s.sort_by(|a, b| a.total_cmp(b));
    s[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
}

/// R's `mad(x)` (constant 1.4826).
fn mad(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    let med = median(&mut s);
    let mut d: Vec<f64> = v.iter().map(|x| (x - med).abs()).collect();
    1.4826 * median(&mut d)
}

// ------------------------------------------------------------------ design

/// A full-rank model matrix and the quantities every gene reuses.
pub struct Design {
    pub x: Vec<f64>, // m × p row-major
    pub m: usize,
    pub p: usize,
    /// (XᵀX)⁻¹Xᵀ, p × m: OLS coefficients of a row vector
    ols: Vec<f64>,
    /// X (XᵀX)⁻¹ Xᵀ, m × m: OLS fitted values (DESeq2 `linearModelMu`)
    hat: Vec<f64>,
    /// number of samples whose model-matrix row is identical to this sample's
    pub cell_size: Vec<usize>,
    /// cell id (first occurrence order) of each sample
    pub cell: Vec<usize>,
    pub n_cells: usize,
}

impl Design {
    pub fn new(x: Vec<f64>, m: usize, p: usize) -> Result<Design, String> {
        if x.len() != m * p || p == 0 {
            return Err("model matrix must be (n_samples, n_coefs)".into());
        }
        let ones = vec![1.0; m];
        let lu = Lu::new(xtwx(&x, m, p, &ones), p)
            .ok_or("the model matrix is not full rank, so the model cannot be fit as specified")?;
        let inv = lu.inverse();
        let mut ols = vec![0.0; p * m];
        for a in 0..p {
            for j in 0..m {
                ols[a * m + j] = (0..p).map(|b| inv[a * p + b] * x[j * p + b]).sum();
            }
        }
        let mut hat = vec![0.0; m * m];
        for i in 0..m {
            for j in 0..m {
                hat[i * m + j] = (0..p).map(|a| x[i * p + a] * ols[a * m + j]).sum();
            }
        }
        let mut cell = vec![usize::MAX; m];
        let mut reps: Vec<usize> = Vec::new();
        for i in 0..m {
            if cell[i] != usize::MAX {
                continue;
            }
            cell[i] = reps.len();
            for j in i + 1..m {
                if cell[j] == usize::MAX && x[i * p..(i + 1) * p] == x[j * p..(j + 1) * p] {
                    cell[j] = reps.len();
                }
            }
            reps.push(i);
        }
        let n_cells = reps.len();
        let mut count = vec![0usize; n_cells];
        for &c in &cell {
            count[c] += 1;
        }
        let cell_size = cell.iter().map(|&c| count[c]).collect();
        Ok(Design { x, m, p, ols, hat, cell_size, cell, n_cells })
    }

    /// DESeq2's `linearMu`: as many distinct rows as coefficients (a cell-means model).
    fn linear_mu(&self) -> bool {
        self.n_cells == self.p
    }

    fn n_or_more(&self, n: f64) -> Vec<bool> {
        self.cell_size.iter().map(|&s| s as f64 >= n).collect()
    }
}

// ------------------------------------------------------------------ parameters / output

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FitType {
    Parametric,
    Mean,
}

#[derive(Clone, Debug)]
pub struct Params {
    pub min_disp: f64,
    pub kappa_0: f64,
    pub disp_tol: f64,
    pub maxit: usize,
    pub beta_tol: f64,
    pub beta_maxit: usize,
    pub min_mu: f64,
    pub outlier_sd: f64,
    pub min_replicates_for_replace: f64,
    pub fit_type: FitType,
}

impl Default for Params {
    fn default() -> Self {
        Params {
            min_disp: 1e-8,
            kappa_0: 1.0,
            disp_tol: 1e-6,
            maxit: 100,
            beta_tol: 1e-8,
            beta_maxit: 100,
            min_mu: 0.5,
            outlier_sd: 2.0,
            min_replicates_for_replace: 7.0,
            fit_type: FitType::Parametric,
        }
    }
}

/// Per-gene outputs (length n_genes; NaN / false for all-zero genes).
pub struct Fit {
    pub size_factors: Vec<f64>,
    pub base_mean: Vec<f64>,
    pub base_var: Vec<f64>,
    pub all_zero: Vec<bool>,
    pub disp_gene_est: Vec<f64>,
    pub disp_gene_iter: Vec<f64>,
    pub disp_fit: Vec<f64>,
    pub disp_map: Vec<f64>,
    pub dispersion: Vec<f64>,
    pub disp_outlier: Vec<bool>,
    /// log2-scale coefficients, n_genes × p
    pub beta: Vec<f64>,
    /// log2-scale coefficient covariance, n_genes × p × p
    pub beta_cov: Vec<f64>,
    pub beta_conv: Vec<bool>,
    pub beta_iter: Vec<f64>,
    pub deviance: Vec<f64>,
    /// Cook's distances of the original fit, n_genes × m
    pub cooks: Vec<f64>,
    pub max_cooks: Vec<f64>,
    pub replace: Vec<bool>,
    /// counts after outlier replacement (only when replacement happened)
    pub replace_counts: Option<Vec<f64>>,
    pub fit_type: FitType,
    pub trend: Vec<f64>,
    pub disp_prior_var: f64,
    pub var_log_disp_ests: f64,
    pub messages: Vec<String>,
}

// ------------------------------------------------------------------ size factors

/// Median-of-ratios size factors of a genes × samples matrix.
pub fn size_factors(counts: &[f64], g: usize, m: usize, poscounts: bool) -> Result<Vec<f64>, String> {
    let loggeo: Vec<f64> = (0..g)
        .into_par_iter()
        .map(|i| {
            let row = &counts[i * m..(i + 1) * m];
            if poscounts {
                if row.iter().all(|&c| c == 0.0) {
                    f64::NEG_INFINITY
                } else {
                    row.iter().map(|&c| if c > 0.0 { c.ln() } else { 0.0 }).sum::<f64>() / m as f64
                }
            } else {
                row.iter().map(|&c| c.ln()).sum::<f64>() / m as f64
            }
        })
        .collect();
    if loggeo.iter().all(|v| v.is_infinite()) {
        return Err("every gene contains at least one zero, cannot compute log geometric means; use sf_type=\"poscounts\"".into());
    }
    let mut sf: Vec<f64> = (0..m)
        .into_par_iter()
        .map(|j| {
            let mut r: Vec<f64> = (0..g)
                .filter(|&i| loggeo[i].is_finite() && counts[i * m + j] > 0.0)
                .map(|i| counts[i * m + j].ln() - loggeo[i])
                .collect();
            median(&mut r).exp()
        })
        .collect();
    if poscounts {
        let lm = sf.iter().map(|s| s.ln()).sum::<f64>() / m as f64;
        sf.iter_mut().for_each(|s| *s /= lm.exp());
    }
    if sf.iter().any(|s| !s.is_finite() || *s <= 0.0) {
        return Err("a sample has no positive counts in genes usable for size factors".into());
    }
    Ok(sf)
}

// ------------------------------------------------------------------ dispersion likelihood

/// Cox-Reid adjusted log posterior of log(alpha) (DESeq2 `log_posterior`).
fn log_posterior(la: f64, y: &[f64], mu: &[f64], d: &Design, prior: Option<(f64, f64)>) -> f64 {
    let alpha = la.exp();
    let r = 1.0 / alpha;
    let ll: f64 = y.iter().zip(mu).map(|(&y, &mu)| nb_core(y, mu, r)).sum();
    let w: Vec<f64> = mu.iter().map(|&mu| 1.0 / (1.0 / mu + alpha)).collect();
    let cr = match Lu::new(xtwx(&d.x, d.m, d.p, &w), d.p) {
        Some(lu) => -0.5 * lu.log_abs_det(),
        None => f64::INFINITY,
    };
    let pr = prior.map_or(0.0, |(mean, s2)| -0.5 * (la - mean) * (la - mean) / s2);
    ll + pr + cr
}

/// d log_posterior / d log(alpha) (DESeq2 `dlog_posterior`).
fn dlog_posterior(la: f64, y: &[f64], mu: &[f64], d: &Design, prior: Option<(f64, f64)>) -> f64 {
    let alpha = la.exp();
    let r = 1.0 / alpha;
    let ll: f64 = y.iter().zip(mu).map(|(&y, &mu)| dll_term(y, mu, r)).sum();
    let w: Vec<f64> = mu.iter().map(|&mu| 1.0 / (1.0 / mu + alpha)).collect();
    let dw: Vec<f64> = w.iter().map(|&w| -w * w).collect();
    let cr = match Lu::new(xtwx(&d.x, d.m, d.p, &w), d.p) {
        Some(lu) => {
            let binv = lu.inverse();
            let db = xtwx(&d.x, d.m, d.p, &dw);
            let p = d.p;
            let tr: f64 = (0..p).map(|i| (0..p).map(|k| binv[i * p + k] * db[k * p + i]).sum::<f64>()).sum();
            -0.5 * tr
        }
        None => 0.0,
    };
    let pr = prior.map_or(0.0, |(mean, s2)| -(la - mean) / s2);
    (ll + cr) * alpha + pr
}

struct DispFit {
    log_alpha: f64,
    iter: usize,
    initial_lp: f64,
    last_lp: f64,
}

/// Backtracking line search on log(alpha) (DESeq2 `fitDisp`).
fn fit_disp(a0: f64, y: &[f64], mu: &[f64], d: &Design, prior: Option<(f64, f64)>, min_log_alpha: f64, pa: &Params) -> DispFit {
    let eps = 1e-4;
    let mut a = a0;
    let mut lp = log_posterior(a, y, mu, d, prior);
    let mut dlp = dlog_posterior(a, y, mu, d, prior);
    let initial_lp = lp;
    let mut kappa = pa.kappa_0;
    let (mut iter, mut accept) = (0usize, 0usize);
    for _ in 0..pa.maxit {
        iter += 1;
        let prop = a + kappa * dlp;
        if prop < -30.0 {
            kappa = (-30.0 - a) / dlp;
        }
        if prop > 10.0 {
            kappa = (10.0 - a) / dlp;
        }
        let theta_kappa = -log_posterior(a + kappa * dlp, y, mu, d, prior);
        let theta_hat_kappa = -lp - kappa * eps * dlp * dlp;
        if theta_kappa <= theta_hat_kappa {
            accept += 1;
            a += kappa * dlp;
            let lpnew = log_posterior(a, y, mu, d, prior);
            let change = lpnew - lp;
            if change < pa.disp_tol {
                lp = lpnew;
                break;
            }
            if a < min_log_alpha {
                break;
            }
            lp = lpnew;
            dlp = dlog_posterior(a, y, mu, d, prior);
            kappa = (kappa * 1.1).min(pa.kappa_0);
            if accept % 5 == 0 {
                kappa /= 2.0;
            }
        } else {
            kappa /= 2.0;
        }
    }
    DispFit { log_alpha: a, iter, initial_lp, last_lp: lp }
}

/// Coarse-then-fine grid maximisation of the log posterior (DESeq2 `fitDispGrid`).
fn fit_disp_grid(y: &[f64], mu: &[f64], d: &Design, prior: Option<(f64, f64)>) -> f64 {
    let n = 20;
    let (lo, hi) = (1e-8f64.ln(), (10f64.max(d.m as f64)).ln());
    let grid: Vec<f64> = (0..n).map(|t| lo + (hi - lo) * t as f64 / (n - 1) as f64).collect();
    let argmax = |g: &[f64]| {
        let mut best = (0usize, f64::NEG_INFINITY);
        for (t, &a) in g.iter().enumerate() {
            let v = log_posterior(a, y, mu, d, prior);
            if v > best.1 {
                best = (t, v);
            }
        }
        g[best.0]
    };
    let a_hat = argmax(&grid);
    let delta = grid[1] - grid[0];
    let fine: Vec<f64> = (0..n).map(|t| a_hat - delta + 2.0 * delta * t as f64 / (n - 1) as f64).collect();
    argmax(&fine).exp()
}

// ------------------------------------------------------------------ GLM fit

struct BetaFit {
    beta: Vec<f64>, // natural log scale
    cov: Vec<f64>,  // natural log scale
    hat: Vec<f64>,
    mu: Vec<f64>, // sf * exp(X beta), not clamped
    iter: usize,
    conv: bool,
    deviance: f64,
}

/// NB GLM by IRLS (DESeq2 `fitBeta`), with a Newton refit on the penalised
/// likelihood for rows IRLS does not converge on (DESeq2: L-BFGS-B).
fn fit_beta(y: &[f64], sf: &[f64], d: &Design, alpha: f64, pa: &Params) -> BetaFit {
    let (m, p) = (d.m, d.p);
    let lambda = 1e-6 / (LN2 * LN2);
    let r = 1.0 / alpha;
    // initial beta: OLS of log(normalised counts + 0.1)
    let ly: Vec<f64> = y.iter().zip(sf).map(|(&y, &s)| (y / s + 0.1).ln()).collect();
    let init: Vec<f64> = (0..p).map(|a| (0..m).map(|j| d.ols[a * m + j] * ly[j]).sum()).collect();
    let mut beta = init.clone();
    let mut eta = vec![0.0; m];
    let clamp_mu = |beta: &[f64], eta: &mut [f64]| -> Vec<f64> {
        xb(&d.x, m, p, beta, eta);
        eta.iter().zip(sf).map(|(&e, &s)| (s * e.exp()).max(pa.min_mu)).collect()
    };
    let mut mu = clamp_mu(&beta, &mut eta);
    let mut dev_old = 0.0;
    let mut iter = 0;
    let mut w = vec![0.0; m];
    let mut z = vec![0.0; m];
    for t in 0..pa.beta_maxit {
        iter += 1;
        for j in 0..m {
            w[j] = mu[j] / (1.0 + alpha * mu[j]);
            z[j] = (mu[j] / sf[j]).ln() + (y[j] - mu[j]) / mu[j];
        }
        let mut a = xtwx(&d.x, m, p, &w);
        for k in 0..p {
            a[k * p + k] += lambda;
        }
        let b = xtwz(&d.x, m, p, &w, &z);
        match Lu::new(a, p) {
            Some(lu) => beta = lu.solve(&b),
            None => {
                iter = pa.beta_maxit;
                break;
            }
        }
        if beta.iter().any(|v| v.abs() > 30.0 || !v.is_finite()) {
            iter = pa.beta_maxit;
            break;
        }
        mu = clamp_mu(&beta, &mut eta);
        let dev = -2.0 * y.iter().zip(&mu).map(|(&y, &mu)| dnbinom_log(y, mu, r)).sum::<f64>();
        let conv = (dev - dev_old).abs() / (dev.abs() + 0.1);
        if conv.is_nan() {
            iter = pa.beta_maxit;
            break;
        }
        if t > 0 && conv < pa.beta_tol {
            break;
        }
        dev_old = dev;
    }
    let mut conv = iter < pa.beta_maxit;
    let unstable = beta.iter().any(|v| !v.is_finite());
    let mut out = finish_beta(&beta, y, sf, d, alpha, lambda, pa.min_mu, r);
    out.iter = iter;
    let var_ok = (0..p).all(|k| out.cov[k * p + k] > 0.0);
    if !conv || unstable || !var_ok {
        let start = if !unstable && beta.iter().all(|v| v.abs() < 30.0 * LN2) { beta } else { init };
        let (b2, ok) = newton_beta(start, y, sf, d, alpha, lambda);
        let mut o2 = finish_beta(&b2, y, sf, d, alpha, lambda, pa.min_mu, r);
        o2.iter = iter;
        conv = ok;
        out = o2;
    }
    out.conv = conv;
    out
}

#[allow(clippy::too_many_arguments)]
fn finish_beta(beta: &[f64], y: &[f64], sf: &[f64], d: &Design, alpha: f64, lambda: f64, min_mu: f64, r: f64) -> BetaFit {
    let (m, p) = (d.m, d.p);
    let mut eta = vec![0.0; m];
    xb(&d.x, m, p, beta, &mut eta);
    let mu: Vec<f64> = eta.iter().zip(sf).map(|(&e, &s)| s * e.exp()).collect();
    let muc: Vec<f64> = mu.iter().map(|&v| v.max(min_mu)).collect();
    let w: Vec<f64> = muc.iter().map(|&v| v / (1.0 + alpha * v)).collect();
    let xtwx_ = xtwx(&d.x, m, p, &w);
    let mut ridged = xtwx_.clone();
    for k in 0..p {
        ridged[k * p + k] += lambda;
    }
    let (cov, hat) = match Lu::new(ridged, p) {
        Some(lu) => {
            let inv = lu.inverse();
            let cov = linalg::matmul(&linalg::matmul(&inv, &xtwx_, p), &inv, p);
            let hat = (0..m)
                .map(|j| {
                    let xr = &d.x[j * p..(j + 1) * p];
                    let mut h = 0.0;
                    for a in 0..p {
                        for b in 0..p {
                            h += xr[a] * inv[b * p + a] * xr[b];
                        }
                    }
                    h * w[j]
                })
                .collect();
            (cov, hat)
        }
        None => (vec![f64::NAN; p * p], vec![f64::NAN; m]),
    };
    // DESeq2 reports the deviance at the unclamped fitted means
    let deviance = -2.0 * y.iter().zip(&mu).map(|(&y, &mu)| dnbinom_log(y, mu, r)).sum::<f64>();
    BetaFit { beta: beta.to_vec(), cov, hat, mu, iter: 0, conv: true, deviance }
}

/// Maximise the ridge-penalised NB log-likelihood by Newton's method with step
/// halving (the objective is concave in beta), keeping |beta| <= 30 on the log2
/// scale as DESeq2's L-BFGS-B bounds do.
fn newton_beta(mut beta: Vec<f64>, y: &[f64], sf: &[f64], d: &Design, alpha: f64, lambda: f64) -> (Vec<f64>, bool) {
    let (m, p) = (d.m, d.p);
    let r = 1.0 / alpha;
    let bound = 30.0 * LN2;
    let mut eta = vec![0.0; m];
    let obj = |b: &[f64], eta: &mut [f64]| -> f64 {
        xb(&d.x, m, p, b, eta);
        let ll: f64 = (0..m).map(|j| dnbinom_log(y[j], sf[j] * eta[j].exp(), r)).sum();
        ll - 0.5 * lambda * b.iter().map(|v| v * v).sum::<f64>()
    };
    beta.iter_mut().for_each(|v| *v = v.clamp(-bound, bound));
    let mut f = obj(&beta, &mut eta);
    for _ in 0..200 {
        xb(&d.x, m, p, &beta, &mut eta);
        let mut g = vec![0.0; p];
        let mut h = vec![0.0; m];
        let mut resid = vec![0.0; m];
        for j in 0..m {
            let mu = sf[j] * eta[j].exp();
            resid[j] = (y[j] - mu) / (1.0 + alpha * mu);
            h[j] = mu * (1.0 + alpha * y[j]) / ((1.0 + alpha * mu) * (1.0 + alpha * mu));
        }
        for j in 0..m {
            for a in 0..p {
                g[a] += d.x[j * p + a] * resid[j];
            }
        }
        for a in 0..p {
            g[a] -= lambda * beta[a];
        }
        let mut hm = xtwx(&d.x, m, p, &h);
        for k in 0..p {
            hm[k * p + k] += lambda;
        }
        let step = match Lu::new(hm, p) {
            Some(lu) => lu.solve(&g),
            None => return (beta, false),
        };
        let mut t = 1.0;
        let mut improved = false;
        for _ in 0..60 {
            let cand: Vec<f64> = beta.iter().zip(&step).map(|(b, s)| (b + t * s).clamp(-bound, bound)).collect();
            let fc = obj(&cand, &mut eta);
            if fc >= f {
                let gain = fc - f;
                beta = cand;
                f = fc;
                improved = true;
                if gain < 1e-10 * (f.abs() + 0.1) {
                    return (beta, true);
                }
                break;
            }
            t /= 2.0;
        }
        if !improved {
            return (beta, true);
        }
    }
    (beta, false)
}

// ------------------------------------------------------------------ pipeline pieces

struct GeneEst {
    disp: f64,
    iter: f64,
    mu: Vec<f64>,
}

/// DESeq2 `estimateDispersionsGeneEst` for one gene (niter = 1).
fn gene_est(y: &[f64], sf: &[f64], d: &Design, xim: f64, pa: &Params) -> GeneEst {
    let (m, p) = (d.m, d.p);
    let max_disp = 10f64.max(m as f64);
    let yn: Vec<f64> = y.iter().zip(sf).map(|(&y, &s)| y / s).collect();
    let bm = yn.iter().sum::<f64>() / m as f64;
    let bv = yn.iter().map(|v| (v - bm) * (v - bm)).sum::<f64>() / (m as f64 - 1.0);
    let lin: Vec<f64> = (0..m).map(|i| (0..m).map(|j| d.hat[i * m + j] * yn[j]).sum()).collect();
    let rough = {
        let s: f64 = yn.iter().zip(&lin).map(|(&y, &mu)| {
            let mu = mu.max(1.0);
            ((y - mu) * (y - mu) - mu) / (mu * mu)
        }).sum();
        (s / (m - p) as f64).max(0.0)
    };
    let moments = (bv - xim * bm) / (bm * bm);
    let alpha_init = rough.min(moments).max(pa.min_disp).min(max_disp);
    let mu: Vec<f64> = if d.linear_mu() {
        lin.iter().zip(sf).map(|(&l, &s)| (l * s).max(pa.min_mu)).collect()
    } else {
        fit_beta(y, sf, d, alpha_init, pa).mu.iter().map(|&v| v.max(pa.min_mu)).collect()
    };
    let la = alpha_init.ln();
    let fd = fit_disp(la, y, &mu, d, None, (pa.min_disp / 10.0).ln(), pa);
    let mut disp = fd.log_alpha.exp().min(max_disp);
    if fd.last_lp < fd.initial_lp + fd.initial_lp.abs() / 1e6 {
        disp = alpha_init;
    }
    let conv = fd.iter < pa.maxit && fd.iter != 1;
    if !conv && disp > pa.min_disp * 10.0 {
        disp = fit_disp_grid(y, &mu, d, None);
    }
    GeneEst { disp: disp.max(pa.min_disp).min(max_disp), iter: fd.iter as f64, mu }
}

/// R `glm(disps ~ I(1/means), family = Gamma(link = "identity"), start = coefs)`.
fn gamma_identity_glm(means: &[f64], disps: &[f64], start: [f64; 2]) -> Option<([f64; 2], bool)> {
    let n = means.len();
    let x: Vec<f64> = means.iter().flat_map(|&mn| [1.0, 1.0 / mn]).collect();
    let dev_of = |c: &[f64; 2]| -> Option<f64> {
        let mut dev = 0.0;
        for i in 0..n {
            let mu = c[0] + c[1] * x[2 * i + 1];
            if !(mu > 0.0) || !mu.is_finite() {
                return None;
            }
            let y = disps[i];
            dev += -2.0 * ((if y == 0.0 { 1.0 } else { y / mu }).ln() - (y - mu) / mu);
        }
        Some(dev)
    };
    let mut coef = start;
    let mut coefold = start;
    let mut devold = dev_of(&coef)?;
    for _ in 0..25 {
        let w: Vec<f64> = (0..n).map(|i| {
            let mu = coef[0] + coef[1] * x[2 * i + 1];
            1.0 / (mu * mu)
        }).collect();
        let a = xtwx(&x, n, 2, &w);
        let b = xtwz(&x, n, 2, &w, disps);
        let sol = Lu::new(a, 2)?.solve(&b);
        let mut cand = [sol[0], sol[1]];
        let mut dev = dev_of(&cand);
        let mut ii = 0;
        while dev.is_none() {
            ii += 1;
            if ii > 25 {
                return None;
            }
            cand = [(cand[0] + coefold[0]) / 2.0, (cand[1] + coefold[1]) / 2.0];
            dev = dev_of(&cand);
        }
        let dev = dev.unwrap();
        coef = cand;
        if (dev - devold).abs() / (dev.abs() + 0.1) < 1e-8 {
            return Some((coef, true));
        }
        devold = dev;
        coefold = coef;
    }
    Some((coef, false))
}

/// DESeq2 `parametricDispersionFit`: disp = a + b / mean.
fn parametric_fit(means: &[f64], disps: &[f64]) -> Option<[f64; 2]> {
    let mut coefs = [0.1, 1.0];
    let mut iter = 0;
    loop {
        let (mg, dg): (Vec<f64>, Vec<f64>) = means
            .iter()
            .zip(disps)
            .filter(|(&mn, &dp)| {
                let r = dp / (coefs[0] + coefs[1] / mn);
                r > 1e-4 && r < 15.0
            })
            .map(|(&a, &b)| (a, b))
            .unzip();
        if mg.len() < 2 {
            return None;
        }
        let (new, converged) = gamma_identity_glm(&mg, &dg, coefs)?;
        let old = coefs;
        coefs = new;
        if !(coefs[0] > 0.0 && coefs[1] > 0.0) {
            return None;
        }
        let ch = (coefs[0] / old[0]).ln().powi(2) + (coefs[1] / old[1]).ln().powi(2);
        if ch < 1e-6 && converged {
            return Some(coefs);
        }
        iter += 1;
        if iter > 10 {
            return None;
        }
    }
}

/// Local quadratic regression with tricube weights (R `loess(span, degree = 2)`,
/// evaluated directly rather than through loess's interpolation surface).
fn loess_predict(x: &[f64], y: &[f64], span: f64, at: &[f64]) -> Vec<f64> {
    let n = x.len();
    let q = ((span * n as f64).floor() as usize).clamp(3, n);
    at.iter()
        .map(|&x0| {
            let mut dist: Vec<f64> = x.iter().map(|&xi| (xi - x0).abs()).collect();
            let mut sorted = dist.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));
            let h = sorted[q - 1].max(1e-12);
            let mut s = [0.0f64; 9]; // Σw, Σw t, Σw t², Σw t³, Σw t⁴, Σw y, Σw t y, Σw t² y
            for i in 0..n {
                let u = dist[i] / h;
                if u >= 1.0 {
                    continue;
                }
                let wv = (1.0 - u * u * u).powi(3);
                let t = x[i] - x0;
                s[0] += wv;
                s[1] += wv * t;
                s[2] += wv * t * t;
                s[3] += wv * t * t * t;
                s[4] += wv * t * t * t * t;
                s[5] += wv * y[i];
                s[6] += wv * t * y[i];
                s[7] += wv * t * t * y[i];
            }
            dist.clear();
            let a = vec![s[0], s[1], s[2], s[1], s[2], s[3], s[2], s[3], s[4]];
            match Lu::new(a, 3) {
                Some(lu) => lu.solve(&[s[5], s[6], s[7]])[0],
                None => s[5] / s[0],
            }
        })
        .collect()
}

/// DESeq2 `estimateDispersionsPriorVar` for (m - p) <= 3: the prior variance
/// whose implied residual distribution best matches the observed residuals
/// (KL divergence over a histogram). DESeq2 simulates the reference
/// distribution with 10⁴ draws; here the histogram of
/// log(χ²_k / k) + N(0, v) is integrated numerically, so it is deterministic.
fn prior_var_small_df(resid: &[f64], df: f64) -> f64 {
    let brks: Vec<f64> = (-20..=20).map(|v| v as f64 / 2.0).collect();
    let nb = brks.len() - 1;
    let hist = |vals: &mut dyn Iterator<Item = f64>| -> Vec<f64> {
        let mut h = vec![0.0; nb];
        let mut n = 0.0;
        for v in vals {
            if v > -10.0 && v < 10.0 {
                let k = (((v + 10.0) / 0.5).ceil() as usize).clamp(1, nb) - 1; // (a, b]
                h[k] += 1.0;
                n += 1.0;
            }
        }
        h.iter().map(|c| c / (n * 0.5)).collect()
    };
    let obs = hist(&mut resid.iter().copied());
    // quadrature for L = log(X/k), X ~ χ²_k
    let chi = ChiSquared::new(df).unwrap();
    let nq = 4000;
    let (llo, lhi) = (-40.0f64, 4.0f64);
    let mut lq = Vec::with_capacity(nq);
    let mut wq = Vec::with_capacity(nq);
    for i in 0..nq {
        let a = llo + (lhi - llo) * i as f64 / nq as f64;
        let b = llo + (lhi - llo) * (i + 1) as f64 / nq as f64;
        let pm = chi.cdf(df * b.exp()) - chi.cdf(df * a.exp());
        lq.push(0.5 * (a + b));
        wq.push(pm);
    }
    let nd = Normal::new(0.0, 1.0).unwrap();
    let grid: Vec<f64> = (0..200).map(|i| 8.0 * i as f64 / 199.0).collect();
    let kl: Vec<f64> = grid
        .par_iter()
        .map(|&v| {
            let sd = v.sqrt();
            let mut ph = vec![0.0; nb];
            for (l, w) in lq.iter().zip(&wq) {
                if *w == 0.0 {
                    continue;
                }
                for k in 0..nb {
                    let (a, b) = (brks[k], brks[k + 1]);
                    let pk = if sd == 0.0 {
                        if *l > a && *l <= b { 1.0 } else { 0.0 }
                    } else {
                        nd.cdf((b - l) / sd) - nd.cdf((a - l) / sd)
                    };
                    ph[k] += w * pk;
                }
            }
            let tot: f64 = ph.iter().sum();
            let rd: Vec<f64> = ph.iter().map(|p| p / (tot * 0.5)).collect();
            let small = obs.iter().chain(&rd).copied().filter(|&z| z > 0.0).fold(f64::INFINITY, f64::min);
            obs.iter().zip(&rd).map(|(&o, &r)| o * ((o + small).ln() - (r + small).ln())).sum()
        })
        .collect();
    let fine: Vec<f64> = (0..1000).map(|i| 8.0 * i as f64 / 999.0).collect();
    let fitted = loess_predict(&grid, &kl, 0.2, &fine);
    let mut best = (0usize, f64::INFINITY);
    for (i, &v) in fitted.iter().enumerate() {
        if v < best.1 {
            best = (i, v);
        }
    }
    fine[best.0].max(0.25)
}

struct DispResult {
    disp_fit: Vec<f64>,
    disp_map: Vec<f64>,
    dispersion: Vec<f64>,
    outlier: Vec<bool>,
}

/// DESeq2 `estimateDispersionsMAP` on a set of genes.
#[allow(clippy::too_many_arguments)]
fn disp_map(ys: &[&[f64]], mus: &[Vec<f64>], gene_est: &[f64], disp_fit: Vec<f64>, d: &Design, prior_var: f64, var_log: f64, pa: &Params) -> DispResult {
    let max_disp = 10f64.max(d.m as f64);
    let res: Vec<(f64, f64, bool)> = (0..ys.len())
        .into_par_iter()
        .map(|i| {
            let (ge, fit) = (gene_est[i], disp_fit[i]);
            let init = if ge > 0.1 * fit { ge } else { fit };
            let prior = Some((fit.ln(), prior_var));
            let fd = fit_disp(init.ln(), ys[i], &mus[i], d, prior, (pa.min_disp / 10.0).ln(), pa);
            let mut map = fd.log_alpha.exp();
            if fd.iter >= pa.maxit {
                map = fit_disp_grid(ys[i], &mus[i], d, prior);
            }
            map = map.max(pa.min_disp).min(max_disp);
            let outlier = ge.ln() > fit.ln() + pa.outlier_sd * var_log.sqrt();
            (map, if outlier { ge } else { map }, outlier)
        })
        .collect();
    DispResult {
        disp_fit,
        disp_map: res.iter().map(|r| r.0).collect(),
        dispersion: res.iter().map(|r| r.1).collect(),
        outlier: res.iter().map(|r| r.2).collect(),
    }
}

/// DESeq2 `robustMethodOfMomentsDisp` (dispersion used only for Cook's distances).
fn robust_mom_disp(yn: &[f64], d: &Design) -> f64 {
    let m = d.m;
    let trim_of = |n: usize| -> (f64, f64) {
        if (n as f64) <= 3.5 {
            (1.0 / 3.0, 2.04)
        } else if (n as f64) <= 23.5 {
            (1.0 / 4.0, 1.86)
        } else {
            (1.0 / 8.0, 1.51)
        }
    };
    let v = if d.cell_size.iter().any(|&s| s >= 3) {
        let mut best = f64::NEG_INFINITY;
        for c in 0..d.n_cells {
            let idx: Vec<usize> = (0..m).filter(|&j| d.cell[j] == c).collect();
            if idx.len() < 3 {
                continue;
            }
            let (tr, sc) = trim_of(idx.len());
            let vals: Vec<f64> = idx.iter().map(|&j| yn[j]).collect();
            let cm = trimmed_mean(&vals, tr);
            let sq: Vec<f64> = vals.iter().map(|v| (v - cm) * (v - cm)).collect();
            best = best.max(sc * trimmed_mean(&sq, tr));
        }
        best
    } else {
        let rm = trimmed_mean(yn, 1.0 / 8.0);
        let sq: Vec<f64> = yn.iter().map(|v| (v - rm) * (v - rm)).collect();
        1.51 * trimmed_mean(&sq, 1.0 / 8.0)
    };
    let mean = yn.iter().sum::<f64>() / m as f64;
    ((v - mean) / (mean * mean)).max(0.04)
}

// ------------------------------------------------------------------ driver

struct Stage {
    disp_gene_est: Vec<f64>,
    disp_gene_iter: Vec<f64>,
    disp: DispResult,
    fits: Vec<BetaFit>,
}

/// Dispersions (gene-wise, then MAP given a trend and prior) and the Wald fit for `genes`.
#[allow(clippy::too_many_arguments)]
fn fit_genes(
    counts: &[f64], genes: &[usize], sf: &[f64], d: &Design, pa: &Params, xim: f64,
    trend: &dyn Fn(&[f64], &[f64]) -> (Vec<f64>, f64, f64), base_mean: &[f64],
) -> (Stage, f64, f64) {
    let m = d.m;
    let ys: Vec<&[f64]> = genes.iter().map(|&i| &counts[i * m..(i + 1) * m]).collect();
    let ge: Vec<GeneEst> = ys.par_iter().map(|y| gene_est(y, sf, d, xim, pa)).collect();
    let gest: Vec<f64> = ge.iter().map(|g| g.disp).collect();
    let bm: Vec<f64> = genes.iter().map(|&i| base_mean[i]).collect();
    let (disp_fit, prior_var, var_log) = trend(&bm, &gest);
    let mus: Vec<Vec<f64>> = ge.iter().map(|g| g.mu.clone()).collect();
    let all_small = gest.iter().all(|&g| g < pa.min_disp * 100.0);
    let disp = if all_small {
        let n = genes.len();
        DispResult { disp_fit, disp_map: vec![pa.min_disp * 10.0; n], dispersion: vec![pa.min_disp * 10.0; n], outlier: vec![false; n] }
    } else {
        disp_map(&ys, &mus, &gest, disp_fit, d, prior_var, var_log, pa)
    };
    let fits: Vec<BetaFit> = (0..genes.len()).into_par_iter().map(|i| fit_beta(ys[i], sf, d, disp.dispersion[i], pa)).collect();
    (Stage { disp_gene_est: gest, disp_gene_iter: ge.iter().map(|g| g.iter).collect(), disp, fits }, prior_var, var_log)
}

/// Run DESeq2 on a genes × samples count matrix.
pub fn deseq(counts: &[f64], g: usize, d: &Design, sf: Option<Vec<f64>>, poscounts: bool, pa: &Params) -> Result<Fit, String> {
    let (m, p) = (d.m, d.p);
    if counts.len() != g * m {
        return Err("counts must be (n_genes, n_samples)".into());
    }
    if counts.iter().any(|&c| c < 0.0 || c.fract() != 0.0 || !c.is_finite()) {
        return Err("counts must be non-negative integers".into());
    }
    if m <= p {
        return Err("the number of samples and the number of model coefficients are equal, i.e., there are no replicates to estimate the dispersion".into());
    }
    let sf = match sf {
        Some(s) if s.len() == m => s,
        Some(_) => return Err("size_factors must have one value per sample".into()),
        None => size_factors(counts, g, m, poscounts)?,
    };
    let xim = sf.iter().map(|s| 1.0 / s).sum::<f64>() / m as f64;
    let mut messages = Vec::new();

    let mean_var = |c: &[f64]| -> (Vec<f64>, Vec<f64>, Vec<bool>) {
        let v: Vec<(f64, f64, bool)> = (0..g)
            .into_par_iter()
            .map(|i| {
                let row = &c[i * m..(i + 1) * m];
                let bm = row.iter().zip(&sf).map(|(y, s)| y / s).sum::<f64>() / m as f64;
                let bv = row.iter().zip(&sf).map(|(y, s)| (y / s - bm).powi(2)).sum::<f64>() / (m as f64 - 1.0);
                (bm, bv, row.iter().all(|&y| y == 0.0))
            })
            .collect();
        (v.iter().map(|t| t.0).collect(), v.iter().map(|t| t.1).collect(), v.iter().map(|t| t.2).collect())
    };
    let (mut base_mean, mut base_var, mut all_zero) = mean_var(counts);
    let nz: Vec<usize> = (0..g).filter(|&i| !all_zero[i]).collect();
    if nz.is_empty() {
        return Err("all genes have zero counts".into());
    }

    // trend + prior variance, decided on the first (full) pass and then frozen
    let fitted: std::sync::Mutex<Option<(FitType, Vec<f64>, f64, f64)>> = std::sync::Mutex::new(None);
    let fit_type_req = pa.fit_type;
    let msgs = std::sync::Mutex::new(Vec::<String>::new());
    let trend = |bm: &[f64], gest: &[f64]| -> (Vec<f64>, f64, f64) {
        let mut guard = fitted.lock().unwrap();
        let (ft, coefs, prior_var, var_log) = match guard.as_ref() {
            Some(t) => t.clone(),
            None => {
                let use_fit: Vec<usize> = (0..gest.len()).filter(|&i| gest[i] > 100.0 * pa.min_disp).collect();
                let mut ft = fit_type_req;
                let mut coefs = vec![];
                if use_fit.is_empty() {
                    msgs.lock().unwrap().push("all gene-wise dispersion estimates are within 2 orders of magnitude from the minimum value; using their mean as the trend".into());
                    ft = FitType::Mean;
                } else if ft == FitType::Parametric {
                    let mb: Vec<f64> = use_fit.iter().map(|&i| bm[i]).collect();
                    let dg: Vec<f64> = use_fit.iter().map(|&i| gest[i]).collect();
                    match parametric_fit(&mb, &dg) {
                        Some(c) => coefs = c.to_vec(),
                        None => {
                            msgs.lock().unwrap().push("the parametric dispersion trend (a + b/mean) did not fit; using fit_type=\"mean\" (DESeq2 would substitute a local regression)".into());
                            ft = FitType::Mean;
                        }
                    }
                }
                if ft == FitType::Mean {
                    let v: Vec<f64> = gest.iter().copied().filter(|&x| x > 10.0 * pa.min_disp).collect();
                    coefs = vec![if v.is_empty() { pa.min_disp * 10.0 } else { trimmed_mean(&v, 0.001) }];
                }
                let f = |mn: f64| if coefs.len() == 2 { coefs[0] + coefs[1] / mn } else { coefs[0] };
                let above: Vec<usize> = (0..gest.len()).filter(|&i| gest[i] >= pa.min_disp * 100.0).collect();
                let resid: Vec<f64> = above.iter().map(|&i| gest[i].ln() - f(bm[i]).ln()).collect();
                let var_log = if resid.is_empty() { f64::NAN } else { mad(&resid).powi(2) };
                let dfr = (m - p) as f64;
                let prior_var = if resid.is_empty() {
                    0.25
                } else if dfr <= 3.0 {
                    prior_var_small_df(&resid, dfr)
                } else {
                    (var_log - trigamma(dfr / 2.0)).max(0.25)
                };
                *guard = Some((ft, coefs.clone(), prior_var, var_log));
                (ft, coefs, prior_var, var_log)
            }
        };
        let fv = bm.iter().map(|&mn| if coefs.len() == 2 && ft == FitType::Parametric { coefs[0] + coefs[1] / mn } else { coefs[0] }).collect();
        (fv, prior_var, var_log)
    };

    let (stage, prior_var, var_log) = fit_genes(counts, &nz, &sf, d, pa, xim, &trend, &base_mean);
    messages.extend(msgs.lock().unwrap().drain(..));
    let (fit_type, trend_coefs) = {
        let t = fitted.lock().unwrap();
        let t = t.as_ref().unwrap();
        (t.0, t.1.clone())
    };

    let nan_g = || vec![f64::NAN; g];
    let mut out = Fit {
        size_factors: sf.clone(),
        base_mean: base_mean.clone(), base_var: base_var.clone(), all_zero: all_zero.clone(),
        disp_gene_est: nan_g(), disp_gene_iter: nan_g(), disp_fit: nan_g(), disp_map: nan_g(), dispersion: nan_g(),
        disp_outlier: vec![false; g], beta: vec![f64::NAN; g * p], beta_cov: vec![f64::NAN; g * p * p],
        beta_conv: vec![false; g], beta_iter: nan_g(), deviance: nan_g(), cooks: vec![f64::NAN; g * m],
        max_cooks: nan_g(), replace: vec![false; g], replace_counts: None,
        fit_type, trend: trend_coefs, disp_prior_var: prior_var, var_log_disp_ests: var_log, messages,
    };
    let write_stage = |out: &mut Fit, genes: &[usize], st: &Stage| {
        for (k, &i) in genes.iter().enumerate() {
            out.disp_gene_est[i] = st.disp_gene_est[k];
            out.disp_gene_iter[i] = st.disp_gene_iter[k];
            out.disp_fit[i] = st.disp.disp_fit[k];
            out.disp_map[i] = st.disp.disp_map[k];
            out.dispersion[i] = st.disp.dispersion[k];
            out.disp_outlier[i] = st.disp.outlier[k];
            let f = &st.fits[k];
            for a in 0..p {
                out.beta[i * p + a] = f.beta[a] / LN2;
                for b in 0..p {
                    out.beta_cov[(i * p + a) * p + b] = f.cov[a * p + b] / (LN2 * LN2);
                }
            }
            out.beta_conv[i] = f.conv;
            out.beta_iter[i] = f.iter as f64;
            out.deviance[i] = f.deviance;
        }
    };
    write_stage(&mut out, &nz, &stage);

    // Cook's distances
    let cooks_rows: Vec<Vec<f64>> = nz
        .par_iter()
        .zip(stage.fits.par_iter())
        .map(|(&i, f)| {
            let y = &counts[i * m..(i + 1) * m];
            let yn: Vec<f64> = y.iter().zip(&sf).map(|(y, s)| y / s).collect();
            let disp = robust_mom_disp(&yn, d);
            (0..m)
                .map(|j| {
                    let mu = f.mu[j];
                    let v = mu + disp * mu * mu;
                    let h = f.hat[j];
                    (y[j] - mu).powi(2) / v / p as f64 * h / ((1.0 - h) * (1.0 - h))
                })
                .collect()
        })
        .collect();
    for (k, &i) in nz.iter().enumerate() {
        out.cooks[i * m..(i + 1) * m].copy_from_slice(&cooks_rows[k]);
    }
    let for_cooks = d.n_or_more(3.0);
    let record_max = |cooks: &[f64], i: usize| -> f64 {
        if m > p && for_cooks.iter().any(|&b| b) {
            (0..m).filter(|&j| for_cooks[j]).map(|j| cooks[i * m + j]).fold(f64::NEG_INFINITY, f64::max)
        } else {
            f64::NAN
        }
    };
    for &i in &nz {
        out.max_cooks[i] = record_max(&out.cooks, i);
    }

    // outlier replacement and refit (DESeq2 refitWithoutOutliers)
    let replaceable = d.n_or_more(pa.min_replicates_for_replace);
    if replaceable.iter().any(|&b| b) && m > p {
        let cutoff = FisherSnedecor::new(p as f64, (m - p) as f64).unwrap().inverse_cdf(0.99);
        let mut new_counts = counts.to_vec();
        let mut nrefit = 0usize;
        for &i in &nz {
            let row = &out.cooks[i * m..(i + 1) * m];
            if !row.iter().any(|&c| c > cutoff) {
                continue;
            }
            out.replace[i] = true;
            nrefit += 1;
            let yn: Vec<f64> = counts[i * m..(i + 1) * m].iter().zip(&sf).map(|(y, s)| y / s).collect();
            let tbm = trimmed_mean(&yn, 0.2);
            for j in 0..m {
                if replaceable[j] && row[j] > cutoff {
                    new_counts[i * m + j] = (tbm * sf[j]).trunc();
                }
            }
        }
        if nrefit > 0 {
            let (bm2, bv2, az2) = mean_var(&new_counts);
            base_mean = bm2;
            base_var = bv2;
            all_zero = az2;
            let refit: Vec<usize> = (0..g).filter(|&i| out.replace[i] && !all_zero[i]).collect();
            let new_zero: Vec<usize> = (0..g).filter(|&i| out.replace[i] && all_zero[i]).collect();
            out.base_mean = base_mean.clone();
            out.base_var = base_var.clone();
            if !refit.is_empty() {
                out.messages.push(format!("replacing outliers and refitting for {} genes (min_replicates_for_replace = {})", nrefit, pa.min_replicates_for_replace));
                let (st, _, _) = fit_genes(&new_counts, &refit, &sf, d, pa, xim, &trend, &base_mean);
                write_stage(&mut out, &refit, &st);
                for &i in &new_zero {
                    for a in 0..p {
                        out.beta[i * p + a] = f64::NAN;
                    }
                    out.deviance[i] = f64::NAN;
                }
                if replaceable.iter().all(|&b| b) {
                    out.max_cooks.iter_mut().for_each(|v| *v = f64::NAN);
                } else {
                    let mut rc = out.cooks.clone();
                    for i in 0..g {
                        for j in 0..m {
                            if replaceable[j] {
                                rc[i * m + j] = 0.0;
                            }
                        }
                    }
                    for i in 0..g {
                        out.max_cooks[i] = if out.all_zero[i] { f64::NAN } else { record_max(&rc, i) };
                    }
                }
            }
            out.replace_counts = Some(new_counts);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nb_core_matches_direct_formula() {
        for &(y, mu, r) in &[(0.0, 3.0, 20.0), (7.0, 5.5, 40.0), (120.0, 90.0, 1e4), (3.0, 2.0, 1e7)] {
            let direct = ln_gamma(y + r) - ln_gamma(r) - y * (mu + r).ln() - r * (1.0 + mu / r).ln();
            let tol = 1e-9 * (1.0 + direct.abs()) + if r > 1e6 { 1e-6 } else { 0.0 };
            assert!((nb_core(y, mu, r) - direct).abs() < tol, "{} {} {}", y, mu, r);
        }
        // log dnbinom sums to ~1 over support
        let s: f64 = (0..2000).map(|y| dnbinom_log(y as f64, 12.0, 3.0).exp()).sum();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn dll_term_matches_finite_difference() {
        for &(y, mu, r) in &[(4.0, 6.0, 5.0), (40.0, 25.0, 60.0), (2.0, 3.0, 3e3)] {
            let alpha: f64 = 1.0 / r;
            let h = 1e-6;
            let f = |a: f64| nb_core(y, mu, 1.0 / a);
            let fd = (f(alpha * (1.0 + h)) - f(alpha * (1.0 - h))) / (2.0 * alpha * h);
            assert!((dll_term(y, mu, r) - fd).abs() < 1e-4 * (1.0 + fd.abs()), "{} vs {}", dll_term(y, mu, r), fd);
        }
    }

    #[test]
    fn trigamma_values() {
        assert!((trigamma(1.0) - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-12);
        assert!((trigamma(0.5) - std::f64::consts::PI.powi(2) / 2.0).abs() < 1e-12);
        assert!((trigamma(10.0) - 0.10516633568168572).abs() < 1e-12);
    }

    #[test]
    fn size_factors_ratio() {
        // gene rows, 3 samples; sample 2 is sample 1 scaled by 2
        let c = vec![10.0, 20.0, 5.0, 4.0, 8.0, 3.0, 100.0, 200.0, 40.0, 0.0, 1.0, 2.0];
        let sf = size_factors(&c, 4, 3, false).unwrap();
        assert!((sf[1] / sf[0] - 2.0).abs() < 1e-12);
    }
}
