//! Utility functions for UMAP

/// Fits the UMAP parameters a and b to the curve 1 / (1 + a * x^(2b))
/// interpolating the spread and min_dist parameters.
pub fn find_ab_params(spread: f32, min_dist: f32) -> (f32, f32) {
    let mut a = 1.0f32;
    let mut b = 1.0f32;
    
    // Create training data along the x-axis
    let n_pts = 300;
    let max_x = spread * 3.0;
    let mut xs = vec![0.0f32; n_pts];
    let mut ys = vec![0.0f32; n_pts];
    
    for i in 0..n_pts {
        let x = (i as f32 / (n_pts - 1) as f32) * max_x;
        xs[i] = x;
        if x < min_dist {
            ys[i] = 1.0;
        } else {
            ys[i] = (-(x - min_dist) / spread).exp();
        }
    }
    
    // Simple Gradient Descent / Adam optimizer to find a and b
    let mut m_a = 0.0;
    let mut v_a = 0.0;
    let mut m_b = 0.0;
    let mut v_b = 0.0;
    
    let lr = 0.05;
    let beta1 = 0.9;
    let beta2 = 0.999;
    let eps = 1e-8;
    
    for epoch in 1..=5000 {
        let mut grad_a = 0.0;
        let mut grad_b = 0.0;
        
        for i in 0..n_pts {
            let x = xs[i];
            let y_true = ys[i];
            
            // Avoid x=0 for b gradient (0^0 is tricky, though x^2b when x>0 is fine)
            if x <= 0.0 { continue; }
            
            let x_2b = x.powf(2.0 * b);
            let denom = 1.0 + a * x_2b;
            let y_pred = 1.0 / denom;
            
            let diff = y_pred - y_true;
            
            // Gradients of MSE loss w.r.t 'a' and 'b'
            let d_y_pred_d_a = -(x_2b) / (denom * denom);
            let d_y_pred_d_b = -2.0 * a * x_2b * x.ln() / (denom * denom);
            
            grad_a += 2.0 * diff * d_y_pred_d_a;
            grad_b += 2.0 * diff * d_y_pred_d_b;
        }
        
        grad_a /= n_pts as f32;
        grad_b /= n_pts as f32;
        
        // Adam update
        m_a = beta1 * m_a + (1.0 - beta1) * grad_a;
        v_a = beta2 * v_a + (1.0 - beta2) * grad_a * grad_a;
        let m_hat_a = m_a / (1.0 - beta1.powi(epoch));
        let v_hat_a = v_a / (1.0 - beta2.powi(epoch));
        a -= lr * m_hat_a / (v_hat_a.sqrt() + eps);
        
        m_b = beta1 * m_b + (1.0 - beta1) * grad_b;
        v_b = beta2 * v_b + (1.0 - beta2) * grad_b * grad_b;
        let m_hat_b = m_b / (1.0 - beta1.powi(epoch));
        let v_hat_b = v_b / (1.0 - beta2.powi(epoch));
        b -= lr * m_hat_b / (v_hat_b.sqrt() + eps);
        
        // Ensure a and b stay positive
        if a < 1e-5 { a = 1e-5; }
        if b < 1e-5 { b = 1e-5; }
    }
    
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_ab_params() {
        let (a, b) = find_ab_params(1.0, 0.1);
        assert!(a > 0.5 && a < 2.5);
        assert!(b > 0.5 && b < 1.5);
    }
}
