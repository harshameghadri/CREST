import time
import os
import psutil
import polars as pl
import numpy as np
import statsmodels.api as sm
from biopolars import *

print("==================================================")
print("  BIOPOLARS VS STATSMODELS: NATIVE DESEQ2 GLM     ")
print("==================================================")

num_genes = 1000
num_cells = 500

print(f"Generating {num_genes} genes across {num_cells} cells...")
np.random.seed(42)

# Generate true Negative Binomial counts with intercept and one covariate
true_beta_0 = np.random.normal(2.0, 0.5, size=num_genes) # Intercepts
true_beta_1 = np.random.normal(0.0, 1.0, size=num_genes) # Condition effects
dispersion = 0.1 # Alpha

# Design matrix: col 0 is intercept (all 1s), col 1 is condition (0 or 1)
condition = np.random.randint(0, 2, size=num_cells)
X = np.column_stack([np.ones(num_cells), condition])

# Generate counts
counts = np.zeros((num_cells, num_genes), dtype=np.float32)
for g in range(num_genes):
    eta = X @ np.array([true_beta_0[g], true_beta_1[g]])
    mu = np.exp(eta)
    
    # Negative Binomial parameterization for numpy: (n, p)
    # Variance V = mu + alpha * mu^2 = n * (1-p) / p^2
    # Mean mu = n * (1-p) / p
    # Therefore: p = mu / V = mu / (mu + alpha * mu^2) = 1 / (1 + alpha * mu)
    # n = mu * p / (1-p) = 1 / alpha
    
    n_param = 1.0 / dispersion
    p_param = 1.0 / (1.0 + dispersion * mu)
    
    counts[:, g] = np.random.negative_binomial(n=n_param, p=p_param)

print("\n--- PHASE 1: StatsModels Baseline (GLM / IRLS) ---")
# To evaluate DESeq2, R handles dispersion estimation and NB GLMs natively.
# In Python, statsmodels GLM is the equivalent.

t0 = time.time()
sm_betas = []
for g in range(num_genes):
    y = counts[:, g]
    
    # Check for zeroes
    if np.sum(y) == 0:
        sm_betas.append([0.0, 0.0])
        continue
        
    try:
        model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=dispersion))
        res = model.fit(method='irls', maxiter=100, tol=1e-6)
        sm_betas.append(res.params)
    except Exception as e:
        sm_betas.append([0.0, 0.0])

t1 = time.time()
sm_time = t1 - t0
print(f"StatsModels Execution (For Loop): {sm_time:.2f} seconds")

print("\n--- PHASE 2: BioPolars Native Parallel Formulation ---")
# Prepare data for BioPolars: Gene-wise format
df = pl.DataFrame({
    "gene_id": np.arange(num_genes),
    "counts": counts.T.tolist(),
    "size_factors": [np.ones(num_cells).tolist()] * num_genes,
    "design_matrix": [X.flatten().tolist()] * num_genes,
    "num_covariates": [2] * num_genes, # Intercept + Condition
    "dispersion": [dispersion] * num_genes
})

t0 = time.time()

# Rust-native parallel DESeq2 
res_df = df.with_columns(
    pl.col("counts").bio.deseq2(
        size_factors=pl.col("size_factors"),
        design_matrix=pl.col("design_matrix"),
        num_covariates=pl.col("num_covariates"),
        dispersion=pl.col("dispersion")
    ).alias("fitted_betas")
)

# Force computation
bp_result = res_df.collect() if isinstance(res_df, pl.LazyFrame) else res_df

t1 = time.time()
bp_time = t1 - t0
print(f"BioPolars Execution: {bp_time:.2f} seconds")

print("\n==================================================")
print("               SCIENTIFIC VERIFICATION            ")
print("==================================================")
# Compare the betas
exact_matches = 0
bp_betas = bp_result["fitted_betas"].to_list()

for g in range(num_genes):
    sm_b = sm_betas[g]
    bp_b = bp_betas[g]
    
    if bp_b is None:
        if sum(sm_b) != 0.0:
            print(f"Gene {g}: SM={sm_b}, BP=None")
        continue
        
    # Check numerical closeness (atol=1e-3 because of floating point C vs Rust differences in inversion)
    if np.allclose(sm_b, bp_b, atol=1e-3, rtol=1e-3):
        exact_matches += 1
    elif sum(sm_b) != 0.0:
        print(f"Gene {g} Mismatch: SM={np.array(sm_b).round(4)}, BP={np.array(bp_b).round(4)}")

print(f"Exact Matches (StatsModels ≈ BioPolars): {exact_matches}/{num_genes}")

if bp_time > 0:
    print(f"\n🚀 Time Speedup: {sm_time / bp_time:.2f}x faster globally!")
