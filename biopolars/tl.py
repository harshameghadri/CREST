import polars as pl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
import warnings

def sparse_masked_pca(
    df: pl.DataFrame,
    n_cells: int,
    n_genes: int,
    n_comps: int = 50,
    random_state: int = 42
) -> dict:
    """
    BioPolars Native Sparse Masked PCA.
    
    Pain Point: Standard PCA dense-centers sparse matrices. On 1M cells x 20k genes, 
    mean centering creates an 80GB dense matrix, crashing the system.
    
    Solution: We use SciPy's LinearOperator to implicitly calculate A_centered @ v
    without ever materializing the centered matrix in memory.
    """
    
    # Extract zero-copy arrays to build the COO matrix pointers
    cell_ids = df["cell_id"].to_numpy(zero_copy_only=True)
    gene_ids = df["gene_id"].to_numpy(zero_copy_only=True)
    counts = df["count"].to_numpy(zero_copy_only=True)
    
    # 1. Build the raw uncentered CSR matrix
    A = sp.coo_matrix((counts, (cell_ids, gene_ids)), shape=(n_cells, n_genes)).tocsr()
    
    # 2. Calculate column means for implicit centering
    # A.sum(0) is a 1 x n_genes matrix. np.A1 flattens it to a 1D array.
    col_means = A.sum(0).A1 / n_cells
    
    def matvec(v):
        # A_centered @ v = (A - 1 * mu) @ v = A @ v - 1 * (mu @ v)
        # where mu @ v is a scalar dot product of the column means and the vector v.
        Av = A.dot(v)
        # mu_dot_v expands to a scalar for 1D v, or a row vector for 2D v
        mu_dot_v = col_means.dot(v)
        # Subtract the broadcasting dot product from the uncentered result
        return Av - mu_dot_v
        
    def rmatvec(u):
        # A_centered.T @ u = (A.T - mu.T * 1.T) @ u = A.T @ u - mu.T * (1.T @ u)
        # 1.T @ u is the sum of the elements in u.
        At_u = A.T.dot(u)
        sum_u = u.sum(axis=0)
        return At_u - np.outer(col_means, sum_u) if u.ndim > 1 else At_u - col_means * sum_u

    # 3. Create the LinearOperator intercepting the SVD dot products
    A_centered_op = LinearOperator(
        shape=(n_cells, n_genes),
        matvec=matvec,
        rmatvec=rmatvec,
        dtype=np.float32
    )
    
    # 4. Compute SVD using ARPACK explicitly bypassing densification
    np.random.seed(random_state)
    v0 = np.random.rand(min(A.shape))
    
    # k must be smaller than the minimum dimension of the matrix
    k = min(n_comps, n_genes - 1, n_cells - 1)
    
    # Catch ARPACK warnings about slow convergence
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        U, S, Vt = svds(A_centered_op, k=k, v0=v0)
    
    # ARPACK returns SVD in increasing order of singular values, we flip them
    U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]
    
    # The PCA coordinates are U * S
    X_pca = U * S
    
    return {
        "X_pca": X_pca,
        "variance": S ** 2 / (n_cells - 1),
        "components": Vt
    }
