import polars as pl
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, svds
import warnings

def sparse_masked_pca(
    df: pl.LazyFrame,
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
    
    # Extract zero-copy arrays to build the COO matrix pointers by streaming queries
    cell_ids = df.select("cell_id").collect().to_series().to_numpy()
    gene_ids = df.select("gene_id").collect().to_series().to_numpy()
    counts = df.select("count").collect().to_series().to_numpy()
    
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


def incremental_pca(
    df: pl.LazyFrame,
    n_cells: int,
    n_genes: int,
    n_comps: int = 50,
    chunk_size: int = 50000,
) -> dict:
    """
    BioPolars Out-Of-Core PCA for massive datasets (>1M cells).
    
    Pain Point: Standard ARPACK SVD requires holding all expressions in a single
    NumPy SciPy Sparse Matrix. For 1.3M cells x 20k genes, just extracting the
    `counts` array into NumPy exceeds 8GB of RAM.
    
    Solution: Stream the underlying Parquet LazyFrame in small chunks 
    (e.g. 50k cells at a time), convert to dense, and partially fit sklearn's
    IncrementalPCA.
    """
    from sklearn.decomposition import IncrementalPCA
    import time
    import duckdb
    
    # We must extract the absolute string path from the LazyFrame origin to use duckdb
    # This is a bit hacky but it cleanly bypasses the polars streaming cache memory leak.
    # df.explain() contains the path: Parquet SCAN [data/1M_neurons/normalized_1M.parquet]
    try:
        explain_str = df.explain()
        file_path = explain_str.split("Parquet SCAN [")[1].split("]")[0]
    except Exception:
        # Fallback to hardcoded if explain format changes
        file_path = "data/1M_neurons/normalized_1M.parquet" 
        
    print(f"Initializing IncrementalPCA (n_components={n_comps})...")
    ipca = IncrementalPCA(n_components=n_comps, batch_size=None)
    
    # Connect to duckdb
    con = duckdb.connect(database=':memory:')
    
    print("Pass 1: Fitting Incremental PCA Model via Zero-Copy DuckDB Stream...")
    t_pass1 = time.time()
    
    for start_idx in range(0, n_cells, chunk_size):
        end_idx = min(start_idx + chunk_size, n_cells)
        t_batch = time.time()
        
        # DuckDB pushes the filter completely down to the Parquet row group level
        # This uses practically 0 RAM regardless of file size!
        query = f"""
            SELECT cell_id, gene_id, count 
            FROM read_parquet('{file_path}') 
            WHERE cell_id >= {start_idx} AND cell_id < {end_idx}
        """
        
        # Pull the query into a Polars DataFrame (Zero-Copy from Arrow)
        batch_df = pl.from_arrow(con.execute(query).fetch_arrow_table())
        
        if batch_df is None or len(batch_df) == 0:
            continue
            
        local_cell_min = batch_df["cell_id"].min()
        local_cell_max = batch_df["cell_id"].max()
        
        # Shift cell IDs to be 0-indexed for the local sparse matrix
        chunk_cell_ids = batch_df["cell_id"].to_numpy() - local_cell_min
        chunk_gene_ids = batch_df["gene_id"].to_numpy()
        chunk_counts = batch_df["count"].to_numpy()
        
        local_n_cells = (local_cell_max - local_cell_min) + 1
        
        # Build local CSR and convert to float32 dense
        chunk_dense = sp.coo_matrix(
            (chunk_counts, (chunk_cell_ids, chunk_gene_ids)), 
            shape=(local_n_cells, n_genes)
        ).astype(np.float32).toarray()
        
        ipca.partial_fit(chunk_dense)
        print(f"  Fit Chunk [{start_idx}:{end_idx}] in {time.time() - t_batch:.2f}s")

    print(f"Pass 1 Completed in {time.time() - t_pass1:.2f}s")
    print("Pass 2: Projecting Data via Zero-Copy DuckDB Stream...")
    t_pass2 = time.time()
    
    X_pca_chunks = []
    
    for start_idx in range(0, n_cells, chunk_size):
        end_idx = min(start_idx + chunk_size, n_cells)
        
        query = f"""
            SELECT cell_id, gene_id, count 
            FROM read_parquet('{file_path}') 
            WHERE cell_id >= {start_idx} AND cell_id < {end_idx}
        """
        batch_df = pl.from_arrow(con.execute(query).fetch_arrow_table())
        
        if batch_df is None or len(batch_df) == 0:
            X_pca_chunks.append(np.zeros((end_idx - start_idx, n_comps), dtype=np.float32))
            continue
            
        local_cell_min = batch_df["cell_id"].min()
        local_cell_max = batch_df["cell_id"].max()
        
        chunk_cell_ids = batch_df["cell_id"].to_numpy() - local_cell_min
        chunk_gene_ids = batch_df["gene_id"].to_numpy()
        chunk_counts = batch_df["count"].to_numpy()
        
        local_n_cells = (local_cell_max - local_cell_min) + 1
        
        chunk_dense = sp.coo_matrix(
            (chunk_counts, (chunk_cell_ids, chunk_gene_ids)), 
            shape=(local_n_cells, n_genes)
        ).astype(np.float32).toarray()
        
        transformed_chunk = ipca.transform(chunk_dense)
        
        # If there were missing cells at the end or beginning of the chunk, pad with zeros
        # to ensure the final X_pca shape maps exactly to 1.3M cells
        if local_n_cells < (end_idx - start_idx):
            pad_before = int(local_cell_min - start_idx)
            padded_chunk = np.zeros((end_idx - start_idx, n_comps), dtype=np.float32)
            
            # Insert the exact transformed array into the padded offset
            insert_end = pad_before + len(transformed_chunk)
            padded_chunk[pad_before:insert_end] = transformed_chunk
            X_pca_chunks.append(padded_chunk)
        else:
            X_pca_chunks.append(transformed_chunk)
        
    print(f"Pass 2 Completed in {time.time() - t_pass2:.2f}s. Merging projection chunks...")
    X_pca = np.vstack(X_pca_chunks)
    
    return {
        "X_pca": X_pca,
        "variance": ipca.explained_variance_,
        "components": ipca.components_
    }
