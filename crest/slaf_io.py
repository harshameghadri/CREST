"""
SLAF I/O Bridge for CREST.

Reads SLAF (Sparse Lazy Array Format) datasets directly into the COO format
expected by CREST's Rust SVD/UMAP/Leiden plugins. Writes embeddings and
cluster results back to SLAF.

Requires: pip install slafdb
"""

import polars as pl
import numpy as np
from typing import Optional, Tuple


def read_slaf_expression(slaf_path: str) -> Tuple[pl.DataFrame, int, int]:
    """
    Read a SLAF dataset's expression table into CREST SVD-ready format.

    SLAF stores expression data as COO triplets (cell_integer_id, gene_integer_id, value)
    in a Lance columnar table. This function reads that data and aggregates it into
    the List(UInt32/Float32) format expected by CREST's Rust SVD plugin.

    Args:
        slaf_path: Path to SLAF dataset directory (local, s3://, or hf://).

    Returns:
        Tuple of (aggregated_df, n_cells, n_genes) where:
        - aggregated_df: Polars DataFrame with columns [cell_id, gene_id, count]
          each as List type, grouped into a single row for the SVD plugin.
        - n_cells: Total number of cells in the dataset.
        - n_genes: Total number of genes in the dataset.

    Example:
        >>> from crest.slaf_io import read_slaf_expression
        >>> df, n_cells, n_genes = read_slaf_expression("pbmc3k.slaf")
        >>> print(f"{n_cells} cells x {n_genes} genes")
    """
    try:
        from slaf import SLAFArray
    except ImportError:
        raise ImportError(
            "slafdb is required for SLAF integration. "
            "Install with: pip install slafdb"
        )

    slaf = SLAFArray(slaf_path)
    n_cells, n_genes = slaf.shape

    # Query the expression table — already in COO format
    # SLAF uses integer IDs for efficiency
    expr_df = slaf.query("""
        SELECT cell_integer_id, gene_integer_id, CAST(value AS FLOAT) as count
        FROM expression
        ORDER BY cell_integer_id, gene_integer_id
    """)

    # Aggregate into List columns for crest plugin input
    # The SVD plugin expects a single row with all COO data as lists
    agg_df = expr_df.select([
        pl.col("cell_integer_id").cast(pl.UInt32).alias("cell_id"),
        pl.col("gene_integer_id").cast(pl.UInt32).alias("gene_id"),
        pl.col("count").cast(pl.Float32)
    ]).group_by(pl.lit(1).alias("dataset_id")).agg([
        pl.col("cell_id"),
        pl.col("gene_id"),
        pl.col("count")
    ])

    return agg_df, n_cells, n_genes


def read_slaf_metadata(slaf_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Read cell and gene metadata from a SLAF dataset.

    Args:
        slaf_path: Path to SLAF dataset directory.

    Returns:
        Tuple of (obs, var) Polars DataFrames.
    """
    try:
        from slaf import SLAFArray
    except ImportError:
        raise ImportError(
            "slafdb is required for SLAF integration. "
            "Install with: pip install slafdb"
        )

    slaf = SLAFArray(slaf_path)
    slaf.wait_for_metadata()
    return slaf.obs, slaf.var


def run_svd_umap(
    slaf_path: str,
    n_comps: int = 50,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    n_epochs: int = 200,
) -> np.ndarray:
    """
    Run the full SVD → UMAP pipeline on a SLAF dataset.

    Args:
        slaf_path: Path to SLAF dataset directory.
        n_comps: Number of SVD components (PCs) to compute.
        n_components: UMAP output dimensions (usually 2).
        n_neighbors: UMAP n_neighbors parameter.
        min_dist: UMAP min_dist parameter.
        n_epochs: UMAP optimization epochs.

    Returns:
        np.ndarray of shape (n_cells, n_components) containing UMAP coordinates.

    Example:
        >>> from crest.slaf_io import run_svd_umap
        >>> umap_coords = run_svd_umap("pbmc3k.slaf", n_comps=50)
        >>> print(umap_coords.shape)  # (2700, 2)
    """
    import crest  # noqa: F401 — registers .bio namespace

    agg_df, n_cells, n_genes = read_slaf_expression(slaf_path)

    # SVD step
    svd_df = agg_df.with_columns(
        pl.col("cell_id").bio.svd(
            pl.col("gene_id"), pl.col("count"),
            n_cells=n_cells, n_genes=n_genes, n_comps=n_comps
        ).alias("pca_coords")
    )

    # UMAP step
    result_df = svd_df.with_columns(
        pl.col("pca_coords").bio.umap(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_epochs=n_epochs
        ).alias("umap_coords")
    )

    # Extract UMAP coordinates as numpy array
    if len(result_df) == 0 or result_df["umap_coords"].is_null().all():
        raise RuntimeError("SVD/UMAP pipeline returned empty results")
    umap_nested = result_df["umap_coords"][0]
    umap_array = np.array(umap_nested.to_list())

    return umap_array


def run_svd_leiden(
    slaf_path: str,
    n_comps: int = 50,
    n_neighbors: int = 15,
    resolution: float = 1.0,
) -> np.ndarray:
    """
    Run the full SVD → Leiden clustering pipeline on a SLAF dataset.

    Args:
        slaf_path: Path to SLAF dataset directory.
        n_comps: Number of SVD components (PCs) to compute.
        n_neighbors: Number of neighbors for KNN graph.

    Returns:
        np.ndarray of shape (n_cells,) containing cluster IDs.
    """
    import crest  # noqa: F401

    agg_df, n_cells, n_genes = read_slaf_expression(slaf_path)

    svd_df = agg_df.with_columns(
        pl.col("cell_id").bio.svd(
            pl.col("gene_id"), pl.col("count"),
            n_cells=n_cells, n_genes=n_genes, n_comps=n_comps
        ).alias("pca_coords")
    )

    result_df = svd_df.with_columns(
        pl.col("pca_coords").bio.leiden(n_neighbors=n_neighbors, resolution=resolution).alias("cluster_ids")
    )

    if len(result_df) == 0 or result_df["cluster_ids"].is_null().all():
        raise RuntimeError("SVD/Leiden pipeline returned empty results")
    cluster_nested = result_df["cluster_ids"][0]
    return np.array(cluster_nested.to_list(), dtype=np.uint32)


def write_embeddings_to_slaf(
    slaf_path: str,
    embeddings: np.ndarray,
    column_name: str = "X_umap",
) -> None:
    """
    Write embedding coordinates back to the SLAF cells table.

    Uses Lance's merge operations to add/update a FixedSizeListArray column
    in the cells table, storing embeddings as obsm.

    Args:
        slaf_path: Path to SLAF dataset directory.
        embeddings: np.ndarray of shape (n_cells, n_dims).
        column_name: Name for the embedding column (e.g., "X_umap", "X_pca").
    """
    try:
        import lance
        import pyarrow as pa
    except ImportError:
        raise ImportError("lance and pyarrow are required for write-back.")

    try:
        from slaf import SLAFArray
    except ImportError:
        raise ImportError("slafdb is required. Install with: pip install slafdb")

    slaf = SLAFArray(slaf_path)
    n_cells = slaf.shape[0]

    if embeddings.shape[0] != n_cells:
        raise ValueError(
            f"Embeddings have {embeddings.shape[0]} rows but dataset has {n_cells} cells"
        )

    n_dims = embeddings.shape[1]
    flat_values = embeddings.astype(np.float32).flatten()

    # Create Arrow FixedSizeListArray
    values_array = pa.array(flat_values, type=pa.float32())
    embeddings_array = pa.FixedSizeListArray.from_arrays(values_array, list_size=n_dims)

    # Create table with cell_integer_id + embedding column
    cell_ids = pa.array(np.arange(n_cells, dtype=np.int64))
    new_table = pa.table({
        "cell_integer_id": cell_ids,
        column_name: embeddings_array,
    })

    # Merge into cells Lance dataset
    cells_path = slaf._join_path(slaf.slaf_path, slaf.config["tables"]["cells"])
    ds = lance.dataset(cells_path)
    ds.merge_insert("cell_integer_id").when_matched_update_all().execute(new_table)

    print(f"✅ Wrote {column_name} ({n_cells} × {n_dims}) to {slaf_path}")
