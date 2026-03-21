import h5py
import polars as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Union

def convert_h5_to_parquet_stream(file_path: Union[str, Path], output_path: Union[str, Path], chunk_size: int = 50_000):
    """
    Stream a 10x Genomics HDF5 cell-by-cell into a Parquet Triplet file.
    This maintains a strict low-memory ceiling regardless of file size.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")

    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    with h5py.File(file_path, 'r') as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"HDF5 file has no top-level groups: {file_path}")
        root = keys[0]
        matrix = f[root]

        for required_key in ('indptr', 'data', 'indices'):
            if required_key not in matrix:
                raise ValueError(f"HDF5 matrix group missing required key '{required_key}'")

        # We only need references, no loading into RAM
        indptr = matrix['indptr']
        data = matrix['data']
        indices = matrix['indices']

        num_cells = len(indptr) - 1
        writer = None

        print(f"Streaming {num_cells} cells to {output_path} in chunks of {chunk_size}...")

        try:
            for i in range(0, num_cells, chunk_size):
                end = min(i + chunk_size, num_cells)

                ptr_start = indptr[i]
                ptr_end = indptr[end]

                # Read exactly the chunk's footprint from disk
                chunk_data = data[ptr_start:ptr_end]
                chunk_indices = indices[ptr_start:ptr_end]
                local_indptr = indptr[i:end+1] - ptr_start

                cell_ids = np.repeat(np.arange(i, end, dtype=np.uint32), np.diff(local_indptr))

                df = pl.DataFrame({
                    "cell_id": cell_ids,
                    "gene_id": chunk_indices.astype(np.uint32),
                    "count": chunk_data.astype(np.float32)
                })

                table = df.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression='ZSTD')

                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    print("Streaming conversion complete.")
