"""Convert a paired RNA/ATAC AnnData dataset (.h5ad) into the single .h5
file format expected by MultiNet.dataloader.BaseData.read_h5 (see data/readme.md).

Assumes: RNA and ATAC cells are already aligned 1:1 by row order, and
adata.X in both files already holds raw counts.

Writes X1/X2 in row chunks straight into the output .h5 so a full dense
copy of either matrix is never held in RAM at once (important for ATAC
matrices with a very large number of peaks/features).

Usage:
    python preread_dataset.py \
        --rna data/SMAGE-3K/RNA.h5ad \
        --atac data/SMAGE-3K/ATAC.h5ad \
        --out data/SMAGE-3K/SMAGE-3K.h5 \
        --chunk-size 500
"""
import argparse
import gc

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def write_matrix(f, name, X, chunk_size):
    n_obs, n_vars = X.shape
    is_sparse = sparse.issparse(X)
    dset = f.create_dataset(
        name, shape=(n_obs, n_vars), dtype=np.float32,
        compression='gzip', chunks=(min(chunk_size, n_obs), n_vars),
    )
    for start in range(0, n_obs, chunk_size):
        end = min(start + chunk_size, n_obs)
        block = X[start:end]
        block = block.toarray() if is_sparse else np.asarray(block)
        # read_dataset() asserts counts are raw (integer-valued) data.
        assert np.allclose(block, np.round(block)), \
            f'{name} rows {start}:{end} are not raw integer counts'
        dset[start:end] = block.astype(np.float32)
    return dset


def main():
    parser = argparse.ArgumentParser(description='Convert RNA/ATAC h5ad pair into MultiNet .h5 input')
    parser.add_argument('--rna', required=True, help='Path to RNA .h5ad')
    parser.add_argument('--atac', required=True, help='Path to ATAC .h5ad')
    parser.add_argument('--out', required=True, help='Output .h5 path')
    parser.add_argument('--label-col', default='cell_type', help='obs column -> Y / Celltypes')
    parser.add_argument('--chunk-size', type=int, default=500,
                         help='Cells processed per chunk when writing X1/X2 (lower = less RAM)')
    parser.add_argument('--hvg', action='store_true',
                         help="Subset RNA/ATAC to var['highly_variable']==True (if present) before writing. "
                              "scMultiNet's own feature selection only runs after the full raw matrix is "
                              "loaded into RAM, so pre-filtering here is the only way to cut peak RAM.")
    args = parser.parse_args()

    print(f'Loading {args.rna} ...')
    rna = ad.read_h5ad(args.rna)
    print(f'Loading {args.atac} ...')
    atac = ad.read_h5ad(args.atac)

    if args.hvg:
        if 'highly_variable' in rna.var:
            n_before = rna.n_vars
            rna = rna[:, rna.var['highly_variable']].copy()
            print(f'  RNA HVG filter: {n_before} -> {rna.n_vars} genes')
        else:
            print("  Warning: --hvg set but rna.var has no 'highly_variable' column, skipping.")
        if 'highly_variable' in atac.var:
            n_before = atac.n_vars
            atac = atac[:, atac.var['highly_variable']].copy()
            print(f'  ATAC HVG filter: {n_before} -> {atac.n_vars} peaks')
        else:
            print("  Warning: --hvg set but atac.var has no 'highly_variable' column, skipping.")

    assert rna.n_obs == atac.n_obs, f'RNA has {rna.n_obs} cells but ATAC has {atac.n_obs} cells'
    rna_shape, atac_shape = rna.shape, atac.shape

    # Pull out metadata (small arrays) before touching the big count matrices.
    # dtype=object (not numpy's .astype(str), which yields fixed-width '<U'
    # strings that h5py's variable-length string_dtype cannot write).
    Y = rna.obs[args.label_col].astype(str).to_numpy(dtype=object)
    Celltypes = rna.obs[args.label_col].astype(str).to_numpy(dtype=object)
    Genes = rna.var_names.to_numpy().astype(object)
    GenesFromPeaks = atac.var_names.to_numpy().astype(object)
    Barcodes = rna.obs_names.to_numpy().astype(object)
    n_classes = len(np.unique(Y))

    str_dtype = h5py.string_dtype(encoding='utf-8')
    print(f'Writing {args.out} ...')
    with h5py.File(args.out, 'w') as f:
        print(f'  X1: {rna_shape} ({args.chunk_size} cells/chunk)')
        write_matrix(f, 'X1', rna.X, args.chunk_size)
        del rna
        gc.collect()

        print(f'  X2: {atac_shape} ({args.chunk_size} cells/chunk)')
        write_matrix(f, 'X2', atac.X, args.chunk_size)
        del atac
        gc.collect()

        f.create_dataset('Y', data=Y, dtype=str_dtype)
        f.create_dataset('Celltypes', data=Celltypes, dtype=str_dtype)
        f.create_dataset('Genes', data=Genes, dtype=str_dtype)
        f.create_dataset('GenesFromPeaks', data=GenesFromPeaks, dtype=str_dtype)
        f.create_dataset('Barcodes', data=Barcodes, dtype=str_dtype)

    print(f'Done. X1: {rna_shape}, X2: {atac_shape}, {n_classes} classes.')


if __name__ == '__main__':
    main()
