"""Convert a paired RNA/ATAC AnnData dataset (.h5ad) into the single .h5
file format expected by MultiNet.dataloader.BaseData.read_h5 (see data/readme.md).

Assumes: RNA and ATAC cells are already aligned 1:1 by row order, and
adata.X in both files already holds raw counts.

Usage:
    python preread_dataset.py \
        --rna data/SMAGE-3K/RNA.h5ad \
        --atac data/SMAGE-3K/ATAC.h5ad \
        --out data/SMAGE-3K/SMAGE-3K.h5
"""
import argparse

import anndata as ad
import h5py
import numpy as np
from scipy import sparse


def to_dense(x):
    return x.toarray() if sparse.issparse(x) else np.asarray(x)


def main():
    parser = argparse.ArgumentParser(description='Convert RNA/ATAC h5ad pair into MultiNet .h5 input')
    parser.add_argument('--rna', required=True, help='Path to RNA .h5ad')
    parser.add_argument('--atac', required=True, help='Path to ATAC .h5ad')
    parser.add_argument('--out', required=True, help='Output .h5 path')
    parser.add_argument('--label-col', default='cell_type', help='obs column -> Y / Celltypes')
    args = parser.parse_args()

    print(f'Loading {args.rna} ...')
    rna = ad.read_h5ad(args.rna)
    print(f'Loading {args.atac} ...')
    atac = ad.read_h5ad(args.atac)

    assert rna.n_obs == atac.n_obs, f'RNA has {rna.n_obs} cells but ATAC has {atac.n_obs} cells'

    X1 = to_dense(rna.X).astype(np.float32)
    X2 = to_dense(atac.X).astype(np.float32)

    # read_dataset() asserts counts are raw (integer-valued) data.
    assert np.allclose(X1, X1.astype(int)), 'rna.X is not raw integer counts'
    assert np.allclose(X2, X2.astype(int)), 'atac.X is not raw integer counts'

    Y = rna.obs[args.label_col].astype(str).to_numpy()
    Celltypes = rna.obs[args.label_col].astype(str).to_numpy()
    Genes = rna.var_names.to_numpy().astype(str)
    GenesFromPeaks = atac.var_names.to_numpy().astype(str)
    Barcodes = rna.obs_names.to_numpy().astype(str)

    str_dtype = h5py.string_dtype(encoding='utf-8')
    print(f'Writing {args.out} ...')
    with h5py.File(args.out, 'w') as f:
        f.create_dataset('X1', data=X1, compression='gzip')
        f.create_dataset('X2', data=X2, compression='gzip')
        f.create_dataset('Y', data=Y)
        f.create_dataset('Celltypes', data=Celltypes, dtype=str_dtype)
        f.create_dataset('Genes', data=Genes, dtype=str_dtype)
        f.create_dataset('GenesFromPeaks', data=GenesFromPeaks, dtype=str_dtype)
        f.create_dataset('Barcodes', data=Barcodes, dtype=str_dtype)

    print(f'Done. X1: {X1.shape}, X2: {X2.shape}, {len(np.unique(Y))} classes.')


if __name__ == '__main__':
    main()
