"""Run scMultiNet over several random seeds and report mean/std of NMI and ARI.

Data preprocessing (geneSelection / normalize) is seed-independent, so the dataset
is built once and reused across seeds; only the seed, the network weights and the
training loop are re-initialised per run.

Example
-------
    python run_multi_seed.py --dataset BMNC --seeds 0 1 2 3 4 \
        --out_dir ./output/multiseed_BMNC
"""
import os

# Must be set before torch/CUDA initialise, otherwise it has no effect.
os.environ.setdefault('PYTHONHASHSEED', '0')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import csv
import json
import time
import torch
import random
import argparse
import warnings
import numpy as np

from MultiNet.trainer import Trainer
from MultiNet.dataloader import SingleCellMultiOmicsData
from MultiNet.multinet import MultiNet
from MultiNet.configs import *

warnings.filterwarnings("ignore")


def build_parser():
    parser = argparse.ArgumentParser(description='multi-seed run of scMultiNet')
    # --- multi-seed specific ---
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2, 3, 4],
                        help='List of training seeds, e.g. --seeds 0 1 2 3 4')
    parser.add_argument('--kmeans_seed', default=None, type=int,
                        help='random_state of the evaluation KMeans. '
                             'Omit to make it follow the training seed of each run.')
    parser.add_argument('--out_dir', default='./output/multiseed', type=str,
                        help='Directory for per-seed metrics and the summary')
    parser.add_argument('--strict_determinism', action='store_true', default=False,
                        help='Also enable torch.use_deterministic_algorithms(True)')
    parser.add_argument('--save_latent', action='store_true', default=False,
                        help='Save latent + KMeans prediction of every seed')
    parser.add_argument('--export_npz', action='store_true', default=False,
                        help='Save <out_dir>/seed_<s>/<npz_method>_<npz_dataset>.npz of every seed (UMAP comparison)')
    parser.add_argument('--npz_method', default='scMultiNet', type=str)
    parser.add_argument('--npz_dataset', default=None, type=str, help='Dataset tag in the file name (default: --dataset)')
    parser.add_argument('--npz_rna_h5ad', default=None, type=str,
                        help='Take cell_ids/y_true from obs_names/obs[--npz_label_col] of this RNA .h5ad '
                             'instead of Barcodes/Y of the .h5')
    parser.add_argument('--npz_label_col', default='Group', type=str)

    # --- same knobs as train.py ---
    parser.add_argument('--dataset', default='BMNC', help='Name of Dataset')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--pretrain_epochs', default=100, type=int)
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--lr_ae', default=0.0005, type=float)
    parser.add_argument('--lr_d', default=0.003, type=float)
    parser.add_argument('--f1', default=2000, type=int)
    parser.add_argument('--f2', default=2000, type=int)
    parser.add_argument('--filter1', action='store_true', default=True)
    parser.add_argument('--filter2', action='store_true', default=True)
    parser.add_argument('--cutoff', default=0.5, type=float)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--phi_1', default=0.005, type=float)
    parser.add_argument('--phi_2', default=0.005, type=float)
    parser.add_argument('--sigma', default=1.0, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--zeta_1', default=0.005, type=float)
    parser.add_argument('--zeta_2', default=0.003, type=float)
    parser.add_argument('--noise_sigma1', default=2.5, type=float)
    parser.add_argument('--noise_sigma2', default=2.5, type=float)
    parser.add_argument('--data_path', default=None, type=str)
    parser.add_argument('--mode', default='integrate', type=str)
    parser.add_argument('--activation1', default='relu', type=str)
    parser.add_argument('--activation2', default='leakyrelu', type=str)
    parser.add_argument('-el1', '--encodeLayer1', nargs='+', default=[256, 64])
    parser.add_argument('-el2', '--encodeLayer2', nargs='+', default=[256, 64])
    parser.add_argument('--hidden_dim', default=32, type=int)
    parser.add_argument('-dl1', '--decodeLayer1', nargs='+', default=[64, 256])
    parser.add_argument('-dl2', '--decodeLayer2', nargs='+', default=[64, 256])
    parser.add_argument('-disl1', '--discriminateLayer1', nargs='+', default=[256, 64, 32])
    parser.add_argument('-disl2', '--discriminateLayer2', nargs='+', default=[256, 64, 32])
    return parser


def setup_seed(seed, strict=False):
    """Same as train.py:setup_seed, with an optional stricter mode."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if strict:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:  # not available on older torch, or op has no det. kernel
            print('[warn] use_deterministic_algorithms unavailable: %s' % e)


def resolve_configs(args):
    if args.dataset in citeseq_list:
        print('Use citeseq_config')
        return citeseq_config
    if args.dataset in smageseq_list:
        print('Use smageseq_config')
        return smageseq_config
    print('Use yourseq_config')
    return makeconfigs(args)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std(ddof=1)) if arr.size > 1 else 0.0,  # sample std
        'min': float(arr.min()),
        'max': float(arr.max()),
        'n': int(arr.size),
    }


def main():
    args = build_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    configs = resolve_configs(args)
    configs['use_indicator'] = True if args.mode == 'imputation' else False
    print('device: %s | use indicator: %s' % (device, configs['use_indicator']))
    print(configs[f'{args.dataset}_Params'])

    # Preprocessing is deterministic and seed-independent -> build the dataset once.
    setup_seed(args.seeds[0], args.strict_determinism)
    data = SingleCellMultiOmicsData(args.dataset, configs, args.data_path)

    rows = []
    for run_id, seed in enumerate(args.seeds, start=1):
        kmeans_seed = seed if args.kmeans_seed is None else args.kmeans_seed
        print('\n' + '=' * 70)
        print('[%d/%d] dataset=%s  seed=%d  kmeans_seed=%d'
              % (run_id, len(args.seeds), args.dataset, seed, kmeans_seed))
        print('=' * 70)

        # Re-seed *before* building the network so weight init is part of the seed.
        setup_seed(seed, args.strict_determinism)
        multinet = MultiNet(data.n_vars_list, configs).to(device)
        trainer = Trainer(data, multinet, device)

        t0 = time.time()
        trainer.train(configs)
        elapsed = time.time() - t0

        nmi, ari, z, y_pred = trainer.test_cluster(kmeans_seed=kmeans_seed, return_outputs=True)
        rows.append({'dataset': args.dataset, 'seed': seed, 'kmeans_seed': kmeans_seed,
                     'NMI': nmi, 'ARI': ari, 'train_seconds': round(elapsed, 1)})

        seed_dir = os.path.join(args.out_dir, 'seed_%d' % seed)
        if args.save_latent:
            os.makedirs(seed_dir, exist_ok=True)
            trainer.save_latent(save_path=seed_dir)
        if args.export_npz:
            npz_path = os.path.join(seed_dir, '%s_%s.npz' % (args.npz_method, args.npz_dataset or args.dataset))
            trainer.export_npz(npz_path, z, y_pred, method=args.npz_method,
                               rna_h5ad=args.npz_rna_h5ad, label_col=args.npz_label_col,
                               dataset=args.dataset, seed=seed, kmeans_seed=kmeans_seed)

        del trainer, multinet
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ---------------- per-seed csv ----------------
    per_seed_csv = os.path.join(args.out_dir, '%s_per_seed.csv' % args.dataset)
    with open(per_seed_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, 'NMI': '%.6f' % r['NMI'], 'ARI': '%.6f' % r['ARI']})

    # ---------------- summary ----------------
    stats = {'NMI': summarize([r['NMI'] for r in rows]),
             'ARI': summarize([r['ARI'] for r in rows])}

    summary_csv = os.path.join(args.out_dir, '%s_summary.csv' % args.dataset)
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['dataset', 'metric', 'mean', 'std', 'min', 'max', 'n_runs', 'seeds'])
        for metric in ('NMI', 'ARI'):
            s = stats[metric]
            writer.writerow([args.dataset, metric,
                             '%.6f' % s['mean'], '%.6f' % s['std'],
                             '%.6f' % s['min'], '%.6f' % s['max'],
                             s['n'], ' '.join(map(str, args.seeds))])

    with open(os.path.join(args.out_dir, '%s_summary.json' % args.dataset), 'w') as f:
        json.dump({'dataset': args.dataset, 'seeds': args.seeds,
                   'kmeans_seed': args.kmeans_seed, 'runs': rows, 'summary': stats},
                  f, indent=2)

    # ---------------- report ----------------
    print('\n' + '=' * 70)
    print('RESULT  dataset=%s  seeds=%s' % (args.dataset, args.seeds))
    print('=' * 70)
    print('%-6s %-10s %-10s %-12s' % ('seed', 'NMI', 'ARI', 'time(s)'))
    for r in rows:
        print('%-6d %-10.4f %-10.4f %-12.1f' % (r['seed'], r['NMI'], r['ARI'], r['train_seconds']))
    print('-' * 70)
    for metric in ('NMI', 'ARI'):
        s = stats[metric]
        print('%-4s mean=%.4f  std=%.4f  (min=%.4f, max=%.4f, n=%d)'
              % (metric, s['mean'], s['std'], s['min'], s['max'], s['n']))
    print('-' * 70)
    print('NMI = %.4f ± %.4f' % (stats['NMI']['mean'], stats['NMI']['std']))
    print('ARI = %.4f ± %.4f' % (stats['ARI']['mean'], stats['ARI']['std']))
    print('\nsaved -> %s' % per_seed_csv)
    print('saved -> %s' % summary_csv)


if __name__ == '__main__':
    main()
