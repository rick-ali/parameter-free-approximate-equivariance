import argparse
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    """Read the per-run CSV into {(lambda_t, lambda_e): {'acc': [...], 'ent': [...]}}."""
    runs = defaultdict(lambda: {'acc': [], 'ent': []})
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            key = (float(row['lambda_t']), float(row['lambda_e']))
            runs[key]['acc'].append(float(row['aug_test_acc']))
            runs[key]['ent'].append(float(row['ent_avg_aug']))
    return runs


def to_grid(runs):
    """Aggregate per-run values into mean/std grids over (lambda_t, lambda_e)."""
    lts = sorted({lt for lt, _ in runs})
    les = sorted({le for _, le in runs})
    shape = (len(lts), len(les))
    acc_mean = np.full(shape, np.nan)
    acc_std = np.full(shape, np.nan)
    ent_mean = np.full(shape, np.nan)
    ent_std = np.full(shape, np.nan)
    for i, lt in enumerate(lts):
        for j, le in enumerate(les):
            if (lt, le) in runs:
                accs = np.array(runs[(lt, le)]['acc'])
                ents = np.array(runs[(lt, le)]['ent'])
                acc_mean[i, j], acc_std[i, j] = accs.mean(), accs.std()
                ent_mean[i, j], ent_std[i, j] = ents.mean(), ents.std()
    return lts, les, acc_mean, acc_std, ent_mean, ent_std


def plot(lts, les, acc_mean, acc_std, ent_mean, ent_std, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mean, std, title, cmap in (
        (axes[0], acc_mean, acc_std, 'Aug test accuracy', 'viridis'),
        (axes[1], ent_mean, ent_std, 'Entanglement', 'magma'),
    ):
        im = ax.imshow(mean, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(les)))
        ax.set_xticklabels(les, rotation=45, ha='right')
        ax.set_yticks(range(len(lts)))
        ax.set_yticklabels(lts)
        ax.set_xlabel('lambda_e')
        ax.set_ylabel('lambda_t')
        ax.set_title(title)
        for i in range(mean.shape[0]):
            for j in range(mean.shape[1]):
                if not np.isnan(mean[i, j]):
                    ax.text(j, i, f'{mean[i, j]:.3f}\n±{std[i, j]:.3f}',
                            ha='center', va='center', color='w', fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--out', default='heatmaps.png')
    args = parser.parse_args()

    runs = load_csv(args.csv_path)
    plot(*to_grid(runs), args.out)
