import argparse
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    """Read the aggregated CSV into {lambda_t: {lambda_e: (acc, ent)}}."""
    data = defaultdict(dict)
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            lt = float(row['lambda_t'])
            le = float(row['lambda_e'])
            data[lt][le] = (
                float(row['aug_test_acc_mean']),
                float(row['ent_avg_aug_mean']),
            )
    return data


def to_grid(data):
    lts = sorted(data.keys())
    les = sorted({le for d in data.values() for le in d})
    acc = np.full((len(lts), len(les)), np.nan)
    ent = np.full((len(lts), len(les)), np.nan)
    for i, lt in enumerate(lts):
        for j, le in enumerate(les):
            if le in data[lt]:
                acc[i, j], ent[i, j] = data[lt][le]
    return lts, les, acc, ent


def plot(lts, les, acc, ent, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, grid, title, cmap in (
        (axes[0], acc, 'Aug test accuracy', 'viridis'),
        (axes[1], ent, 'Entanglement', 'magma'),
    ):
        im = ax.imshow(grid, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(range(len(les)))
        ax.set_xticklabels(les, rotation=45, ha='right')
        ax.set_yticks(range(len(lts)))
        ax.set_yticklabels(lts)
        ax.set_xlabel('lambda_e')
        ax.set_ylabel('lambda_t')
        ax.set_title(title)
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center',
                            color='w', fontsize=8)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    parser.add_argument('--out', default='heatmaps.png')
    args = parser.parse_args()

    data = load_csv(args.csv_path)
    lts, les, acc, ent = to_grid(data)
    plot(lts, les, acc, ent, args.out)
