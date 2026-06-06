"""
Plot the reduced-density PCA produced by orbit_reduced_density_pca.py.

Reads the .npz file and draws two scatter panels (PC1 vs PC2):
  left  : rho_A  (digit-1 marginal, the FIXED digit  -> expect a tight cluster)
  right : rho_B  (digit-2 marginal, the swept digit  -> expect a spread)
Each point is labelled by its variation code (D4 element for --variation symmetry,
digit class for --variation digit). For symmetry mode the points are connected in
element order to show the orbit trajectory.

Runs locally (needs only numpy + matplotlib; no torch/medmnist). Example:
    python plot_orbit_pca.py --npz results/orbit.npz --out results/orbit_pca.png
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

D4_LABELS = ['e', 'r', 'r2', 'r3', 's', 'rs', 'r2s', 'r3s']  # identity, rotations, flip+rotations


def scatter_panel(ax, coords, var, codes, point_labels, pred_unit, pcs, title, draw_line):
    px, py = pcs
    xs, ys = coords[:, px], coords[:, py]
    if draw_line:
        # faint line through the points in code order to show the orbit path
        order = np.argsort(codes)
        ax.plot(xs[order], ys[order], '-', color='0.7', lw=1, zorder=1)
    sc = ax.scatter(xs, ys, c=codes, cmap='viridis', s=120, zorder=2, edgecolor='k', linewidth=0.5)
    for i in range(len(codes)):
        lab = point_labels[i]
        if pred_unit is not None:
            lab += f"→{int(pred_unit[i])}"  # predicted digit-2 class
        ax.annotate(lab, (xs[i], ys[i]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    vx = var[px] * 100 if px < len(var) else 0.0
    vy = var[py] * 100 if py < len(var) else 0.0
    ax.set_xlabel(f"PC{px + 1} ({vx:.1f}%)")
    ax.set_ylabel(f"PC{py + 1} ({vy:.1f}%)")
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    return sc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', type=str, required=True, help='Path to the .npz from orbit_reduced_density_pca.py')
    parser.add_argument('--out', type=str, default=None, help='Output image path (default: <npz>_pca.png)')
    parser.add_argument('--pcs', type=str, default='1,2', help='Which two PCs to plot, 1-indexed (e.g. "1,2").')
    parser.add_argument('--no-pred', action='store_true', help='Do not annotate predicted digit-2 class.')
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    # variation mode + per-point codes, with backward-compat for older npz files (symmetry only).
    variation = str(data['variation']) if 'variation' in data else 'symmetry'
    codes = data['var_codes'] if 'var_codes' in data else data['d4_codes']
    cA, cB = data['pca_A_coords'], data['pca_B_coords']
    vA, vB = data['pca_A_explained_var'], data['pca_B_explained_var']
    pred_unit = None if args.no_pred else data.get('pred_unit')
    label = int(data['label']) if 'label' in data else None

    # Total variance across the samples (same number the analysis script prints).
    # Computed from the reduced-density arrays so it works on older npz files too.
    rho_A, rho_B = data['rho_A'], data['rho_B']
    tvar_A = float(np.var(rho_A.reshape(rho_A.shape[0], -1), axis=0).sum())
    tvar_B = float(np.var(rho_B.reshape(rho_B.shape[0], -1), axis=0).sum())

    # Mode-dependent presentation.
    if variation == 'symmetry':
        point_labels = [D4_LABELS[int(c)] for c in codes]
        swept_title = "digit-2, rotated"
        cbar_label = "D4 element (code)"
        sup_what = "digit-2's D4 orbit"
        draw_line = True
    else:  # 'digit'
        point_labels = [str(int(c)) for c in codes]
        swept_title = "digit-2, swept over classes"
        cbar_label = "digit-2 class"
        sup_what = "digit-2's 10 classes"
        draw_line = False

    px, py = [int(p) - 1 for p in args.pcs.split(',')]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    scatter_panel(axes[0], cA, vA, codes, point_labels, pred_unit, (px, py),
                  f"rho_A  (digit-1, fixed)\ntotal variance = {tvar_A:.3e}", draw_line)
    sc = scatter_panel(axes[1], cB, vB, codes, point_labels, pred_unit, (px, py),
                       f"rho_B  ({swept_title})\ntotal variance = {tvar_B:.3e}", draw_line)
    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(cbar_label)

    sup = f"Reduced-density PCA over {sup_what}"
    if label is not None:
        sup += f"   (base label {label}: digit-1={label // 10}, digit-2={label % 10})"
    fig.suptitle(sup)

    out = args.out or (args.npz.rsplit('.', 1)[0] + '_pca.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Wrote {out}")
    # also report total spread so the fixed-vs-rotated asymmetry is quantified
    print(f"total variance  rho_A (fixed)  = {np.var(cA, axis=0).sum():.3e}")
    print(f"total variance  rho_B (rotated)= {np.var(cB, axis=0).sum():.3e}")


if __name__ == "__main__":
    main()
