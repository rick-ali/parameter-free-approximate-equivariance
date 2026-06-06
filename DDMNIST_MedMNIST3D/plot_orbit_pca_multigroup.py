"""
Plot the reduced-density PCA produced by orbit_reduced_density_pca_multigroup.py.

Reads the .npz file and draws one panel per subsystem (PC1 vs PC2), driven by the
`subsystem_names` stored in the NPZ:
  bipartite  (D4xD4)        : rho_digit1 (fixed) | rho_digit2 (swept)
  tripartite (C4xC4, D1xD1) : rho_outer (I)      | rho_digit1 (fixed) | rho_digit2 (swept)
Each sample is drawn as its label text (group element for --variation symmetry, digit
class for --variation digit), coloured black if classified correctly and red if not. For
symmetry mode the points are connected in element order to show the orbit trajectory.

Runs locally (needs only numpy + matplotlib; no torch/medmnist). Example:
    python plot_orbit_pca_multigroup.py --npz results/orbit.npz --out results/orbit_pca.png
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

# Symmetry-element labels for the swept-digit sweep, keyed by group.
GROUP_SYM_LABELS = {
    'D4xD4': ['e', 'r', 'r2', 'r3', 's', 'rs', 'r2s', 'r3s'],
    'C4xC4': ['e', 'r', 'r2', 'r3'],
    'D1xD1': ['e', 's'],
}


def marker_panel(ax, coords, codes, point_labels, correct, var, pcs, title, draw_line):
    px, py = pcs
    xs, ys = coords[:, px], coords[:, py]
    if draw_line:
        # faint line through the points in code order to show the orbit path
        order = np.argsort(codes)
        ax.plot(xs[order], ys[order], '-', color='0.8', lw=1, zorder=1)
    for i in range(len(codes)):
        ax.text(xs[i], ys[i], point_labels[i],
                color=('black' if correct[i] else 'red'),
                ha='center', va='center', fontsize=12, fontweight='bold', zorder=2)
    vx = var[px] * 100 if px < len(var) else 0.0
    vy = var[py] * 100 if py < len(var) else 0.0
    ax.set_xlabel(f"PC{px + 1} ({vx:.1f}%)")
    ax.set_ylabel(f"PC{py + 1} ({vy:.1f}%)")
    ax.set_title(title)


def sym_range(coord_arrays, px, py):
    """Symmetric half-range (about 0) covering the given PCA coord arrays, with padding."""
    m = max(float(np.abs(c[:, [px, py]]).max()) for c in coord_arrays)
    return max(m * 1.1, 1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', type=str, required=True,
                        help='Path to the .npz from orbit_reduced_density_pca_multigroup.py')
    parser.add_argument('--out', type=str, default=None, help='Output image path (default: <npz>_pca.png)')
    parser.add_argument('--pcs', type=str, default='1,2', help='Which two PCs to plot, 1-indexed (e.g. "1,2").')
    parser.add_argument('--correctness', type=str, default='swept', choices=['swept', 'full'],
                        help="What makes a marker black: 'swept' = the swept digit's predicted class "
                             "matches its true class; 'full' = the whole 2-digit prediction is correct.")
    parser.add_argument('--independent-axes', action='store_true',
                        help='Let each panel autoscale separately (default: share the larger panel scale).')
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    group = str(data['group'])
    variation = str(data['variation'])
    sweep_digit = int(data['sweep_digit']) if 'sweep_digit' in data.files else 2
    names = [str(s) for s in data['subsystem_names']]
    codes = data['var_codes']
    label = int(data['label'])
    pred_class = data['pred_class'].astype(int)

    coords = {name: data[f'pca_{name}_coords'] for name in names}
    var = {name: data[f'pca_{name}_explained_var'] for name in names}
    rho = {name: data[f'rho_{name}'] for name in names}
    # Total variance across the samples (same number the analysis script prints).
    tvar = {name: float(np.var(rho[name].reshape(rho[name].shape[0], -1), axis=0).sum())
            for name in names}

    # ---- classification correctness per sample ----
    # tens place = digit-1, unit place = digit-2. The swept digit's class varies with
    # `codes` in 'digit' mode; everything else comes from the fixed base label.
    if variation == 'digit':
        if sweep_digit == 2:
            unit_true = codes.astype(int)
            tens_true = np.full(len(codes), label // 10, dtype=int)
        else:
            tens_true = codes.astype(int)
            unit_true = np.full(len(codes), label % 10, dtype=int)
    else:                                         # symmetry: classes are group-invariant
        tens_true = np.full(len(codes), label // 10, dtype=int)
        unit_true = np.full(len(codes), label % 10, dtype=int)
    if args.correctness == 'full':
        correct = pred_class == (tens_true * 10 + unit_true)
    elif sweep_digit == 2:                        # 'swept' = the swept digit's class
        correct = (pred_class % 10) == unit_true
    else:
        correct = (pred_class // 10) == tens_true

    # ---- mode-dependent presentation ----
    if variation == 'symmetry':
        sym_labels = GROUP_SYM_LABELS[group]
        point_labels = [sym_labels[int(c)] for c in codes]
        swept_word = 'rotated'
        sup_what = f"digit-{sweep_digit}'s {group.split('x')[0]} orbit"
        draw_line = True
    else:  # 'digit'
        point_labels = [str(int(c)) for c in codes]
        swept_word = 'swept over classes'
        sup_what = f"digit-{sweep_digit}'s 10 classes"
        draw_line = False

    def role_for(name):
        if name == 'outer':
            return 'I, multiplicity'
        digit_num = 1 if name == 'digit1' else 2
        return f'digit-{digit_num}, ' + (swept_word if digit_num == sweep_digit else 'fixed')

    px, py = [int(p) - 1 for p in args.pcs.split(',')]

    n_panels = len(names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5))
    if n_panels == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        title = f"rho_{name}  ({role_for(name)})\ntotal variance = {tvar[name]:.3e}"
        marker_panel(ax, coords[name], codes, point_labels, correct, var[name],
                     (px, py), title, draw_line)

    # Set axis limits explicitly (text markers don't drive autoscaling).
    # PCA coords are mean-centered, so use symmetric limits about 0.
    if args.independent_axes:
        for ax, name in zip(axes, names):
            R = sym_range([coords[name]], px, py)
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    else:
        # share the largest panel's scale so the variance asymmetry is visible
        R = sym_range(list(coords.values()), px, py)
        for ax in axes:
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    sup = f"Reduced-density PCA over {sup_what}   [{group}, {str(data['decomposition'])}]"
    sup += f"   (base label {label}: digit-1={label // 10}, digit-2={label % 10})"
    sup += "    [black=correct, red=misclassified]"
    fig.suptitle(sup)

    out = args.out or (args.npz.rsplit('.', 1)[0] + '_pca.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Wrote {out}")
    for name in names:
        print(f"total variance  rho_{name} = {tvar[name]:.3e}")
    print(f"correctly classified: {int(correct.sum())}/{len(correct)}")


if __name__ == "__main__":
    main()
