"""
Plot the all-pairs robustness PCA produced by orbit_reduced_density_pca_robustness.py.

Reads one <variation>_sweep<d>.npz and draws the combined, all-digit-pair figure (plus an
optional quantitative summary). The pooling rule depends on the variation stored in the NPZ:

  digit    : a SINGLE pooled PCA per subsystem over all 100 (fixed-digit x swept-class)
             samples, overlaid in one axes -> 1 row x N-subsystem columns. Colour = fixed
             digit; text label = swept class. Robustness reads as: in the swept-digit panel,
             equal swept-class labels cluster regardless of colour; in the fixed-digit panel,
             equal-colour points cluster.

  symmetry : per fixed digit -> 10 rows (fixed digit 0-9) x N-subsystem columns. Each cell is
             its own pooled PCA over that fixed digit's (10 classes x n_sym transforms)
             samples. Colour = swept class; text = group element; orbit lines join the n_sym
             transforms of each class. The fixed-digit and `outer` columns should stay
             collapsed in every row; the swept-digit column shows class-separated orbits.

Misclassified samples get a red ring behind the label. Runs locally (numpy + matplotlib).

Example:
    python plot_orbit_pca_robustness.py --npz results/C4xC4/robust_digit_sweep2.npz \
        --out results/C4xC4/robust_digit_sweep2.png --summary
"""
import argparse

import numpy as np
import matplotlib.pyplot as plt

from plot_orbit_pca_multigroup import GROUP_SYM_LABELS, sym_range


def pca_across_samples(flat):
    """PCA over rows of `flat` (n_samples, n_features) via SVD of the centered data.

    Local numpy copy of the helper in orbit_reduced_density_pca_multigroup so this plot step
    stays torch-free. Returns (coords (n, k), explained_var_ratio (k,), components (k, feat)).
    """
    centered = flat - flat.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U * S
    total = (S ** 2).sum()
    explained = (S ** 2) / total if total > 0 else np.zeros_like(S)
    return coords, explained, Vt


def correctness(pred, fixed_val, swept_class, sweep_digit, mode):
    """Boolean per-sample correctness. 'swept' = the swept digit's class is right; 'full' = the
    whole 2-digit prediction is right. Digit-1 is the tens place, digit-2 the units place."""
    if sweep_digit == 2:
        tens_true, unit_true = fixed_val, swept_class
    else:
        tens_true, unit_true = swept_class, fixed_val
    if mode == 'full':
        return pred == (tens_true * 10 + unit_true)
    return (pred % 10 == unit_true) if sweep_digit == 2 else (pred // 10 == tens_true)


def role_for(name, sweep_digit, swept_word):
    if name == 'outer':
        return 'I, multiplicity'
    digit_num = 1 if name == 'digit1' else 2
    return f'digit-{digit_num}, ' + (swept_word if digit_num == sweep_digit else 'fixed')


def scatter_panel(ax, coords, pcs, point_labels, color_idx, correct, var, title,
                  orbit_keys=None):
    """Scatter text labels coloured by `color_idx` (tab10); ring misclassified in red.

    orbit_keys: optional per-point group key; points sharing a key are joined by a faint line
    in point order (used to trace each class's symmetry orbit in the symmetry figure).
    """
    px, py = pcs
    xs = coords[:, px] if px < coords.shape[1] else np.zeros(len(coords))
    ys = coords[:, py] if py < coords.shape[1] else np.zeros(len(coords))
    cmap = plt.cm.tab10
    if orbit_keys is not None:
        for k in np.unique(orbit_keys):
            m = np.where(orbit_keys == k)[0]
            ax.plot(xs[m], ys[m], '-', color='0.85', lw=0.8, zorder=1)
    for i in range(len(xs)):
        if not correct[i]:
            ax.scatter([xs[i]], [ys[i]], s=160, facecolors='none', edgecolors='red',
                       linewidths=1.0, zorder=2)
        ax.text(xs[i], ys[i], str(point_labels[i]), color=cmap(int(color_idx[i]) % 10),
                ha='center', va='center', fontsize=9, fontweight='bold', zorder=3)
    vx = var[px] * 100 if px < len(var) else 0.0
    vy = var[py] * 100 if py < len(var) else 0.0
    ax.set_xlabel(f"PC{px + 1} ({vx:.1f}%)")
    ax.set_ylabel(f"PC{py + 1} ({vy:.1f}%)")
    ax.set_title(title, fontsize=10)


def plot_digit(data, names, sweep_digit, pcs, correct, out):
    """Pooled-PCA overlay: 1 row x N-subsystem columns; colour = fixed digit, text = swept."""
    fixed_val = data['fixed_val'].astype(int)
    swept_class = data['swept_class'].astype(int)
    n_panels = len(names)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))
    if n_panels == 1:
        axes = [axes]

    coords_by_name, var_by_name = {}, {}
    for name in names:
        rho = data[f'rho_{name}']
        coords, var, _ = pca_across_samples(rho.reshape(rho.shape[0], -1))
        coords_by_name[name], var_by_name[name] = coords, var

    px, py = pcs
    R = sym_range(list(coords_by_name.values()), px, py)
    print(f"\n[digit, sweep digit-{sweep_digit}] per-subsystem stats (pooled over all "
          f"{len(fixed_val)} samples):")
    for ax, name in zip(axes, names):
        tvar = float(np.var(data[f'rho_{name}'].reshape(len(fixed_val), -1), axis=0).sum())
        align = data[f'align_{name}']
        title = (f"rho_{name}  ({role_for(name, sweep_digit, 'swept over classes')})\n"
                 f"total variance = {tvar:.3e}\n"
                 f"align(ones) = {align.mean():.3f}+/-{align.std():.3f}   "
                 f"dev(ones) = {1.0 - align.mean():.3f}")
        scatter_panel(ax, coords_by_name[name], (px, py), swept_class, fixed_val, correct,
                      var_by_name[name], title)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R)
        ax.set_aspect('equal', adjustable='box')
        print(f"  rho_{name:<7} ({role_for(name, sweep_digit, 'swept over classes')}): "
              f"total var = {tvar:.3e}   mean align(ones) = {align.mean():.4f}   "
              f"mean dev(ones) = {1.0 - align.mean():.4f}")

    fixed_digit = 1 if sweep_digit == 2 else 2
    sup = (f"All-pairs reduced-density PCA (pooled), sweeping digit-{sweep_digit} over its 10 "
           f"classes   [{data['group']}, {data['decomposition']}]\n"
           f"colour = fixed digit-{fixed_digit} (0-9),  text = swept class,  red ring = misclassified")
    fig.suptitle(sup, fontsize=11)
    fig.subplots_adjust(top=0.80, wspace=0.3)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Wrote {out}")


def plot_symmetry(data, names, sweep_digit, pcs, correct, out):
    """Per-fixed-digit small multiples: 10 rows x N-subsystem columns; colour = swept class."""
    group = str(data['group'])
    fixed_val = data['fixed_val'].astype(int)
    swept_class = data['swept_class'].astype(int)
    group_code = data['group_code'].astype(int)
    sym_labels = GROUP_SYM_LABELS[group]
    point_labels = [sym_labels[c] for c in group_code]

    fixed_vals = sorted(np.unique(fixed_val).tolist())
    n_rows, n_cols = len(fixed_vals), len(names)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.6 * n_rows),
                             squeeze=False)
    px, py = pcs

    # Pre-compute PCA per (fixed digit, subsystem) so columns can share a scale.
    cell = {}
    for fv in fixed_vals:
        sel = np.where(fixed_val == fv)[0]
        for name in names:
            rho = data[f'rho_{name}'][sel]
            coords, var, _ = pca_across_samples(rho.reshape(rho.shape[0], -1))
            cell[(fv, name)] = (sel, coords, var)
    col_R = {name: sym_range([cell[(fv, name)][1] for fv in fixed_vals], px, py)
             for name in names}

    fixed_digit = 1 if sweep_digit == 2 else 2
    print(f"\n[symmetry, sweep digit-{sweep_digit}] per fixed digit-{fixed_digit} x subsystem "
          f"stats (total variance | mean align(ones) | mean dev(ones)):")
    for r, fv in enumerate(fixed_vals):
        row_bits = []
        for c, name in enumerate(names):
            ax = axes[r][c]
            sel, coords, var = cell[(fv, name)]
            tvar = float(np.var(data[f'rho_{name}'][sel].reshape(len(sel), -1), axis=0).sum())
            align = data[f'align_{name}'][sel]
            # orbit key = swept class (joins the n_sym transforms of one class), ordered by code
            order = np.lexsort((group_code[sel], swept_class[sel]))
            title = (f"fix digit-{fixed_digit}={fv} | rho_{name}\n"
                     f"var={tvar:.2e}  align={align.mean():.3f}  dev={1.0 - align.mean():.3f}")
            scatter_panel(ax, coords[order], (px, py), np.array(point_labels)[sel][order],
                          swept_class[sel][order], correct[sel][order], var, title,
                          orbit_keys=swept_class[sel][order])
            R = col_R[name]
            ax.set_xlim(-R, R); ax.set_ylim(-R, R)
            ax.set_aspect('equal', adjustable='box')
            row_bits.append(f"rho_{name}: var={tvar:.3e} align={align.mean():.4f} "
                            f"dev={1.0 - align.mean():.4f}")
        print(f"  digit-{fixed_digit}={fv}:  " + "   ".join(row_bits))

    sup = (f"All-pairs reduced-density PCA, sweeping digit-{sweep_digit}'s "
           f"{group.split('x')[0]} orbit, one row per fixed digit-{fixed_digit}   "
           f"[{group}, {data['decomposition']}]\n"
           f"colour = swept class (0-9),  text = group element,  lines = per-class orbit,  "
           f"red ring = misclassified")
    # The figure is very tall, so set the title band in absolute inches (a fractional top
    # margin would scale with height and open a huge gap below the suptitle).
    fig_h = 4.6 * n_rows
    fig.subplots_adjust(top=1.0 - 1.1 / fig_h, hspace=0.55, wspace=0.3)
    fig.suptitle(sup, fontsize=12, y=1.0 - 0.35 / fig_h)
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Wrote {out}")


def plot_summary(data, names, sweep_digit, out):
    """Box/strip plots of robustness metrics across all pairs, per subsystem."""
    fixed_val = data['fixed_val'].astype(int)
    fixed_vals = sorted(np.unique(fixed_val).tolist())
    swept_name = f'digit{sweep_digit}'
    fixed_name = f'digit{1 if sweep_digit == 2 else 2}'

    # per-fixed-digit total variance of each subsystem (10 values -> a distribution)
    tvar = {name: [] for name in names}
    for fv in fixed_vals:
        sel = np.where(fixed_val == fv)[0]
        for name in names:
            rho = data[f'rho_{name}'][sel]
            tvar[name].append(float(np.var(rho.reshape(len(sel), -1), axis=0).sum()))
    tvar = {k: np.array(v) for k, v in tvar.items()}

    spread = {f'{swept_name}/{fixed_name}':
              tvar[swept_name] / np.clip(tvar[fixed_name], 1e-300, None)}
    if 'outer' in names:
        spread[f'{swept_name}/outer'] = tvar[swept_name] / np.clip(tvar['outer'], 1e-300, None)

    entropy_keys = [k for k in data.files if k.startswith('entropy')]

    def box(ax, series, ticklabels):
        ax.boxplot(series)  # positions default to 1..len(series); set names cross-version
        ax.set_xticks(range(1, len(ticklabels) + 1))
        ax.set_xticklabels(ticklabels)

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    box(axes[0], list(spread.values()), list(spread.keys()))
    axes[0].axhline(1.0, color='red', ls='--', lw=1)
    axes[0].set_yscale('log'); axes[0].set_ylabel('spread ratio (per fixed digit)')
    axes[0].set_title('spread ratio swept / fixed (>1 = swept varies more)')

    box(axes[1], [tvar[name] for name in names], names)
    axes[1].set_yscale('log'); axes[1].set_ylabel('total variance (per fixed digit)')
    axes[1].set_title('subsystem total variance')

    box(axes[2], [data[f'align_{name}'] for name in names], names)
    axes[2].set_ylim(0, 1.02); axes[2].set_ylabel('|<top eigvec, all-ones>|')
    axes[2].set_title('all-ones alignment (1 = collapsed to trivial irrep)')

    if entropy_keys:
        box(axes[3], [data[k] for k in entropy_keys],
            [k.replace('entropy_', '').replace('entropy', 'd1:d2') for k in entropy_keys])
        axes[3].set_ylim(0, 1.02); axes[3].set_ylabel('normalized vN entropy')
        axes[3].set_title('entanglement per cut')
        axes[3].tick_params(axis='x', rotation=20)
    else:
        axes[3].axis('off')

    sup = (f"Robustness summary across all pairs: {data['variation']}, sweep digit-{sweep_digit}"
           f"   [{data['group']}, {data['decomposition']}]")
    fig.suptitle(sup, fontsize=12)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--npz', type=str, required=True,
                        help='Path to a <variation>_sweep<d>.npz from the robustness compute script.')
    parser.add_argument('--out', type=str, default=None, help='Output image path (default: <npz>.png)')
    parser.add_argument('--pcs', type=str, default='1,2', help='Which two PCs to plot, 1-indexed.')
    parser.add_argument('--correctness', type=str, default='swept', choices=['swept', 'full'],
                        help="What counts as correct for the red ring: the swept digit's class, "
                             "or the full 2-digit prediction.")
    parser.add_argument('--summary', action='store_true',
                        help='Also write a quantitative robustness-summary figure (<out>_summary.png).')
    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)
    variation = str(data['variation'])
    sweep_digit = int(data['sweep_digit'])
    names = [str(s) for s in data['subsystem_names']]
    px, py = [int(p) - 1 for p in args.pcs.split(',')]

    correct = correctness(data['pred_class'].astype(int), data['fixed_val'].astype(int),
                          data['swept_class'].astype(int), sweep_digit, args.correctness)

    out = args.out or (args.npz.rsplit('.', 1)[0] + '.png')
    if variation == 'digit':
        plot_digit(data, names, sweep_digit, (px, py), correct, out)
    else:
        plot_symmetry(data, names, sweep_digit, (px, py), correct, out)
    print(f"correctly classified ({args.correctness}): {int(correct.sum())}/{len(correct)}")

    if args.summary:
        plot_summary(data, names, sweep_digit, out.rsplit('.', 1)[0] + '_summary.png')


if __name__ == "__main__":
    main()
