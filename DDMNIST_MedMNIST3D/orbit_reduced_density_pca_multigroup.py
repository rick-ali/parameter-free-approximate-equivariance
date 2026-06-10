"""
Reduced-density PCA over a single digit's variation (DDMNIST), multi-group.

Generalizes orbit_reduced_density_pca.py (which is D4xD4-only, bipartite) to also
handle the product groups C4xC4 and D1xD1 via tripartite decomposition. The leading
64-dim latent block factorizes per group (slowest -> fastest tensor index):

    D4xD4 : digit1(8) (x) digit2(8)                 -> bipartite  (no outer factor)
    C4xC4 : outer(4) (x) digit1(4) (x) digit2(4)    -> tripartite (I (x) rho (x) rho)
    D1xD1 : outer(16) (x) digit1(2) (x) digit2(2)   -> tripartite (I (x) rho (x) rho)

The "outer" factor is the identity multiplicity space the architecture introduces
when the product group's regular rep (size B) is smaller than the 64-dim block
(W_full = kron(I_outer, W_productgroup); see GxGRegularFunctorModel._build_W_full).

Pick one random base digit pair, fix digit-1, and sweep digit-2 according to --variation:
    symmetry : the per-digit group transforms of the same digit-2 (8 D4 / 4 C4 / 2 D1)
    digit    : one instance of each of the 10 digit classes as digit-2 (canonical orientation)
Encode each image with a trained GxGRegularFunctor checkpoint, form the reduced density
matrix of each subsystem, then run PCA across the n samples separately per subsystem.

Single checkpoint. Writes a CSV (PCA coords + predictions + entropy per element) and an
NPZ (full arrays, subsystem-indexed). No plotting.

Run from inside DDMNIST_MedMNIST3D/ so the `models`/`datasets`/`utils` imports resolve, e.g.:
    python orbit_reduced_density_pca_multigroup.py --group C4xC4 --ckpt <path/best_model.ckpt> \
        --out_prefix /tmp/orbit --variation symmetry --seed 0
"""
import argparse
import csv
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
import pytorch_lightning as pl

from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule
from utils.entanglement import Entanglement, TripartiteEntanglement


# ---- clean group actions (exact, no interpolation jitter); code 0 is a true no-op ----
D4_NET_ROT = [0, 90, 180, 270, 0, 90, 180, 270]  # codes >= 4 are flip-then-rotate


def transform_d4_clean(img: torch.Tensor, code: int) -> torch.Tensor:
    """Exact D4 action. Matches PairedD4xD4DDMNIST.transform_d4 minus the random jitter."""
    x = TF.hflip(img) if code >= 4 else img
    net_rot = D4_NET_ROT[code]
    if net_rot != 0:
        x = TF.rotate(x, float(net_rot), InterpolationMode.BILINEAR)
    return x


def transform_c4_clean(img: torch.Tensor, code: int) -> torch.Tensor:
    """Exact C4 rotation. Matches PairedC4xC4DDMNIST.augment_image minus the jitter."""
    net_rot = [0, 90, 180, 270][code]
    if net_rot != 0:
        img = TF.rotate(img, float(net_rot), InterpolationMode.BILINEAR)
    return img


def transform_d1_clean(img: torch.Tensor, code: int) -> torch.Tensor:
    """Exact D1 action. Matches PairedD1xD1DDMNIST.transform_d1."""
    return TF.hflip(img) if code == 1 else img


# ---- per-group configuration ----
# dims/names are for the leading 64-dim latent block (slowest -> fastest tensor index).
GROUP_CONFIG = {
    'D4xD4': {
        'datamodule': D4xD4DDMNISTDataModule,
        'transform': transform_d4_clean,
        'n_sym': 8,
        'sym_labels': ['e', 'r', 'r2', 'r3', 's', 'rs', 'r2s', 'r3s'],
        'decomposition': 'bipartite',
        'dims': (8, 8),
        'names': ['digit1', 'digit2'],
    },
    'C4xC4': {
        'datamodule': C4xC4DDMNISTDataModule,
        'transform': transform_c4_clean,
        'n_sym': 4,
        'sym_labels': ['e', 'r', 'r2', 'r3'],
        'decomposition': 'tripartite',
        'dims': (4, 4, 4),
        'names': ['outer', 'digit1', 'digit2'],
    },
    'D1xD1': {
        'datamodule': D1xD1DDMNISTDataModule,
        'transform': transform_d1_clean,
        'n_sym': 2,
        'sym_labels': ['e', 's'],
        'decomposition': 'tripartite',
        'dims': (16, 2, 2),
        'names': ['outer', 'digit1', 'digit2'],
    },
}


def pca_across_samples(flat: np.ndarray):
    """PCA over rows of `flat` (n_samples, n_features) via SVD of the centered data.

    Returns (coords (n_samples, k), explained_var_ratio (k,), components (k, n_features))
    where k = min(n_samples - 1, n_features) nonzero-ish directions retained.
    """
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean
    # full_matrices=False -> U (n, r), S (r,), Vt (r, features) with r = min(n, features)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    coords = U * S  # (n_samples, r) projection onto principal axes
    total = (S ** 2).sum()
    explained = (S ** 2) / total if total > 0 else np.zeros_like(S)
    return coords, explained, Vt


def bipartite_marginals(states: torch.Tensor, dim_a: int, dim_b: int):
    """Reduced density matrices + entropy for a bipartite pure state (digit1, digit2).

    Returns dict name -> rho (n, d_name, d_name) and an entropy array (n,).
    By the einsum (not the misleading method names), partial_trace_A -> rho_A and
    partial_trace_B -> rho_B (see note in orbit_reduced_density_pca.py).
    """
    ent = Entanglement(states, dim_a, dim_b)
    rho = ent.rho
    rho_d1 = ent.partial_trace_A(rho, dim_a, dim_b)  # (n, dim_a, dim_a) digit-1 marginal
    rho_d2 = ent.partial_trace_B(rho, dim_a, dim_b)  # (n, dim_b, dim_b) digit-2 marginal
    entropy = ent.compute(normalize=True)["entanglement_a"]  # (n,) normalized vN entropy
    rhos = {'digit1': rho_d1.numpy(), 'digit2': rho_d2.numpy()}
    return rhos, {'entropy': entropy.numpy()}


def tripartite_marginals(states: torch.Tensor, dim_a: int, dim_b: int, dim_c: int):
    """Reduced density matrices + the three cut entropies for a tripartite pure state.

    Subsystems are (outer, digit1, digit2) = (A, B, C). For each isolated subsystem we
    reuse TripartiteEntanglement's axis permutation, then keep the FIRST factor with
    Entanglement.partial_trace_A (which retains subsystem A; the names are misleading).
    """
    tri = TripartiteEntanglement(states, dim_a, dim_b, dim_c)
    name_by_sub = {'A': 'outer', 'B': 'digit1', 'C': 'digit2'}
    rhos = {}
    for sub, name in name_by_sub.items():
        vecs, dim_iso, dim_rest = tri._isolated_bipartite_vectors(sub)
        e = Entanglement(vecs, dim_iso, dim_rest)
        rho_iso = e.partial_trace_A(e.rho, dim_iso, dim_rest)  # (n, dim_iso, dim_iso)
        rhos[name] = rho_iso.numpy()
    cuts = tri.compute(normalize=True)
    entropy = {
        'entropy_a_bc': cuts['entanglement_a_bc'].numpy(),  # outer : (digit1 digit2)
        'entropy_b_ac': cuts['entanglement_b_ac'].numpy(),  # digit1 : (outer digit2)
        'entropy_c_ab': cuts['entanglement_c_ab'].numpy(),  # digit2 : (outer digit1)
    }
    return rhos, entropy


def report_subsystem_vectors(rhos, names, dims, var_codes, cfg, variation, role,
                             decimals=3, eig_spectrum=False):
    """Print the dominant learned vector of each subsystem and its alignment with all-ones.

    The all-ones direction is the unique G-invariant (trivial-irrep) direction of a regular
    representation: every permutation matrix fixes it, so a subsystem that collapses onto it
    satisfies its equivariance constraint for free. For each subsystem reduced density matrix
    rho (n, d, d) we take the top eigenvector as the subsystem's dominant pure direction
    (magnitude irrelevant: it is unit norm). We report, across the sweep:
        topeig : the top eigenvalue (purity; ~1 => the marginal is ~pure / product-like)
        align  : |<top_vec, 1/sqrt(d)>| in [0, 1]  (1 => collapsed to the all-ones direction)
    plus a per-subsystem summary (mean alignment, and a consensus 'drift' = how much the
    direction moves across the sweep).
    """
    np.set_printoptions(precision=decimals, suppress=True)
    dim_by_name = dict(zip(names, dims))
    # row labels: symmetry -> group-element names; digit -> the swept digit class
    if variation == 'symmetry':
        row_labels = [cfg['sym_labels'][int(c)] for c in var_codes]
    else:
        row_labels = [str(int(c)) for c in var_codes]
    lbl_w = max(4, *(len(s) for s in row_labels))

    print("\n=== Subsystem learned vectors (alignment with the all-ones / trivial-irrep direction) ===")
    print("align = |<top eigvec, 1/sqrt(d)>| in [0,1];  align~1 AND topeig~1 => collapsed to all-ones "
          "(trivial equivariance)")

    out = {}  # name -> dict(topvec, topeig, align) for optional NPZ stashing
    for name in names:
        d = int(dim_by_name[name])
        ones = np.ones(d) / np.sqrt(d)
        rho = rhos[name]                       # (n, d, d), real symmetric (PSD)
        evals, evecs = np.linalg.eigh(rho)     # ascending; evecs[:, :, k] is k-th eigvec
        top_eval = evals[:, -1]                # (n,)
        top_vec = evecs[:, :, -1]              # (n, d), unit norm
        # canonical sign: make the largest-|component| positive so rows are comparable
        lead = np.argmax(np.abs(top_vec), axis=1)
        sign = np.sign(top_vec[np.arange(top_vec.shape[0]), lead])
        sign[sign == 0] = 1.0
        top_vec = top_vec * sign[:, None]
        align = np.abs(top_vec @ ones)         # (n,) in [0, 1]

        print(f"\n[{name}]  ({role[name]})   d={d}   ones-ref = 1/sqrt({d}) = {1.0/np.sqrt(d):.4f}")
        print(f"  {'code':<{lbl_w}}  topeig   align   top-eigenvector (unit direction, canonical sign)")
        for i, rl in enumerate(row_labels):
            print(f"  {rl:<{lbl_w}}  {top_eval[i]:.4f}  {align[i]:.4f}  {top_vec[i]}")
            if eig_spectrum:
                print(f"  {'':<{lbl_w}}  eigenvalues: {evals[i][::-1]}")

        # consensus direction across the sweep (top singular vector of the stacked unit vectors),
        # drift = 1 - mean |cos(v_i, consensus)|: ~0 for a fixed subsystem, larger when it transforms
        _, _, Vt = np.linalg.svd(top_vec, full_matrices=False)
        consensus = Vt[0]
        drift = 1.0 - float(np.mean(np.abs(top_vec @ consensus)))
        verdict = ("COLLAPSED to all-ones (trivial equivariance)"
                   if align.mean() > 0.9 else "varying / not collapsed")
        print(f"  summary: mean align = {align.mean():.4f} | mean topeig = {top_eval.mean():.4f} "
              f"| drift = {drift:.4f}  ->  {verdict}")
        out[name] = dict(topvec=top_vec, topeig=top_eval, align=align)

    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.ckpt).')
    parser.add_argument('--out_prefix', type=str, default=None,
                        help='Output path prefix; writes <prefix>.csv and <prefix>.npz. '
                             'If omitted, defaults to '
                             'results/<group>/T<lambda_t>_E<lambda_e>_<digit1><digit2>')
    parser.add_argument('--group', type=str, default=None, choices=list(GROUP_CONFIG.keys()),
                        help="Group / decomposition. If omitted, taken from the checkpoint's "
                             "hparams.group. Must match the training group.")
    parser.add_argument('--variation', type=str, default='symmetry', choices=['symmetry', 'digit'],
                        help="What to sweep the moved digit over: 'symmetry' = the per-digit group "
                             "transforms of the same digit; 'digit' = one instance of each of the "
                             "10 digit classes. The other digit is fixed in both.")
    parser.add_argument('--sweep_digit', type=int, default=2, choices=[1, 2],
                        help="Which digit to vary (the other is held fixed). Default 2 = sweep "
                             "digit-2 (backward compatible). The latent factor structure is always "
                             "outer (x) digit1 (x) digit2; only the fixed/swept roles flip.")
    parser.add_argument('--layer_id', type=int, default=12, help='DDMNISTCNN layer to extract latents from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed; selects the random base pair.')
    parser.add_argument('--base_idx', type=int, default=None, help='Override the random base-pair index.')
    parser.add_argument('--digit1', type=int, default=None, choices=range(10), metavar='[0-9]',
                        help="Desired digit-1 class (0-9) for the base pair. If omitted, taken from "
                             "the random / --base_idx pair. Note: if you sweep digit-1 with "
                             "--variation digit, its base class is overwritten by the class sweep.")
    parser.add_argument('--digit2', type=int, default=None, choices=range(10), metavar='[0-9]',
                        help="Desired digit-2 class (0-9) for the base pair. If omitted, taken from "
                             "the random / --base_idx pair. Note: if you sweep digit-2 with "
                             "--variation digit, its base class is overwritten by the class sweep.")
    parser.add_argument('--n_pcs_csv', type=int, default=3, help='Number of PCs to write to the CSV.')
    parser.add_argument('--print_vectors', action='store_true',
                        help="Print each subsystem's dominant learned vector and its alignment with "
                             "the all-ones (trivial-irrep) direction across the sweep.")
    parser.add_argument('--eig_spectrum', action='store_true',
                        help="With --print_vectors, also print the full eigenvalue spectrum of each "
                             "subsystem's reduced density matrix per swept sample.")
    parser.add_argument('--vec_decimals', type=int, default=3,
                        help='Decimal places for printed eigenvectors (with --print_vectors).')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    # ---- model ----
    model = GxGRegularFunctor.load_from_checkpoint(args.ckpt, map_location=device)
    model.eval()
    model.to(device)
    model.model.get_latent = True  # required when the ckpt trained with lambda_t = lambda_e = 0

    # ---- group / decomposition ----
    group = args.group if args.group is not None else str(model.hparams.get('group'))
    if group not in GROUP_CONFIG:
        raise ValueError(f"Group {group!r} not supported. Choose from {list(GROUP_CONFIG)}.")
    cfg = GROUP_CONFIG[group]
    dims = cfg['dims']
    d = int(np.prod(dims))  # 64 for all supported groups
    assert d == 64, f"Expected a 64-dim latent block for {group}, got {d}."
    if args.group is None:
        print(f"--group not given; using checkpoint group: {group}")

    # ---- data: one random base pair ----
    dm = cfg['datamodule'](batch_size=8)
    dm.setup(stage='test')
    base = dm.test_dataset.data  # DDMNIST: raw [1,28,28] digits + label

    if args.base_idx is not None:
        if args.digit1 is not None or args.digit2 is not None:
            print("Note: --base_idx given; ignoring --digit1/--digit2.")
        idx = args.base_idx
    elif args.digit1 is not None or args.digit2 is not None:
        # pick a random base pair whose digit-1 / digit-2 classes match the request
        tens, units = base.labels // 10, base.labels % 10
        mask = torch.ones(len(base.labels), dtype=torch.bool)
        if args.digit1 is not None:
            mask &= (tens == args.digit1)
        if args.digit2 is not None:
            mask &= (units == args.digit2)
        cand = mask.nonzero(as_tuple=True)[0].tolist()
        if not cand:
            raise ValueError(f"No base pair with digit-1={args.digit1}, digit-2={args.digit2}.")
        idx = random.choice(cand)
    else:
        idx = random.randrange(len(base.labels))
    img1, img2, y = base[idx]  # img1, img2: [1,28,28]; img1 = the FIXED digit-1
    label = int(y)
    sweep_digit = args.sweep_digit
    fixed_digit = 1 if sweep_digit == 2 else 2
    print(f"Group {group} ({cfg['decomposition']}), base pair index {idx}, label {label} "
          f"(digit-1={label // 10}, digit-2={label % 10})")
    print(f"Variation mode: {args.variation}; sweeping digit-{sweep_digit} (digit-{fixed_digit} fixed)")

    # ---- resolve the output prefix (default: results/<group>/T<lambda_t>_E<lambda_e>_<d1><d2>) ----
    if args.out_prefix is not None:
        out_prefix = args.out_prefix
    else:
        lt, le = model.hparams.get('lambda_t'), model.hparams.get('lambda_e')
        out_prefix = os.path.join('results', group, f"{'digit' if args.variation == 'digit' else 'orbit'}_T{lt}_E{le}_{'v' if args.sweep_digit == 1 else ''}{label // 10}{'v' if args.sweep_digit == 2 else ''}{label % 10}")
    out_dir = os.path.dirname(out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # ---- build the swept-digit samples (the other digit is held fixed) ----
    # The latent factor structure is always outer (x) digit1 (x) digit2; sweep_digit only
    # decides which digit image we vary, so just the fixed/swept roles flip.
    rng = random.Random(args.seed)
    transform = cfg['transform']
    swept_base = img2 if sweep_digit == 2 else img1  # the digit image we vary

    if args.variation == 'symmetry':
        # the per-digit group transforms of the SAME swept digit
        samples = [(transform(swept_base, h), h) for h in range(cfg['n_sym'])]
    else:  # 'digit': one instance of each of the 10 digit classes, canonical orientation
        class_labels = (base.labels % 10) if sweep_digit == 2 else (base.labels // 10)
        samples = []
        for k in range(10):
            cand = (class_labels == k).nonzero(as_tuple=True)[0].tolist()
            j = rng.choice(cand)
            pair = base[j]
            swept_img_k = pair[1] if sweep_digit == 2 else pair[0]  # that sample's swept-digit image
            samples.append((swept_img_k, k))

    var_codes = [code for _, code in samples]
    n = len(var_codes)
    # _combine_images(first=digit1, second=digit2); keep the fixed digit in its slot.
    if sweep_digit == 2:
        images = [base._combine_images(img1[0], s[0]) for s, _ in samples]  # each [1,56,56], normalized
    else:
        images = [base._combine_images(s[0], img2[0]) for s, _ in samples]
    X = torch.stack(images, dim=0).to(device)  # (n,1,56,56)

    # ---- forward: latents + predictions ----
    with torch.no_grad():
        logits, latents = model.model(X, [args.layer_id])
        Z = latents[0]
        pred = logits.argmax(dim=1)

    Z = Z.cpu()
    pred = pred.cpu().numpy()
    pred_tens = pred // 10                 # digit-1 class (should be ~constant)
    pred_unit = pred % 10                  # digit-2 class

    # ---- states + reduced density matrices ----
    states = Z[:, :d].float()
    states = states / states.norm(dim=1, keepdim=True).clamp_min(1e-12)  # unit vectors

    if cfg['decomposition'] == 'bipartite':
        rhos, entropy = bipartite_marginals(states, dims[0], dims[1])
    else:
        rhos, entropy = tripartite_marginals(states, dims[0], dims[1], dims[2])

    names = cfg['names']

    # ---- PCA across the n samples, separately per subsystem ----
    pca = {}  # name -> dict(coords, var, comp, tvar, absvar)
    for name in names:
        M = rhos[name].reshape(n, -1)  # (n, d_name*d_name)
        coords, var, comp = pca_across_samples(M)
        tvar = float(np.var(M, axis=0).sum())
        pca[name] = dict(coords=coords, var=var, comp=comp, tvar=tvar, absvar=var * tvar)

    swept = 'rotated' if args.variation == 'symmetry' else 'swept over classes'
    np.set_printoptions(precision=4, suppress=True)
    swept_name = f'digit{sweep_digit}'
    fixed_name = f'digit{fixed_digit}'
    role = {'outer': 'I, multiplicity',
            fixed_name: f'digit-{fixed_digit}, fixed',
            swept_name: f'digit-{sweep_digit}, {swept}'}
    for name in names:
        p = pca[name]
        print(f"rho_{name} ({role[name]})   total variance = {p['tvar']:.3e}")
        print(f"    explained-variance ratio: {np.round(p['var'], 4)}")
        print(f"    absolute variance per PC: {p['absvar']}")
    tv = {name: pca[name]['tvar'] for name in names}
    print(f"spread ratio  rho_{swept_name} / rho_{fixed_name} = "
          f"{tv[swept_name] / max(tv[fixed_name], 1e-300):.1f}")
    if 'outer' in names:
        print(f"spread ratio  rho_{swept_name} / rho_outer  = "
              f"{tv[swept_name] / max(tv['outer'], 1e-300):.1f}")

    # ---- average entanglement of the n encodings plotted in the PCA ----
    # `entropy` holds the per-sample normalized von Neumann entropy in [0,1] for each cut
    # (bipartite: the single digit1:digit2 cut; tripartite: the three isolated-subsystem cuts).
    entropy_cut_label = {
        'entropy': f'digit1 : digit2',
        'entropy_a_bc': 'outer : (digit1 digit2)',
        'entropy_b_ac': 'digit1 : (outer digit2)',
        'entropy_c_ab': 'digit2 : (outer digit1)',
    }
    print(f"\naverage entanglement over the {n} plotted encodings (normalized vN entropy, [0,1]):")
    for key, vals in entropy.items():
        vals = np.asarray(vals)
        cut_label = entropy_cut_label.get(key, key)
        print(f"    {cut_label:<24} mean = {vals.mean():.4f}  (std {vals.std():.4f}, "
              f"min {vals.min():.4f}, max {vals.max():.4f})")

    # ---- optional: dominant subsystem vectors + all-ones alignment ----
    subsystem_vectors = None
    if args.print_vectors:
        subsystem_vectors = report_subsystem_vectors(
            rhos, names, dims, var_codes, cfg, args.variation, role,
            decimals=args.vec_decimals, eig_spectrum=args.eig_spectrum)

    # ---- entropy column for the CSV ----
    # bipartite: the single normalized vN entropy; tripartite: the digit-2 cut (digit2 : rest).
    csv_entropy = entropy['entropy'] if cfg['decomposition'] == 'bipartite' else entropy['entropy_c_ab']

    # ---- CSV ----
    n_pcs = args.n_pcs_csv
    csv_path = f"{out_prefix}.csv"

    def pc_row(coords, i):
        vals = coords[i].tolist()
        vals = vals[:n_pcs] + [float('nan')] * max(0, n_pcs - len(vals))
        return vals

    header = ['subsystem', 'group', 'var_kind', 'swept_digit', 'var_code', 'pred_class',
              'pred_tens', 'pred_unit', 'entropy'] + [f'pc{j + 1}' for j in range(n_pcs)]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in names:
            for i, code in enumerate(var_codes):
                writer.writerow([name, group, args.variation, sweep_digit, int(code), int(pred[i]),
                                 int(pred_tens[i]), int(pred_unit[i]), float(csv_entropy[i])]
                                + pc_row(pca[name]['coords'], i))
    print(f"Wrote {csv_path}")

    # ---- NPZ ----
    npz_path = f"{out_prefix}.npz"
    payload = dict(
        group=group,
        decomposition=cfg['decomposition'],
        subsystem_names=np.array(names),
        subsystem_dims=np.array(dims),
        base_idx=idx,
        label=label,
        variation=args.variation,
        sweep_digit=sweep_digit,
        var_codes=np.array(var_codes),
        images=X.cpu().numpy(),
        latents=Z.numpy(),
        states=states.numpy(),
        pred_class=pred,
        pred_tens=pred_tens,
        pred_unit=pred_unit,
    )
    payload.update({f'entropy_{k}' if not k.startswith('entropy') else k: v
                    for k, v in entropy.items()})
    # average entanglement over the n plotted encodings, per cut
    payload.update({(f'avg_entropy_{k}' if not k.startswith('entropy')
                     else f'avg_{k}'): float(np.asarray(v).mean())
                    for k, v in entropy.items()})
    for name in names:
        p = pca[name]
        payload[f'rho_{name}'] = rhos[name]
        payload[f'pca_{name}_coords'] = p['coords']
        payload[f'pca_{name}_explained_var'] = p['var']
        payload[f'pca_{name}_absolute_var'] = p['absvar']
        payload[f'pca_{name}_total_var'] = p['tvar']
        payload[f'pca_{name}_components'] = p['comp']
    if subsystem_vectors is not None:
        for name, v in subsystem_vectors.items():
            payload[f'topvec_{name}'] = v['topvec']   # (n, d_name) dominant unit direction
            payload[f'topeig_{name}'] = v['topeig']   # (n,) top eigenvalue (purity)
            payload[f'align_{name}'] = v['align']     # (n,) |<topvec, all-ones>|
    np.savez(npz_path, **payload)
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
