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


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.ckpt).')
    parser.add_argument('--out_prefix', type=str, required=True,
                        help='Output path prefix; writes <prefix>.csv and <prefix>.npz')
    parser.add_argument('--group', type=str, default=None, choices=list(GROUP_CONFIG.keys()),
                        help="Group / decomposition. If omitted, taken from the checkpoint's "
                             "hparams.group. Must match the training group.")
    parser.add_argument('--variation', type=str, default='symmetry', choices=['symmetry', 'digit'],
                        help="What to sweep digit-2 over: 'symmetry' = the per-digit group "
                             "transforms of the same digit; 'digit' = one instance of each of the "
                             "10 digit classes. Digit-1 is fixed in both.")
    parser.add_argument('--layer_id', type=int, default=12, help='DDMNISTCNN layer to extract latents from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed; selects the random base pair.')
    parser.add_argument('--base_idx', type=int, default=None, help='Override the random base-pair index.')
    parser.add_argument('--n_pcs_csv', type=int, default=3, help='Number of PCs to write to the CSV.')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    out_dir = os.path.dirname(args.out_prefix)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

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

    idx = args.base_idx if args.base_idx is not None else random.randrange(len(base.labels))
    img1, img2, y = base[idx]  # img1, img2: [1,28,28]; img1 = the FIXED digit-1
    label = int(y)
    print(f"Group {group} ({cfg['decomposition']}), base pair index {idx}, label {label} "
          f"(digit-1={label // 10}, digit-2={label % 10})")
    print(f"Variation mode: {args.variation}")

    # ---- build the digit-2 samples (digit-1 fixed in both modes) ----
    rng = random.Random(args.seed)
    transform = cfg['transform']
    if args.variation == 'symmetry':
        # the per-digit group transforms of the SAME digit-2
        samples = [(transform(img2, h), h) for h in range(cfg['n_sym'])]
    else:  # 'digit': one instance of each of the 10 digit classes, canonical orientation
        unit_labels = base.labels % 10
        samples = []
        for k in range(10):
            cand = (unit_labels == k).nonzero(as_tuple=True)[0].tolist()
            j = rng.choice(cand)
            _, img2_k, _ = base[j]  # use that sample's digit-2 image
            samples.append((img2_k, k))

    var_codes = [code for _, code in samples]
    n = len(var_codes)
    images = [base._combine_images(img1[0], d2[0]) for d2, _ in samples]  # each [1,56,56], normalized
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
    role = {'outer': 'I, multiplicity', 'digit1': 'digit-1, fixed', 'digit2': f'digit-2, {swept}'}
    for name in names:
        p = pca[name]
        print(f"rho_{name} ({role[name]})   total variance = {p['tvar']:.3e}")
        print(f"    explained-variance ratio: {np.round(p['var'], 4)}")
        print(f"    absolute variance per PC: {p['absvar']}")
    tv = {name: pca[name]['tvar'] for name in names}
    print(f"spread ratio  rho_digit2 / rho_digit1 = {tv['digit2'] / max(tv['digit1'], 1e-300):.1f}")
    if 'outer' in names:
        print(f"spread ratio  rho_digit2 / rho_outer  = {tv['digit2'] / max(tv['outer'], 1e-300):.1f}")

    # ---- entropy column for the CSV ----
    # bipartite: the single normalized vN entropy; tripartite: the digit-2 cut (digit2 : rest).
    csv_entropy = entropy['entropy'] if cfg['decomposition'] == 'bipartite' else entropy['entropy_c_ab']

    # ---- CSV ----
    n_pcs = args.n_pcs_csv
    csv_path = f"{args.out_prefix}.csv"

    def pc_row(coords, i):
        vals = coords[i].tolist()
        vals = vals[:n_pcs] + [float('nan')] * max(0, n_pcs - len(vals))
        return vals

    header = ['subsystem', 'group', 'var_kind', 'var_code', 'pred_class', 'pred_tens',
              'pred_unit', 'entropy'] + [f'pc{j + 1}' for j in range(n_pcs)]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for name in names:
            for i, code in enumerate(var_codes):
                writer.writerow([name, group, args.variation, int(code), int(pred[i]),
                                 int(pred_tens[i]), int(pred_unit[i]), float(csv_entropy[i])]
                                + pc_row(pca[name]['coords'], i))
    print(f"Wrote {csv_path}")

    # ---- NPZ ----
    npz_path = f"{args.out_prefix}.npz"
    payload = dict(
        group=group,
        decomposition=cfg['decomposition'],
        subsystem_names=np.array(names),
        subsystem_dims=np.array(dims),
        base_idx=idx,
        label=label,
        variation=args.variation,
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
    for name in names:
        p = pca[name]
        payload[f'rho_{name}'] = rhos[name]
        payload[f'pca_{name}_coords'] = p['coords']
        payload[f'pca_{name}_explained_var'] = p['var']
        payload[f'pca_{name}_absolute_var'] = p['absvar']
        payload[f'pca_{name}_total_var'] = p['tvar']
        payload[f'pca_{name}_components'] = p['comp']
    np.savez(npz_path, **payload)
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
