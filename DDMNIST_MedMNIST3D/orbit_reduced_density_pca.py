"""
Reduced-density PCA over a single digit's D4 orbit (DDMNIST, D4xD4).

Pick one random base digit pair, fix digit-1, sweep digit-2 over all 8 D4 group
elements, encode each image with a trained GxGRegularFunctor checkpoint, and form
the two reduced density matrices of the leading 8x8 latent factor:
    rho_A = digit-1 marginal   (expected ~invariant: digit-1 is held fixed)
    rho_B = digit-2 marginal   (expected to trace the rotation orbit)
Then run PCA across the 8 samples, separately for the rho_A set and the rho_B set.

Single checkpoint. Writes a CSV (PCA coords + predictions + entropy per element)
and an NPZ (full arrays). No plotting.

Run from inside DDMNIST_MedMNIST3D/ so the `models`/`datasets`/`utils` imports resolve, e.g.:
    python orbit_reduced_density_pca.py --ckpt <path/best_model.ckpt> --out_prefix /tmp/orbit --seed 0
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
from utils.entanglement import Entanglement


# D4 net rotation (degrees) per code; codes >= 4 are flip-then-rotate.
# Matches PairedD4xD4DDMNIST.transform_d4 conventions, minus the random jitter.
D4_NET_ROT = [0, 90, 180, 270, 0, 90, 180, 270]


def transform_d4_clean(img: torch.Tensor, code: int) -> torch.Tensor:
    """Exact D4 action (no interpolation jitter). code 0 is a true no-op."""
    x = TF.hflip(img) if code >= 4 else img
    net_rot = D4_NET_ROT[code]
    if net_rot != 0:
        x = TF.rotate(x, float(net_rot), InterpolationMode.BILINEAR)
    return x


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.ckpt).')
    parser.add_argument('--out_prefix', type=str, required=True,
                        help='Output path prefix; writes <prefix>.csv and <prefix>.npz')
    parser.add_argument('--group', type=str, default='D4xD4')
    parser.add_argument('--layer_id', type=int, default=12, help='DDMNISTCNN layer to extract latents from.')
    parser.add_argument('--dim_a', type=int, default=8, help='Digit-1 factor dimension.')
    parser.add_argument('--dim_b', type=int, default=8, help='Digit-2 factor dimension.')
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

    # ---- data: one random base pair ----
    dm = D4xD4DDMNISTDataModule(batch_size=8)
    dm.setup(stage='test')
    base = dm.test_dataset.data  # DDMNIST: raw [1,28,28] digits + label

    idx = args.base_idx if args.base_idx is not None else random.randrange(len(base.labels))
    img1, img2, y = base[idx]  # img1, img2: [1,28,28]
    label = int(y)
    print(f"Base pair index {idx}, label {label} (digit-1={label // 10}, digit-2={label % 10})")

    # ---- 8 images: digit-1 fixed, digit-2 swept over D4 ----
    d4_codes = list(range(8))
    images = []
    for h in d4_codes:
        t2 = transform_d4_clean(img2, h)
        combined = base._combine_images(img1[0], t2[0])  # [1,56,56], normalized
        images.append(combined)
    X = torch.stack(images, dim=0).to(device)  # (8,1,56,56)

    # ---- forward: latents + predictions ----
    with torch.no_grad():
        logits, latents = model.model(X, [args.layer_id])
        Z = latents[0]                     # (8, 66)
        pred = logits.argmax(dim=1)        # (8,)

    Z = Z.cpu()
    pred = pred.cpu().numpy()
    pred_tens = pred // 10                 # digit-1 class (should be ~constant)
    pred_unit = pred % 10                  # digit-2 class

    # ---- states + reduced density matrices (reuse Entanglement) ----
    dim_a, dim_b = args.dim_a, args.dim_b
    d = dim_a * dim_b
    states = Z[:, :d].float()
    states = states / states.norm(dim=1, keepdim=True).clamp_min(1e-12)  # unit vectors

    ent = Entanglement(states, dim_a, dim_b)
    rho = ent.rho                                      # (8, d, d)
    # NOTE: by einsum+shape (not the misleading names), partial_trace_A -> rho_A, partial_trace_B -> rho_B.
    rho_A = ent.partial_trace_A(rho, dim_a, dim_b)     # (8, dim_a, dim_a) digit-1 marginal
    rho_B = ent.partial_trace_B(rho, dim_a, dim_b)     # (8, dim_b, dim_b) digit-2 marginal
    entropy = ent.compute(normalize=True)["entanglement_a"]  # (8,) normalized von Neumann entropy

    rho_A = rho_A.numpy()
    rho_B = rho_B.numpy()
    entropy = entropy.numpy()

    # ---- PCA across the 8 samples, separately per subsystem ----
    M_A = rho_A.reshape(len(d4_codes), -1)  # (8, dim_a*dim_a)
    M_B = rho_B.reshape(len(d4_codes), -1)  # (8, dim_b*dim_b)
    pca_A_coords, pca_A_var, pca_A_comp = pca_across_samples(M_A)
    pca_B_coords, pca_B_var, pca_B_comp = pca_across_samples(M_B)

    print(f"rho_A (digit-1, fixed)   PCA explained variance: {np.round(pca_A_var, 4)}  "
          f"total spread (sum S^2) = {np.var(M_A, axis=0).sum():.3e}")
    print(f"rho_B (digit-2, rotated) PCA explained variance: {np.round(pca_B_var, 4)}  "
          f"total spread (sum S^2) = {np.var(M_B, axis=0).sum():.3e}")

    # ---- CSV ----
    n_pcs = args.n_pcs_csv
    csv_path = f"{args.out_prefix}.csv"

    def pc_row(coords, i):
        # pad with NaN if fewer than n_pcs available (single pair -> up to 7 PCs)
        vals = coords[i].tolist()
        vals = vals[:n_pcs] + [float('nan')] * max(0, n_pcs - len(vals))
        return vals

    header = ['subsystem', 'd4_element', 'pred_class', 'pred_tens', 'pred_unit', 'entropy'] \
        + [f'pc{j + 1}' for j in range(n_pcs)]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for i, h in enumerate(d4_codes):
            writer.writerow(['A_digit1', h, int(pred[i]), int(pred_tens[i]), int(pred_unit[i]),
                             float(entropy[i])] + pc_row(pca_A_coords, i))
        for i, h in enumerate(d4_codes):
            writer.writerow(['B_digit2', h, int(pred[i]), int(pred_tens[i]), int(pred_unit[i]),
                             float(entropy[i])] + pc_row(pca_B_coords, i))
    print(f"Wrote {csv_path}")

    # ---- NPZ ----
    npz_path = f"{args.out_prefix}.npz"
    np.savez(
        npz_path,
        base_idx=idx,
        label=label,
        d4_codes=np.array(d4_codes),
        images=X.cpu().numpy(),
        latents=Z.numpy(),
        states=states.numpy(),
        rho_A=rho_A,
        rho_B=rho_B,
        pred_class=pred,
        pred_tens=pred_tens,
        pred_unit=pred_unit,
        entropy=entropy,
        pca_A_coords=pca_A_coords,
        pca_B_coords=pca_B_coords,
        pca_A_explained_var=pca_A_var,
        pca_B_explained_var=pca_B_var,
        pca_A_components=pca_A_comp,
        pca_B_components=pca_B_comp,
    )
    print(f"Wrote {npz_path}")


if __name__ == "__main__":
    main()
