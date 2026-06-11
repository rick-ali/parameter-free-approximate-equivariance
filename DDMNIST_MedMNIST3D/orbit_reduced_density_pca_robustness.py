"""
Robustness sweep of the reduced-density PCA over ALL digit pairs (DDMNIST), multi-group.

Generalizes orbit_reduced_density_pca_multigroup.py (one random base pair) to the whole
10x10 digit grid, so the equivariance finding -- the *swept* digit's subsystem spreads while
the *fixed* digit's subsystem (and the `outer` multiplicity) stay collapsed onto the
all-ones / trivial-irrep direction -- can be shown to hold across every pair rather than one
cherry-picked example.

For a given (variation, sweep_digit) we fix one digit (digit-1 if sweep_digit==2, else
digit-2) and, for each of its 10 classes, build the swept-digit samples:
    digit    : one instance of each of the 10 digit classes as the swept digit -> 10 samples
    symmetry : for each of the 10 swept classes, all per-digit group transforms (8 D4 / 4 C4
               / 2 D1) of that class                                  -> 10 * n_sym samples
The model is loaded ONCE and every (variation, sweep_digit) requested is encoded in the same
process so the expensive forward passes are amortized over the full grid.

For each sample we form the leading-64-dim latent block, build the per-subsystem reduced
density matrices, and record per-sample metadata (predictions, embedding norms, entanglement,
top-eigenvector alignment with the all-ones direction). PCA is deliberately NOT done here:
the pooling rule differs per variant and is cheap, so plot_orbit_pca_robustness.py computes
it from the stored rho stacks (keeping the plot step torch-free).

Writes one NPZ per (variation, sweep_digit): <out_prefix>_<variation>_sweep<d>.npz

Run from inside DDMNIST_MedMNIST3D/, e.g.:
    python orbit_reduced_density_pca_robustness.py --group C4xC4 \
        --ckpt <path/best_model.ckpt> --out_prefix results/C4xC4/robust \
        --variations symmetry,digit --sweep_digits 1,2 --seed 0
"""
import argparse
import os
import random

import numpy as np
import torch
import pytorch_lightning as pl

from models.GxGRegularFunctorModel import GxGRegularFunctor

# Reuse the per-group config, PCA, and reduced-density helpers from the single-pair script
# (all module-level; its argparse only runs under __main__, so importing has no side effects).
from orbit_reduced_density_pca_multigroup import (
    GROUP_CONFIG,
    bipartite_marginals,
    tripartite_marginals,
)


def subsystem_top(rho: np.ndarray, d: int):
    """Top eigenvalue (purity) and all-ones alignment of each (n,d,d) reduced density matrix.

    Mirrors the eigh logic in report_subsystem_vectors: the top eigenvector is the subsystem's
    dominant pure direction (unit / gauge-free); align = |<top_vec, 1/sqrt(d)>| in [0,1] is how
    close it sits to the unique G-invariant (trivial-irrep) all-ones direction.
    """
    ones = np.ones(d) / np.sqrt(d)
    evals, evecs = np.linalg.eigh(rho)         # ascending; evecs[:, :, k] is k-th eigvec
    top_eval = evals[:, -1]                     # (n,)
    top_vec = evecs[:, :, -1]                   # (n, d), unit norm
    align = np.abs(top_vec @ ones)             # (n,) in [0, 1]
    return align, top_eval


def pick_digit_image(base, rng, slot, cls):
    """Return one [1,28,28] image of class `cls` for the given digit slot (1 or 2).

    DDMNIST pair labels encode digit-1 in the tens place, digit-2 in the units place. We pick
    a random pair whose chosen-slot digit matches `cls` and return that slot's image. The
    channel dim is kept (as in the single-pair script) so the clean transforms see [C,H,W].
    """
    digit_of = (base.labels // 10) if slot == 1 else (base.labels % 10)
    cand = (digit_of == cls).nonzero(as_tuple=True)[0].tolist()
    if not cand:
        raise ValueError(f"No pair with digit-{slot} == {cls}.")
    pair = base[rng.choice(cand)]
    return pair[0] if slot == 1 else pair[1]  # [1,28,28]


def build_samples(base, cfg, variation, sweep_digit, reps, rng):
    """Build the full sample grid for one (variation, sweep_digit).

    Returns combined images (N,1,56,56) and parallel metadata lists fixed_val, swept_class,
    group_code, rep -- one entry per sample. The latent factor structure is always
    outer (x) digit1 (x) digit2; sweep_digit only flips which slot is fixed vs swept.
    """
    fixed_digit = 1 if sweep_digit == 2 else 2
    transform = cfg['transform']
    n_sym = cfg['n_sym']

    images, fixed_val, swept_class, group_code, rep_idx = [], [], [], [], []
    for rep in range(reps):
        for fv in range(10):                       # fixed digit class
            fixed_img = pick_digit_image(base, rng, fixed_digit, fv)
            for c in range(10):                    # swept digit class
                swept_img = pick_digit_image(base, rng, sweep_digit, c)
                codes = range(n_sym) if variation == 'symmetry' else [0]
                for code in codes:
                    sw = transform(swept_img, code) if variation == 'symmetry' else swept_img
                    # _combine_images takes [28,28] tiles (first=digit1, second=digit2)
                    if sweep_digit == 2:
                        x = base._combine_images(fixed_img[0], sw[0])
                    else:
                        x = base._combine_images(sw[0], fixed_img[0])
                    images.append(x)
                    fixed_val.append(fv)
                    swept_class.append(c)
                    group_code.append(code)
                    rep_idx.append(rep)
    X = torch.stack(images, dim=0)                 # (N,1,56,56)
    meta = dict(fixed_val=np.array(fixed_val), swept_class=np.array(swept_class),
                group_code=np.array(group_code), rep=np.array(rep_idx))
    return X, meta


def encode(model, X, layer_id, d, device, batch_size=256):
    """Forward X through the model in batches; return the unit 64-dim states, latents, preds."""
    states_all, latents_all, pred_all = [], [], []
    with torch.no_grad():
        for i in range(0, X.shape[0], batch_size):
            xb = X[i:i + batch_size].to(device)
            logits, latents = model.model(xb, [layer_id])
            Z = latents[0].cpu()
            block = Z[:, :d].float()
            unit = block / block.norm(dim=1, keepdim=True).clamp_min(1e-12)
            states_all.append(unit)
            latents_all.append(Z)
            pred_all.append(logits.argmax(dim=1).cpu())
    return torch.cat(states_all), torch.cat(latents_all), torch.cat(pred_all).numpy()


def run_combo(model, base, cfg, group, variation, sweep_digit, dims, d, names, layer_id,
              reps, rng, device, out_prefix):
    """Compute and save the NPZ for one (variation, sweep_digit)."""
    X, meta = build_samples(base, cfg, variation, sweep_digit, reps, rng)
    n = X.shape[0]
    n_sym = cfg['n_sym']
    expected = reps * 10 * 10 * (n_sym if variation == 'symmetry' else 1)
    print(f"\n[{group}] variation={variation} sweep_digit={sweep_digit}: {n} samples "
          f"(expected {expected}; 10 fixed x 10 swept"
          + (f" x {n_sym} transforms" if variation == 'symmetry' else "")
          + (f" x {reps} reps" if reps > 1 else "") + ")")

    states, Z, pred = encode(model, X, layer_id, d, device)
    emb_norm = Z[:, :d].float().norm(dim=1).numpy()
    full_norm = Z.float().norm(dim=1).numpy()

    if cfg['decomposition'] == 'bipartite':
        rhos, entropy = bipartite_marginals(states, dims[0], dims[1])
    else:
        rhos, entropy = tripartite_marginals(states, dims[0], dims[1], dims[2])

    dim_by_name = dict(zip(names, dims))
    payload = dict(
        group=group,
        decomposition=cfg['decomposition'],
        subsystem_names=np.array(names),
        subsystem_dims=np.array(dims),
        variation=variation,
        sweep_digit=sweep_digit,
        n_sym=n_sym,
        pred_class=pred,
        pred_tens=pred // 10,
        pred_unit=pred % 10,
        emb_norm=emb_norm,
        full_norm=full_norm,
        **meta,
    )
    payload.update({(f'entropy_{k}' if not k.startswith('entropy') else k): np.asarray(v)
                    for k, v in entropy.items()})
    for name in names:
        rho = rhos[name]
        align, topeig = subsystem_top(rho, int(dim_by_name[name]))
        payload[f'rho_{name}'] = rho
        payload[f'align_{name}'] = align
        payload[f'topeig_{name}'] = topeig

    out_path = f"{out_prefix}_{variation}_sweep{sweep_digit}.npz"
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(out_path, **payload)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ckpt', type=str, required=True, help='Path to the checkpoint (.ckpt).')
    parser.add_argument('--out_prefix', type=str, default=None,
                        help='Output path prefix; writes <prefix>_<variation>_sweep<d>.npz. '
                             'Defaults to results/<group>/robust_T<lambda_t>_E<lambda_e>')
    parser.add_argument('--group', type=str, default=None, choices=list(GROUP_CONFIG.keys()),
                        help="Group / decomposition. If omitted, taken from the checkpoint's "
                             "hparams.group. Must match the training group.")
    parser.add_argument('--variations', type=str, default='symmetry,digit',
                        help="Comma list of variations to compute: symmetry and/or digit.")
    parser.add_argument('--sweep_digits', type=str, default='1,2',
                        help="Comma list of which digit to sweep (1 and/or 2); the other is fixed.")
    parser.add_argument('--reps', type=int, default=1,
                        help="Instances of each fixed-digit image to draw (1 = one PCA per fixed "
                             "digit; >1 averages out instance noise).")
    parser.add_argument('--layer_id', type=int, default=12, help='DDMNISTCNN layer to extract latents from.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for instance selection.')
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')

    variations = [v.strip() for v in args.variations.split(',') if v.strip()]
    sweep_digits = [int(s) for s in args.sweep_digits.split(',') if s.strip()]
    bad = [v for v in variations if v not in ('symmetry', 'digit')]
    if bad:
        raise ValueError(f"Unknown variation(s) {bad}; choose from symmetry, digit.")
    if any(s not in (1, 2) for s in sweep_digits):
        raise ValueError("--sweep_digits must be among 1,2.")

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
    d = int(np.prod(dims))
    assert d == 64, f"Expected a 64-dim latent block for {group}, got {d}."
    names = cfg['names']
    if args.group is None:
        print(f"--group not given; using checkpoint group: {group}")

    # ---- output prefix ----
    if args.out_prefix is not None:
        out_prefix = args.out_prefix
    else:
        lt, le = model.hparams.get('lambda_t'), model.hparams.get('lambda_e')
        out_prefix = os.path.join('results', group, f"robust_T{lt}_E{le}")

    # ---- data ----
    dm = cfg['datamodule'](batch_size=8)
    dm.setup(stage='test')
    base = dm.test_dataset.data  # raw [1,28,28] digits + label

    # One RNG drives all instance selection; reseeded per combo so combos are independent
    # of the order/contents of the variations/sweep lists.
    for variation in variations:
        for sweep_digit in sweep_digits:
            rng = random.Random(args.seed)
            run_combo(model, base, cfg, group, variation, sweep_digit, dims, d, names,
                      args.layer_id, args.reps, rng, device, out_prefix)


if __name__ == "__main__":
    main()
