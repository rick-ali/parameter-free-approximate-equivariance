import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.GxGRegularFunctorModel import GxGRegularFunctor
from datasets.C4xC4DDMNIST_dataset import C4xC4DDMNISTDataModule
from datasets.D4xD4DDMNIST_dataset import D4xD4DDMNISTDataModule
from datasets.D1xD1DDMNIST_dataset import D1xD1DDMNISTDataModule
from utils.entanglement import TripartiteEntanglement, Entanglement

DATASET_TO_DATAMODULE = {
    'ddmnist_c4': C4xC4DDMNISTDataModule,
    'ddmnist_d4': D4xD4DDMNISTDataModule,
    'ddmnist_d1': D1xD1DDMNISTDataModule,
}
BATCH_SIZE = 256

# (dim_a, dim_b, dim_c) tripartite split per dataset: outer x G1 x G2.
TRIPARTITE_DIMS = {
    'ddmnist_c4': (4, 4, 4),    # 64 = 4 * 4 * 4
    'ddmnist_d1': (16, 2, 2),   # 64 = 16 * 2 * 2
}
BIPARTITE_DIMS = {
    'ddmnist_d4': (8, 8),       # 64 = 8 * 8
}


def seed_all():
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)


@torch.no_grad()
def test_accuracy_and_entanglement(model, dataloader, device, tri_dims, bi_dims):
    """Return (accuracy, ent_avg) where ent_avg is the mean across the three
    tripartite cuts (A:BC, B:AC, C:AB), computed per-sample on the full
    concatenated latent set and then averaged across samples. This matches the
    aggregation in compute_entanglement.py exactly."""
    model.eval()
    correct = total = 0
    all_latents = []
    for (x1, y1), *_ in dataloader:
        x1 = x1.to(device)
        logits, latents = model.model(x1, [12])
        latent = torch.cat(latents, dim=-1).cpu()
        all_latents.append(latent)

        pred = torch.argmax(logits, dim=1).cpu()
        correct += (pred == y1).sum().item()
        total += y1.numel()

    acc = correct / total

    latents_full = torch.cat(all_latents, dim=0)
    # bipartite:
    if bi_dims is not None:
        dim_a, dim_b = bi_dims
        tensor_dims = dim_a * dim_b
        tensor_latents = latents_full[:, :tensor_dims]
        norm = torch.linalg.vector_norm(tensor_latents, dim=1, keepdim=True)
        norm_tensor_latents = tensor_latents / norm
        ent = Entanglement(norm_tensor_latents, dim_a, dim_b).compute(normalize=True)
        vne = ent.get("entanglement_a", None)
        ent_avg = vne.mean().item() if vne is not None else 0.0
        return acc, ent_avg
    else:
        # tripartite:
        dim_a, dim_b, dim_c = tri_dims
        tensor_dims = dim_a * dim_b * dim_c
        tensor_latents = latents_full[:, :tensor_dims]
        norm = torch.linalg.vector_norm(tensor_latents, dim=1, keepdim=True)
        norm_tensor_latents = tensor_latents / norm
        tri = TripartiteEntanglement(norm_tensor_latents, dim_a, dim_b, dim_c).compute(normalize=True)
        ent_avg = (
            tri["entanglement_a_bc"].mean().item()
            + tri["entanglement_b_ac"].mean().item()
            + tri["entanglement_c_ab"].mean().item()
        ) / 3.0
        return acc, ent_avg



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_paths', type=str, nargs='+', required=True,
                        help='Full paths to one or more trained model checkpoint (.ckpt) files.')
    parser.add_argument('--dataset', type=str, default='ddmnist_d4', choices=DATASET_TO_DATAMODULE.keys(),
                        help='Dataset to use for extracting test accuracy.')
    args = parser.parse_args()

    if args.dataset not in TRIPARTITE_DIMS:
        raise NotImplementedError(
            f"Tripartite entanglement (used for the ent_avg column) is not configured "
            f"for dataset {args.dataset}. Supported: {sorted(TRIPARTITE_DIMS)}"
        )
    tri_dims = TRIPARTITE_DIMS.get(args.dataset, None)
    bi_dims = BIPARTITE_DIMS.get(args.dataset, None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # plain and augmented test loaders
    DataModuleClass = DATASET_TO_DATAMODULE[args.dataset]
    seed_all()
    dm_plain = DataModuleClass(BATCH_SIZE, augment_test=False)
    dm_plain.setup()
    dm_aug = DataModuleClass(BATCH_SIZE, augment_test=True)
    dm_aug.setup()
    plain_loader = DataLoader(dm_plain.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    aug_loader = DataLoader(dm_aug.test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    header = (
        f"{'lambda_t':>10} {'lambda_e':>10} "
        f"{'test_acc':>10} {'aug_test_acc':>14} "
        f"{'ent_avg_test':>14} {'ent_avg_aug':>14}"
    )
    print(header)
    print("-" * len(header))
    for path in args.checkpoint_paths:
        seed_all()
        model = GxGRegularFunctor.load_from_checkpoint(path, map_location=device).to(device)
        model.model.get_latent = True

        seed_all()
        plain_acc, plain_ent = test_accuracy_and_entanglement(model, plain_loader, device, tri_dims=tri_dims, bi_dims=bi_dims)
        seed_all()
        aug_acc, aug_ent = test_accuracy_and_entanglement(model, aug_loader, device, tri_dims=tri_dims, bi_dims=bi_dims)

        lt = model.hparams['lambda_t']
        le = model.hparams['lambda_e']
        print(
            f"{lt:>10} {le:>10} "
            f"{plain_acc:>10.4f} {aug_acc:>14.4f} "
            f"{plain_ent:>14.4f} {aug_ent:>14.4f}"
        )
