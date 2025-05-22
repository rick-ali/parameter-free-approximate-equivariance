import torch
import random
from typing import List

def initialise_W_random(dim, device='cuda'):
    W = torch.randn(dim, dim, device=device)
    return W

def initialise_W_orthogonal(dim, noise_level=0, device='cuda'):
    W = torch.nn.init.orthogonal_(torch.empty((dim, dim), device=device))
    W += noise_level * torch.randn((dim, dim), device=device)
    return W

def initialise_W_random_roots_of_unity(dim, N, noise_level=0, device='cuda'):
    roots = [torch.exp(torch.tensor(2 * torch.pi * 1j * i / N, device=device)) for i in range(N)]
    W = torch.eye(dim, device=device)
    for i in range(dim):
        W[i, i] = random.choice(roots)

    W += noise_level * torch.randn((dim, dim), device=device)
    return W


def initialise_W_real_Cn_irreps(irrep_dims: List[int], N: int, change_basis: bool = False, device='cuda'):
    """
    Constructs a real-valued matrix W with the given multiplicities of the Nth roots of unity.
    Remember that if W is real valued then conjugate pairs of complex roots must come in pairs.
    
    Args:
        irrep_dims (list of int): List where irrep_dims[i] specifies the multiplicity of the i-th Nth root of unity.
        N (int): Order of the roots of unity.
        
    Returns:
        torch.Tensor: The constructed real-valued matrix W.
    """
    # Compute Nth roots of unity
    roots_of_unity = [torch.exp(torch.tensor(2j * torch.pi * k / N, device=device)) for k in range(N)]
    
    # Create real-valued block matrix
    blocks = []
    
    for k, dim in enumerate(irrep_dims):
        root = roots_of_unity[k]
        
        if torch.isclose(root.imag, torch.tensor(0.0, device=device)):  # If root is real (e.g., ±1)
            blocks.extend([torch.tensor([[root.real]], device=device)] * dim)
        else:  # If root is complex, create 2x2 rotation blocks
            theta = torch.tensor(2 * torch.pi * k / N, device=device)
            rotation_block = torch.tensor([
                [torch.cos(theta), -torch.sin(theta)],
                [torch.sin(theta), torch.cos(theta)],
            ], device=device)
            # Each complex root needs an even number of dimensions (because they come in conjugate pairs)
            for _ in range(dim // 2):  
                blocks.append(rotation_block)
    
    # Construct the final block-diagonal matrix
    W = torch.block_diag(*blocks) if blocks else torch.tensor([], device=device)  # Handle empty case

    # Set all values less than 1e-5 to zero
    W[torch.abs(W) < 1e-5] = 0.0
    
    if change_basis:
        # initialise orthogonal matrix P
        P = torch.nn.init.orthogonal_(torch.empty((W.shape[0], W.shape[0])), device=device)
        W = P @ W @ P.T
    
    return W
    

