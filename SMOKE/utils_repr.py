import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Pre-define the 4×4 regular-representation matrices for C4:
#   W(0) = I
#   W(1) = rotation by 90° CCW
#   W(2) = rotation by 180°
#   W(3) = rotation by 270° CCW
_W = [
    torch.tensor([
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
    ], dtype=torch.float32, device=device, requires_grad=False),
    torch.tensor([
        [0,0,0,1],
        [1,0,0,0],
        [0,1,0,0],
        [0,0,1,0],
    ], dtype=torch.float32, device=device, requires_grad=False),
    torch.tensor([
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
        [0,1,0,0],
    ], dtype=torch.float32, device=device, requires_grad=False),
    torch.tensor([
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [1,0,0,0],
    ], dtype=torch.float32, device=device, requires_grad=False),
]

def W(g: int, d: int) -> torch.Tensor:
    """
    Return the 4×4 regular-representation matrix W(g) for the cyclic group C4.
    
    Args:
        g: group element in {0,1,2,3}, corresponding to r^g.
        d: latent dimension of the input data.

    Returns:
        A 4×4 torch.float32 tensor (permutation matrix).
    """
    base = _W[g]                # B x B base block
    B = base.size(-1)
    n_blocks = d // B
    rem = d - n_blocks * B

    # build entirely on GPU
    eye_nb = torch.eye(n_blocks, device=device, requires_grad=False)
    W_block = torch.kron(eye_nb, base)  # (n_blocks*B) x (n_blocks*B)

    if rem > 0:
        I_rem = torch.eye(rem, device=device, requires_grad=False)
        W_full = torch.block_diag(W_block, I_rem)
    else:
        W_full = W_block

    return W_full



class D4RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D4RegularRepresentation, self).__init__()
        # Precompute the 8 permutation matrices
        self.matrices = self.get_d4_regular_representation(device, dtype)

    def group_mult(self, i: int, j: int) -> int:
        """
        Multiply elements of D4 by their indices 0..7:
          0..3 : r^0, r^1, r^2, r^3
          4..7 : r^0 s, r^1 s, r^2 s, r^3 s
        Returns index of product g_i * g_j.
        """
        # rotation * rotation
        if i < 4 and j < 4:
            return (i + j) % 4
        # rotation * reflection
        elif i < 4 and j >= 4:
            b = j - 4
            return 4 + ((i + b) % 4)
        # reflection * rotation
        elif i >= 4 and j < 4:
            a = i - 4
            return 4 + ((a - j) % 4)
        # reflection * reflection
        else:
            a = i - 4
            b = j - 4
            return (a - b) % 4

    def inverse(self, i: int) -> int:
        """
        Inverse in D4:
         - rotations (0..3): inverse is (-i) mod 4
         - reflections (4..7): each is its own inverse
        """
        if i < 4:
            return (-i) % 4
        else:
            return i

    def get_d4_regular_representation(self, device, dtype):
        rep = {}
        n = 8  # order of D4
        for i in range(n):
            M = torch.zeros(n, n, device=device, dtype=dtype, requires_grad=False)
            for j in range(n):
                k = self.group_mult(i, j)
                M[k, j] = 1
            rep[i] = M
        return rep

    def forward(self, x: int) -> torch.Tensor:
        """
        Given an index x (0..7), returns the corresponding 8×8 permutation matrix.
        """
        return self.matrices[x]

    def mapping(self) -> dict:
        """
        Human‐readable names for the 8 group elements:
        0:'1', 1:'r', 2:'r^2', 3:'r^3',
        4:'s', 5:'r s', 6:'r^2 s', 7:'r^3 s'
        """
        return {
            0: '1',
            1: 'r',
            2: 'r^2',
            3: 'r^3',
            4: 's',
            5: 'r s',
            6: 'r^2 s',
            7: 'r^3 s',
        }