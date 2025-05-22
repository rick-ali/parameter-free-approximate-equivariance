import torch
import torch.nn as nn
from typing import Tuple, Dict


class D1RegularRepresentation(nn.Module):
    """
    Regular representation of D1 (order 2):
      - index 0: identity e
      - index 1: reflection s (horizontal flip)
    """
    def __init__(self, device, dtype=torch.float32):
        super(D1RegularRepresentation, self).__init__()
        self.matrices = self._build_representation(device, dtype)

    def group_mult(self, i: int, j: int) -> int:
        """Multiply elements of D1: s^2 = e, abelian."""
        return (i + j) % 2

    def inverse(self, i: int) -> int:
        """Inverse in D1: each element is its own inverse."""
        return i  # since s^2 = e, and e inverse = e

    def _build_representation(self, device, dtype):
        rep = {}
        n = 2
        for i in range(n):
            M = torch.zeros(n, n, device=device, dtype=dtype, requires_grad=False)
            for j in range(n):
                k = self.group_mult(i, j)
                M[k, j] = 1
            rep[i] = M
        return rep

    def forward(self, idx: int) -> torch.Tensor:
        """Return the 2x2 permutation matrix for element idx."""
        return self.matrices[idx]

    def mapping(self) -> dict:
        """Human-readable labels."""
        return {0: 'e', 1: 's (horizontal)'}


class D1xD1RegularRepresentation(nn.Module):
    """
    Regular representation of D1 x D1 (order 4): elements (a,b), a,b in D1,
    flat index = a*2 + b in 0..3, matrices are 4x4.
    """
    def __init__(self, device, dtype=torch.float32):
        super(D1xD1RegularRepresentation, self).__init__()
        self.d1 = D1RegularRepresentation(device, dtype)
        self.matrices = self._build_representation(device, dtype)

    def group_mult(self, idx1: int, idx2: int) -> int:
        a1, b1 = divmod(idx1, 2)
        a2, b2 = divmod(idx2, 2)
        a3 = self.d1.group_mult(a1, a2)
        b3 = self.d1.group_mult(b1, b2)
        return a3 * 2 + b3

    def inverse(self, idx: int) -> int:
        a, b = divmod(idx, 2)
        ia = self.d1.inverse(a)
        ib = self.d1.inverse(b)
        return ia * 2 + ib

    def _build_representation(self, device, dtype):
        rep = {}
        for a in range(2):
            for b in range(2):
                idx = a * 2 + b
                P_a = self.d1.matrices[a]
                P_b = self.d1.matrices[b]
                rep[idx] = torch.kron(P_a, P_b)
        return rep

    def forward(self, idx: int) -> torch.Tensor:
        return self.matrices[idx]

    def mapping(self) -> dict:
        base = self.d1.mapping()
        return {i: f"({base[i//2]}, {base[i%2]})" for i in range(4)}


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


class D4xD4RegularRepresentation(nn.Module):
    """
    Regular representation of the direct product D4 x D4.
    Elements are pairs (g, h), each in D4 (indices 0..7),
    flattened to a single index idx = g*8 + h in 0..63.
    The corresponding 64x64 permutation matrix is P_g tensor P_h.
    """
    def __init__(self, device, dtype=torch.float32):
        super(D4xD4RegularRepresentation, self).__init__()
        # Underlying D4 regular representation
        self.d4 = D4RegularRepresentation(device, dtype)
        # Precompute all 64 permutation matrices
        self.matrices = self._build_representation(device, dtype)
        self.order = 64

    def group_mult(self, idx1: int, idx2: int) -> int:
        """
        Multiply two elements of D4xD4 given their flat indices.
        Decompose each into (g1,h1) and (g2,h2), multiply componentwise,
        then re-flatten.
        """
        g1, h1 = divmod(idx1, 8)
        g2, h2 = divmod(idx2, 8)
        g3 = self.d4.group_mult(g1, g2)
        h3 = self.d4.group_mult(h1, h2)
        return g3 * 8 + h3

    def inverse(self, idx: int) -> int:
        """
        Inverse of (g, h) is (g^{-1}, h^{-1}).
        """
        g, h = divmod(idx, 8)
        inv_g = self.d4.inverse(g)
        inv_h = self.d4.inverse(h)
        return inv_g * 8 + inv_h

    def _build_representation(self, device, dtype):
        rep = {}
        for g in range(8):
            for h in range(8):
                idx = g * 8 + h
                P_g = self.d4.matrices[g]
                P_h = self.d4.matrices[h]
                # Tensor/Kronecker product gives 64x64 matrix
                rep[idx] = torch.kron(P_g, P_h)
        return rep

    def forward(self, idx: int) -> torch.Tensor:
        """
        Given a flat index (0..63), return its 64x64 permutation matrix.
        """
        return self.matrices[idx]

    def mapping(self) -> dict:
        """
        Returns a mapping from flat index to human-readable pair.
        E.g. 0: ('1','1'), 1: ('r','1'), ..., 63: ('r^3 s','r^3 s')
        """
        base_map = self.d4.mapping()
        return {g*8 + h: f"({base_map[g]}, {base_map[h]})"
                for g in range(8) for h in range(8)}


class D8RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D8RegularRepresentation, self).__init__()

        self.matrices = self.get_d8_regular_representation(device, dtype)

    # -- Group Multiplication Function for D_{8} --
    # Ordering: 
    #   Indices 0..7       : r^0, r^1, ..., r^7
    #   Indices 8..15      : r^0 s, r^1 s, ..., r^7 s
    def group_mult(self, i, j):
        """
        Multiplies two elements of D_{8} given by their indices.
        Returns the index of the product.
        """
        # Case 1: Both rotations
        if i < 8 and j < 8:
            return (i + j) % 8
        # Case 2: Rotation * Reflection
        elif i < 8 and j >= 8:
            b = j - 8
            return 8 + ((i + b) % 8)
        # Case 3: Reflection * Rotation
        elif i >= 8 and j < 8:
            a = i - 8
            return 8 + ((a - j) % 8)
        # Case 4: Reflection * Reflection
        else:
            a = i - 8
            b = j - 8
            return (a - b) % 8
        
    def d8_inverse(self, i: int) -> int:
        """
        Returns the inverse of a D8 element given its index.
        - For rotations (i in 0..7): inverse is (8-i) mod 8.
        - For reflections (i in 8..15): inverse is itself.
        """
        if i < 8:
            return (-i) % 8  # same as (8 - i) mod 8, since 0's inverse is 0.
        else:
            return i

    # -- Regular Representation --
    # For each group element g (identified by index i), the regular representation
    # is given by a 16x16 permutation matrix P_i acting on the group algebra.
    # The (k, j) entry of P_i is 1 if g * (element with index j) equals 
    # the element with index k, and 0 otherwise.
    def get_d8_regular_representation(self, device, dtype):
        rep = {}
        n = 16  # Order of D8
        for i in range(n):
            M = torch.zeros(n, n, device=device, dtype=dtype, requires_grad=False)
            # For each basis element j, the product g * (element j) is element with index k:
            for j in range(n):
                k = self.group_mult(i, j)
                M[k, j] = 1  # Place a 1 at row k, column j.
            rep[i] = M
        return rep

    def forward(self, x):
        """
        Given a number x (0 to 15), return the corresponding 16x16 matrix.
        """
        return self.matrices[x]
    
    def mapping(self) -> dict:
        #   Indices 0..7       : r^0, r^1, ..., r^7
        #   Indices 8..15      : r^0 s, r^1 s, ..., r^7 s
        mapping = {
            0:'1', 1:'r', 2:'rr', 3:'rrr', 
            4:'rrrr', 5:'rrrrr', 6:'rrrrrr', 7:'rrrrrrr',
            8:'s', 9:'r s', 10:'r^2 s', 11:'r^3 s',
            12:'r^4 s', 13:'r^5 s', 14:'r^6 s', 15:'r^7 s'
        }
        return mapping


class C4xC4RegularRepresentation(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        """
        Regular representation of C4 x C4 of order 16.
        Elements are pairs (i,j) with i,j in 0..3, indexed as 4*i + j.
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.dtype  = dtype
        self.matrices = self._build_representation()

    def _index(self, i: int, j: int) -> int:
        """Map pair (i,j) to index in 0..15."""
        return 4 * (i % 4) + (j % 4)

    def group_mult(self, a: int, b: int) -> int:
        """
        Multiply two elements of C4 x C4.
        a, b in 0..15.  Return index of a*b.
        """
        i1, j1 = divmod(a, 4)
        i2, j2 = divmod(b, 4)
        # component-wise addition mod 4
        return self._index(i1 + i2, j1 + j2)

    def inverse(self, a: int) -> int:
        """
        Inverse in C4 x C4 is negation mod 4 in each component.
        """
        i, j = divmod(a, 4)
        return self._index(-i, -j)

    def _build_representation(self):
        """
        Builds a dict of 16 permutation matrices M[a] of size 16×16,
        where M[a] realizes left‐multiplication by the element a.
        """
        rep = {}
        n = 16
        for a in range(n):
            M = torch.zeros(n, n, device=self.device, dtype=self.dtype, requires_grad=False)
            for b in range(n):
                c = self.group_mult(a, b)
                M[c, b] = 1
            rep[a] = M
        return rep

    def forward(self, idx: int) -> torch.Tensor:
        """
        Given an element index in 0..15, returns the corresponding 16×16 matrix.
        """
        return self.matrices[idx]

    def mapping(self) -> dict:
        """
        Returns a human‐readable map from index to group element.
        We'll write elements as r^i s^j.
        """
        mp = {}
        for i in range(4):
            for j in range(4):
                idx = self._index(i, j)
                mp[idx] = f"r^{i} s^{j}"
        return mp


class ProductRegularRepresentation(nn.Module):
    """
    Constructs the regular representation of the direct product G x H,
    given any two group regular-representation modules G and H.

    You must provide instances g_rep and h_rep that implement:
      - g_rep.matrices: Dict[int, Tensor] of size |G| x |G|
      - g_rep.group_mult(i,j) -> int
      - g_rep.inverse(i) -> int
      - g_rep.forward(i) -> Tensor
      - g_rep.mapping() -> Dict[int, str]
    (Similarly for h_rep.)

    The product group has order n = |G| * |H|, flat indices 0..n-1,
    where idx = i*|H| + j corresponds to (g_i, h_j).
    Its regular representation matrices are P_{i,j} = P^G_i otimes P^H_j.
    """
    def __init__(self,
                 g_rep: nn.Module,
                 h_rep: nn.Module,
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        # Underlying group reps
        self.g = g_rep
        self.h = h_rep
        # Sizes
        self.n_g = len(self.g.matrices)
        self.n_h = len(self.h.matrices)
        self.order = self.n_g * self.n_h
        # Build product matrices
        self.matrices = self._build_product(device, dtype)

    def _build_product(self, device: torch.device, dtype: torch.dtype) -> Dict[int, torch.Tensor]:
        prod = {}
        for i in range(self.n_g):
            P_i = self.g.matrices[i]
            for j in range(self.n_h):
                P_j = self.h.matrices[j]
                idx = i * self.n_h + j
                prod[idx] = torch.kron(P_i.to(device, dtype), P_j.to(device, dtype))
        return prod

    def group_mult(self, idx1: int, idx2: int) -> int:
        """
        Flat-index multiplication in GxH:
          (i1,j1)*(i2,j2) = (i1*i2, j1*j2)
        """
        i1, j1 = divmod(idx1, self.n_h)
        i2, j2 = divmod(idx2, self.n_h)
        i3 = self.g.group_mult(i1, i2)
        j3 = self.h.group_mult(j1, j2)
        return i3 * self.n_h + j3

    def inverse(self, idx: int) -> int:
        """Inverse in GxH: (i,j)^{-1} = (i^{-1}, j^{-1})."""
        i, j = divmod(idx, self.n_h)
        inv_i = self.g.inverse(i)
        inv_j = self.h.inverse(j)
        return inv_i * self.n_h + inv_j

    def forward(self, idx: int) -> torch.Tensor:
        """Return the regular-representation matrix for element idx."""
        return self.matrices[idx]

    def mapping(self) -> Dict[int, str]:
        """Human-readable mapping: flat idx -> (g_label, h_label)."""
        g_map = self.g.mapping()
        h_map = self.h.mapping()
        return {i * self.n_h + j: f"({g_map[i]}, {h_map[j]})"
                for i in range(self.n_g) for j in range(self.n_h)}
    


class OctahedralRegularRepresentation(nn.Module):
    """
    Regular representation of the rotational octahedral group (symmetries of the cube) of order 24.
    Elements are 3x3 rotation matrices acting on the cube's vertices.
    We enumerate all 24 unique rotations and build the 24x24 permutation matrices.
    """
    def __init__(self, device=torch.device('cpu'), dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Generate the 24 rotation matrices
        self.rotations = self._generate_rotations()
        self.order = len(self.rotations)
        # Build index lookup: map from matrix tuple to index
        self._matrix_to_index = {
            self._key(R): idx
            for idx, R in enumerate(self.rotations)
        }
        # Build regular representation matrices
        self.matrices = self._build_regular()

    def _round_matrix(self, R: torch.Tensor) -> torch.Tensor:
        """Round matrix entries to 6 decimal places to stabilize comparisons."""
        return (R * 1e6).round() / 1e6

    def _key(self, R: torch.Tensor) -> tuple:
        """Create a hashable key from a rounded matrix."""
        Rr = self._round_matrix(R)
        return tuple(Rr.flatten().tolist())

    def _generate_rotations(self):
        """
        Generate all 24 proper rotation matrices of the cube using axis-angle.
        """
        device = self.device
        dtype = self.dtype

        def axis_angle(axis, angle):
            # angle might be Python float or tensor
            ang = torch.tensor(angle, dtype=dtype, device=device)
            axis_t = torch.tensor(axis, dtype=dtype, device=device)
            axis_t = axis_t / torch.norm(axis_t)
            x, y, z = axis_t
            c = torch.cos(ang)
            s = torch.sin(ang)
            C = 1 - c
            R = torch.stack([
                torch.stack([c + x*x*C,     x*y*C - z*s, x*z*C + y*s]),
                torch.stack([y*x*C + z*s, c + y*y*C,     y*z*C - x*s]),
                torch.stack([z*x*C - y*s, z*y*C + x*s, c + z*z*C    ])
            ])
            return self._round_matrix(R)

        angles = {
            (1,0,0): [0.0, torch.pi/2, torch.pi, 3*torch.pi/2],
            (0,1,0): [0.0, torch.pi/2, torch.pi, 3*torch.pi/2],
            (0,0,1): [0.0, torch.pi/2, torch.pi, 3*torch.pi/2],
            (1,1,1): [0.0, 2*torch.pi/3, 4*torch.pi/3],
            (1,1,-1): [0.0, 2*torch.pi/3, 4*torch.pi/3],
            (1,-1,1): [0.0, 2*torch.pi/3, 4*torch.pi/3],
            (-1,1,1): [0.0, 2*torch.pi/3, 4*torch.pi/3],
            (1,0,1): [0.0, torch.pi],
            (1,0,-1): [0.0, torch.pi],
            (0,1,1): [0.0, torch.pi],
            (0,1,-1): [0.0, torch.pi],
            (1,1,0): [0.0, torch.pi],
            (1,-1,0): [0.0, torch.pi],
        }
        mats = []
        for axis, ang_list in angles.items():
            for ang in ang_list:
                R = axis_angle(axis, ang)
                mats.append(R)
        # Deduplicate preserving order
        unique = []
        seen = set()
        for R in mats:
            key = self._key(R)
            if key not in seen:
                seen.add(key)
                unique.append(R)
        if len(unique) != 24:
            raise ValueError(f"Expected 24 rotations, got {len(unique)}")
        return unique

    def group_mult(self, i, j):
        """Multiply two elements by composing their rotation matrices."""
        R = self.rotations[i].matmul(self.rotations[j])
        return self._matrix_to_index[self._key(R)]

    def inverse(self, i):
        """Inverse is the transpose of the rotation matrix."""
        R_t = self.rotations[i].t()
        return self._matrix_to_index[self._key(R_t)]

    def _build_regular(self):
        rep = {}
        for i in range(self.order):
            M = torch.zeros(self.order, self.order, device=self.device, dtype=self.dtype)
            for j in range(self.order):
                k = self.group_mult(i, j)
                M[k, j] = 1
            rep[i] = M
        return rep

    def forward(self, idx):
        """Return the 24×24 permutation matrix for element idx."""
        return self.matrices[idx]

    def mapping(self):
        """Return mapping idx -> rotation matrix."""
        return {idx: R for idx, R in enumerate(self.rotations)}

