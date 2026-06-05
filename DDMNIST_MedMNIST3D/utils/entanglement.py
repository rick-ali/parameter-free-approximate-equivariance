"""

Code copied from qbjection/quantum-entangled-representation-learning

"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch

class Entanglement():
    """
    Compute entanglement entropy for quantum state vectors.
    
    Supports both single vectors and batched vectors:
    - Single: vector of shape (dim_a * dim_b,)
    - Batched: vectors of shape (batch_size, dim_a * dim_b)
    """
    def __init__(self, vectors, dim_a, dim_b):
        if not torch.is_tensor(vectors):
            vectors = torch.tensor(vectors, dtype=torch.float32)
        
        # Handle both single vector and batch of vectors
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)  # Add batch dimension
            self.is_single = True
        else:
            self.is_single = False
        
        self.vectors = vectors  # Shape: (batch_size, dim)
        self.rho = self.vector_to_density_matrix(vectors)
        self.dim_a = int(dim_a)
        self.dim_b = int(dim_b)

    def compute(self, normalize: bool = False):
        """
        Function to compute the von Neumann entropy.

        Args:
            normalize (bool): If True, normalize the entropy by the maximum possible entropy for given dimensionality of the system, back to the range [0,1].
        """

        rho_a = self.partial_trace_B(self.rho, self.dim_a, self.dim_b) # partial trace over subsystem B
        rho_b = self.partial_trace_A(self.rho, self.dim_a, self.dim_b) # partial trace over subsystem A
        entropy_a = self.von_neumann_entropy(rho_a)
        entropy_b = self.von_neumann_entropy(rho_b)
        
        # If input was a single vector, return scalars instead of 1-element tensors
        if self.is_single:
            entropy_a = entropy_a.squeeze(0)
            entropy_b = entropy_b.squeeze(0)
        
        if normalize:
            max_entropy = torch.log2(torch.tensor(min(self.dim_a, self.dim_b), dtype=torch.float32))
            entropy_a = entropy_a / max_entropy
            entropy_b = entropy_b / max_entropy
        # By the Schmidt decomposition, we expect entropy_a == entropy_b
        return {"entanglement_a": entropy_a, "entanglement_b": entropy_b}  

    def vector_to_density_matrix(self, vectors):
        """
        Convert state vectors to density matrices.
        
        Args:
            vectors: Tensor of shape (batch_size, dim)
        Returns:
            rho: Tensor of shape (batch_size, dim, dim)
        """
        # vectors: (batch_size, dim) -> (batch_size, dim, 1)
        v = vectors.unsqueeze(-1)
        # Outer product: (batch_size, dim, 1) @ (batch_size, 1, dim) -> (batch_size, dim, dim)
        rho = torch.bmm(v, v.conj().transpose(-2, -1))
        return rho

    #TODO perhaps use qutip instead of numpy — which has built-in implementations of ops like partial trace
    #note qutip is not compatible with pytorch tensors.
    #if we want to use entanglement in the loss function we would have to manually implement these functions anyway

    #TODO combine partial trace functions into a single function with a parameter that dictates which variable to trace out
    def partial_trace_B(self, rho: torch.Tensor, dim_a, dim_b):
        """
        Compute the partial trace over subsystem B (batched).
        # this link motivates and explains the procedure well: 
        # https://scicomp.stackexchange.com/a/35102

        Args:
            rho: Density matrix of shape (batch_size, dim, dim).
            dim_a (int): Dimension of subsystem A (not traced over).
            dim_b (int): Dimension of subsystem B (traced over).
        Returns:
            Reduced density matrix of shape (batch_size, dim_b, dim_b).
        """
        batch_size = rho.shape[0]
        rho = rho.reshape(batch_size, dim_a, dim_b, dim_a, dim_b)
        # Trace over subsystem A (indices 1 and 3)
        return torch.einsum('bijik->bjk', rho)

    def partial_trace_A(self, rho: torch.Tensor, dim_a, dim_b):
        """
        Compute the partial trace over subsystem A (batched).
        # this link motivates and explains the procedure well: 
        # https://scicomp.stackexchange.com/a/35102

        Args:
            rho: Density matrix of shape (batch_size, dim, dim).
            dim_a (int): Dimension of subsystem A (traced over).
            dim_b (int): Dimension of subsystem B (not traced over).
        Returns:
            Reduced density matrix of shape (batch_size, dim_a, dim_a).
        """
        batch_size = rho.shape[0]
        rho = rho.reshape(batch_size, dim_a, dim_b, dim_a, dim_b)
        # Trace over subsystem B (indices 2 and 4)
        return torch.einsum('bijkj->bik', rho)

    def von_neumann_entropy(self, rho):
        """
        Compute the von Neumann entropy of density matrices (batched).
        
        Args:
            rho: Density matrices of shape (batch_size, dim, dim)
        Returns:
            Entropy tensor of shape (batch_size,)
        """
        #TODO torch.linalg.eigvalsh NOT IMPLEMENTED FOR MPS BACKEND — RAISE ISSUE
        # eigenvalues shape: (batch_size, dim)
        try:
            eigenvalues = torch.linalg.eigvalsh(rho)
        except torch.linalg.LinAlgError:
            # eigvalsh can fail to converge on rank-deficient PSD matrices,
            # add a tiny diagonal jitter and retry: small effect on entanglement vals
            jitter = 1e-6 * torch.eye(rho.shape[-1], dtype=rho.dtype, device=rho.device)
            eigenvalues = torch.linalg.eigvalsh(rho + jitter)

        # Clamp small/negative eigenvalues to avoid log(0) or log(negative)
        # Use a small positive value instead of filtering, to keep batch dimension consistent
        #this is OK, since mutliplying 1e-10 by log2(1e-10) is around -1e-10
        eigenvalues = torch.clamp(eigenvalues, min=1e-10) 
        
        # Compute entropy for each sample in the batch
        # Sum over eigenvalue dimension, keep batch dimension
        entropy = -torch.sum(eigenvalues * torch.log2(eigenvalues), dim=-1)
        
        return entropy

class TripartiteEntanglement():
    """
    Compute the three bipartite entanglement entropies (A:BC, B:AC, C:AB) of a
    tripartite pure state vector in a space A (x) B (x) C.

    The middle cut B:AC is not directly expressible by the bipartite
    `Entanglement` class because B sits between A and C in the tensor ordering.
    We handle all three cuts uniformly by permuting axes so that the isolated
    subsystem is on one side of the tensor product, then reusing the bipartite
    machinery as-is.

    Supports a single vector of shape (dim_a*dim_b*dim_c,) or a batch of shape
    (batch_size, dim_a*dim_b*dim_c).
    """
    def __init__(self, vectors, dim_a, dim_b, dim_c):
        if not torch.is_tensor(vectors):
            vectors = torch.tensor(vectors, dtype=torch.float32)

        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)
            self.is_single = True
        else:
            self.is_single = False

        self.vectors = vectors
        self.dim_a = int(dim_a)
        self.dim_b = int(dim_b)
        self.dim_c = int(dim_c)

    def _isolated_bipartite_vectors(self, subsystem: str):
        """
        Permute axes so that `subsystem` sits in the first tensor factor and
        the other two are merged into the second. Returns a (batch, dim_iso *
        dim_rest) tensor along with (dim_iso, dim_rest).
        """
        batch = self.vectors.shape[0]
        # Reshape flat vector into (batch, dim_a, dim_b, dim_c).
        reshaped = self.vectors.reshape(batch, self.dim_a, self.dim_b, self.dim_c)

        if subsystem == 'A':
            permuted = reshaped  # (batch, A, B, C)
            dim_iso, dim_rest = self.dim_a, self.dim_b * self.dim_c
        elif subsystem == 'B':
            permuted = reshaped.permute(0, 2, 1, 3)  # (batch, B, A, C)
            dim_iso, dim_rest = self.dim_b, self.dim_a * self.dim_c
        elif subsystem == 'C':
            permuted = reshaped.permute(0, 3, 1, 2)  # (batch, C, A, B)
            dim_iso, dim_rest = self.dim_c, self.dim_a * self.dim_b
        else:
            raise ValueError(f"subsystem must be one of 'A', 'B', 'C'; got {subsystem!r}")

        bipartite_vectors = permuted.reshape(batch, dim_iso * dim_rest)
        return bipartite_vectors, dim_iso, dim_rest

    def _entanglement_of_isolated(self, subsystem: str, normalize: bool):
        bipartite_vectors, dim_iso, dim_rest = self._isolated_bipartite_vectors(subsystem)
        # Build a bipartite Entanglement with the isolated subsystem as A; its
        # `entanglement_a` is then exactly S(rho_iso), i.e. the iso:rest cut.
        ent = Entanglement(bipartite_vectors, dim_iso, dim_rest)
        result = ent.compute(normalize=normalize)
        return result["entanglement_a"]

    def compute(self, normalize: bool = False):
        """
        Returns the three bipartite entanglement entropies as a dict:

            entanglement_a_bc : S(rho_A) for the cut A : BC
            entanglement_b_ac : S(rho_B) for the cut B : AC  (the middle cut)
            entanglement_c_ab : S(rho_C) for the cut C : AB
        """
        e_a = self._entanglement_of_isolated('A', normalize)
        e_b = self._entanglement_of_isolated('B', normalize)
        e_c = self._entanglement_of_isolated('C', normalize)

        if self.is_single:
            e_a = e_a.squeeze(0)
            e_b = e_b.squeeze(0)
            e_c = e_c.squeeze(0)

        return {
            "entanglement_a_bc": e_a,
            "entanglement_b_ac": e_b,
            "entanglement_c_ab": e_c,
        }


def get_average_tripartite_entanglement(num_samples, dim_a, dim_b, dim_c, normalize: bool = False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    total_a_bc = 0
    total_b_ac = 0
    total_c_ab = 0

    for _ in range(num_samples):
        random_unit_vector = torch.randn(int(dim_a * dim_b * dim_c), generator=generator)
        random_unit_vector /= torch.norm(random_unit_vector)

        entanglement = TripartiteEntanglement(random_unit_vector, dim_a, dim_b, dim_c)
        vne = entanglement.compute(normalize=normalize)
        total_a_bc += vne.get("entanglement_a_bc")
        total_b_ac += vne.get("entanglement_b_ac")
        total_c_ab += vne.get("entanglement_c_ab")

    return {
        "avg_entropy_a_bc": total_a_bc / num_samples,
        "avg_entropy_b_ac": total_b_ac / num_samples,
        "avg_entropy_c_ab": total_c_ab / num_samples,
    }


def get_normalized_average_tripartite_entanglement(num_samples, dim_a, dim_b, dim_c):
    avg = get_average_tripartite_entanglement(num_samples, dim_a, dim_b, dim_c)
    # For each cut, the maximum entropy is log2(min(dim_iso, dim_rest)).
    max_a = torch.log2(torch.tensor(min(dim_a, dim_b * dim_c), dtype=torch.float32))
    max_b = torch.log2(torch.tensor(min(dim_b, dim_a * dim_c), dtype=torch.float32))
    max_c = torch.log2(torch.tensor(min(dim_c, dim_a * dim_b), dtype=torch.float32))

    return {
        "normalized_avg_entropy_a_bc": avg.get("avg_entropy_a_bc") / max_a,
        "normalized_avg_entropy_b_ac": avg.get("avg_entropy_b_ac") / max_b,
        "normalized_avg_entropy_c_ab": avg.get("avg_entropy_c_ab") / max_c,
    }


def get_average_entanglement(num_samples, dim_a, dim_b, normalize: bool = False, seed=0):
    generator = torch.Generator().manual_seed(seed)  # For reproducibility
    total_entropy_A = 0
    total_entropy_B = 0

    for i in range(num_samples):
        random_unit_vector = torch.randn(int(dim_a * dim_b), generator=generator)
        random_unit_vector /= torch.norm(random_unit_vector)
        
        entanglement = Entanglement(random_unit_vector, dim_a, dim_b)
        vne_entropy = entanglement.compute(normalize=normalize)
        vne_A = vne_entropy.get("entanglement_a")
        vne_B = vne_entropy.get("entanglement_b")
        
        total_entropy_A += vne_A
        total_entropy_B += vne_B

    avg_entropy_A = total_entropy_A / num_samples
    avg_entropy_B = total_entropy_B / num_samples
    
    return {"avg_entropy_A": avg_entropy_A, "avg_entropy_B": avg_entropy_B,}

def get_normalized_average_entanglement(num_samples, dim_a, dim_b):
    avg_entanglement = get_average_entanglement(num_samples, dim_a, dim_b)
    max_entropy = torch.log2(torch.tensor(min(dim_a, dim_b), dtype=torch.float32))
    
    normalized_avg_entropy_A = avg_entanglement.get("avg_entropy_A") / max_entropy
    normalized_avg_entropy_B = avg_entanglement.get("avg_entropy_B") / max_entropy
    
    return {"normalized_avg_entropy_A": normalized_avg_entropy_A, "normalized_avg_entropy_B": normalized_avg_entropy_B,}

if __name__ == "__main__":
    # # Sanity check, Bell pair example:
    # bell_pair = (1/torch.sqrt(torch.tensor(2.0))) * (torch.kron(torch.tensor([1, 0]), torch.tensor([1, 0])) + torch.kron(torch.tensor([0, 1]), torch.tensor([0, 1])))
    # entanglement = Entanglement(bell_pair, 2, 2)
    # vne_entropy = entanglement.compute()
    # print("Bell pair vector:", bell_pair)
    # print("Von Neumann Entropy (Bell pair):", vne_entropy)
    # print("-"*20)
    # # -----------------------------------------------

    # # Sanity check, also calculated by hand:
    # vector = 1/torch.sqrt(torch.tensor(3.0)) * (torch.kron(torch.tensor([1, 0]), torch.tensor([1, 0])) + torch.kron(torch.tensor([0, 1]), torch.tensor([0, 1])) + torch.kron(torch.tensor([1, 0]), torch.tensor([0, 1])))
    # entanglement = Entanglement(vector, 2, 2)
    # vne_entropy = entanglement.compute()
    # print("Vector:", vector)
    # print("Von Neumann Entropy of systems A and B:", vne_entropy)

    # l_m = 1/2 - torch.sqrt(torch.tensor(5.0))/6
    # l_p = 1/2 + torch.sqrt(torch.tensor(5.0))/6
    # s = - (l_m * torch.log2(l_m) + l_p * torch.log2(l_p))
    # print("Expected Von Neumann Entropy:", s)
    # print("-"*20)
    # # -----------------------------------------------


    # #implement for vectors in the tensor product space: I_8 \otimes rho_reg_g 
    # # let's say that the group representation we fixed was
    # # I_8 \otimes R where R is the matrix [[-1, 0], [0, -1]] (180 degree rotation in 2D)
    # # So, we have an 8-dim space tensored with a 2-dim space, making a 16-dim latent space

    # LATENT_DIMS = 16
    # IRREP_DIMS = 2

    # #TODO make the vectors come from the complex unit circle instead of a normal distribution — maybe
    # random_unit_vector = torch.randn(LATENT_DIMS) 
    # random_unit_vector /= torch.norm(random_unit_vector)
    # print("Random Unit Vector:", random_unit_vector)


    # entanglement = Entanglement(random_unit_vector, LATENT_DIMS / IRREP_DIMS, IRREP_DIMS)
    # vne_entropy = entanglement.compute()
    # print("\nVon Neumann Entropy of subsystems A and B for random unit vector:", vne_entropy)

    # # ------------------------------------------------

    # # Now let's calculate the average entanglement over many random vectors
    # num_samples = 1000
    # average_entanglement = get_average_entanglement(num_samples, LATENT_DIMS / IRREP_DIMS, IRREP_DIMS)
    # avg_entropy_A = average_entanglement.get("avg_entropy_A")
    # avg_entropy_B = average_entanglement.get("avg_entropy_B")

    # print(f"Average Von Neumann Entropy of subsystem A over {num_samples} samples:", avg_entropy_A)
    # print(f"Average Von Neumann Entropy of subsystem B over {num_samples} samples:", avg_entropy_B)

    # #TODO we could implement some sort of confidence interval here to tell whether the 
    # # entanglement of the learned latent vectors significantly differs from entanglement in random noise
    # # ------------------------------------------------

    # #sanity checking a specific vector taken from the latent space of the basic VAE
    # vector = torch.tensor([-0.25589415, -0.03218814,  0.07429797,  0.38550967, -0.21019217, -0.0940889,
    # 0.14527982,  0.4434251,   0.50494885,  0.08832321, -0.35899743,  0.01654475,
    # 0.18896575,  0.2596877,   0.1034211,   0.05300807])
    # entanglement = Entanglement(vector, LATENT_DIMS / IRREP_DIMS, IRREP_DIMS)
    # vne_entropy = entanglement.compute()
    # vne_entropy_A = vne_entropy.get("entanglement_a")

    # print("\nVon Neumann Entropy of subsystems A and B for given vector:", vne_entropy_A, type(vne_entropy_A))

    # sanity check for tripartite entanglement, averaged over many random vectors in 2x2x2 space:
    dim_a, dim_b, dim_c = 16, 2, 2
    avg_tripartite_entanglement = get_average_tripartite_entanglement(num_samples=1000, dim_a=dim_a, dim_b=dim_b, dim_c=dim_c, normalize=True)
    print(f"\nAverage tripartite entanglement for in {dim_a}x{dim_b}x{dim_c} space:", avg_tripartite_entanglement)

    # sanity check bipartite 2 x 4 space averaged over many random vectors:
    dim_a, dim_b = 2, 2
    avg_bipartite_entanglement = get_average_entanglement(num_samples=1000, dim_a=dim_a, dim_b=dim_b, normalize=True)
    print(f"\nAverage bipartite entanglement entropies for random unit vectors in {dim_a}x{dim_b} space:", avg_bipartite_entanglement)