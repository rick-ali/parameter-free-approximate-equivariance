from typing import List
import torch
import numpy as np

class CharTable:
    def __init__(self, table, group_order, conjugacy_classes_sizes):
        self.table = table
        self.group_order = group_order
        self.conjugacy_classes_sizes = conjugacy_classes_sizes

    def get_table(self) -> torch.Tensor:
        return
    
    def get_conjugacy_classes(self) -> str:
        return

    def get_generators_and_relations(self) -> str:
        return
    
    def calculate_irreducible_reps_dimensions(self, learnt_reps: torch.Tensor) -> List[int]:
        """
        Computes the n_i's in the decomposition V = n_1 * V_1 + ... + n_r * V_r, or rho = n_1 * rho_1 + ... + n_r * rho_r

        Args:
        char_table: character table of the group. It is a complex-valued tensor of shape (r, m), 
                    where r is the number of irreducible representations, and m is the number of conjugacy classes.
                    char_table[i, j] = chi_i(g_j), where chi_i is the character of the i-th irrep, and g_j representative of conjugacy class.
        learnt_reps: list of torch tensors with length m, learnt_reps[j] = rho(g_j). 
                     learnt_reps MUST be in the same order as the conjugacy classes in the character table (given by get_conjugacy_classes()).

        Returns: a list of length r, where the i-th element is n_i. It is computed through n_i = <chi_i, chi_rho>
        """
        assert len(learnt_reps) == self.table.shape[1]

        n_is = []
        
        # Compute trace of each representation
        traces = torch.tensor([rep.trace() for rep in learnt_reps])

        # Compute the complex inner product of each character with the character of the representation
        for chi in self.table:
            chi = chi * self.conjugacy_classes_sizes
            n_i = torch.dot(chi.conj(), traces.to(chi.conj().dtype)) / self.group_order
            n_is.append(n_i.item())

        return n_is


class Z2_CharTable(CharTable):
    def __init__(self):
        table = torch.tensor([[1., 1.], [1., -1.]])
        super().__init__(table=table, group_order=2., conjugacy_classes_sizes=torch.tensor([1., 1.]))

    def get_table(self) -> torch.Tensor:
        return self.table

    def get_conjugacy_classes(self) -> str:
        return "{1}, {g}"
    
    def get_generators_and_relations(self) -> str:
        return "<g | g^2 = 1>"
    

class D3_CharTable(CharTable):
    def __init__(self):
        table = torch.tensor([[1., 1., 1.], [1., -1., 1.], [2., 0., -1.]])
        super().__init__(table=table, group_order=6., conjugacy_classes_sizes=torch.tensor([1., 3., 2.]))

    def get_table(self) -> torch.Tensor:
        return self.table

    def get_conjugacy_classes(self) -> str:
        return "{1}, {s, sr, rs}, {r, r^2}"
    
    def get_generators_and_relations(self) -> str:
        return "<r, s | r^3 = s^2 = rsrs = 1>"
    

class Cn_CharTable(CharTable):
    def __init__(self, N: int):
        table = "No Table, directly compute eigenvalues of the representation rho(r)"
        self.N = N
        super().__init__(table=table, group_order=self.N, conjugacy_classes_sizes=torch.ones(self.N))

    def get_table(self) -> torch.Tensor:
        return self.table

    def get_conjugacy_classes(self) -> str:
        return "{1}, {r, r^2, ..., r^("+ str(self.N) +"-1)}"
    
    def get_generators_and_relations(self) -> str:
        return f"<r | r^{self.N} = 1>" 
    
    def calculate_irreducible_reps_dimensions(self, learnt_rep: torch.Tensor) -> List[int]:
        roots_of_unity = [torch.exp(torch.tensor([2 * np.pi * 1j * i / self.N])) for i in range(self.N)]

        eigenvalues, _ = torch.linalg.eig(learnt_rep)

        result = [0 for _ in range(self.N)]

        for eig in eigenvalues:
            best_distance = float('inf')
            for i, root in enumerate(roots_of_unity):
                current_distance = torch.dist(eig, root).item()
                if current_distance <= best_distance:
                    best_distance = current_distance
                    best_index = i
        
            result[best_index] += 1
        
        return result



if __name__ == "__main__":
    # Example usage
    print('Z2 example\n')
    learnt_reps = [torch.tensor([[1, 0], [0, 1]]), torch.tensor([[1, 0], [0, -1]])]
    print(f'Learnt reps:\n{learnt_reps}\n')
    table = Z2_CharTable()
    print(f'Table:\n{table.get_table()}\n')
    print(f'Conjugacy classes:\n{table.get_conjugacy_classes()}\n')
    print(f'Generators and relations:\n{table.get_generators_and_relations()}\n')
    print(f'Irreducible reps dimensions:\n{table.calculate_irreducible_reps_dimensions(learnt_reps)}\n')
    print('\n\n')
    # Example usage
    print('C2 example\n')
    learnt_reps = torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
    print(f'Learnt reps:\n{learnt_reps}\n')
    table = Cn_CharTable(2)
    print(f'Table:\n{table.get_table()}\n')
    print(f'Conjugacy classes:\n{table.get_conjugacy_classes()}\n')
    print(f'Generators and relations:\n{table.get_generators_and_relations()}\n')
    print(f'Irreducible reps dimensions:\n{table.calculate_irreducible_reps_dimensions(learnt_reps)}\n')
    