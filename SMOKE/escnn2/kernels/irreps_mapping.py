from collections import defaultdict
from functools import partial
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import torch
from escnn import gspaces
from escnn.group import *
from escnn.kernels.steerable_filters_basis import SteerableFiltersBasis
from torch.nn.functional import normalize

from .fouriernonliniearity import FourierPointwise, InvFourier
from escnn.group import o2_group, so2_group
import time


class IrrepsMapFourierBLact(torch.nn.Module):
    def __init__(self, layer_id: Union[int, None], group: Group, L=2, L_out=None):
        """
        Module which computes the mapping between a basis irrep and a TODO through
        modeling the fourier transform of a bandlimited learnable probability
        distribution initialized as uniform over the group. This allows the kernels
        to break equivariance for specific parts of the group.

        Args:
            layer_id (int, None): id for the layer. Layers with the same id
                                share the degree of equivariance
            group (Group): the group of the basis

        """
        super(IrrepsMapFourierBLact, self).__init__()
        self.key = (layer_id, str(group), L)
        self.layer_id = layer_id
        self.group = group
        # Dictionary of cached irrepsmap matrices
        self.c = {}
        # Dictionary of cached psi_fts
        self.psi_fts = {}
        # Dict of tensor product elements of irrep pairs
        self.psis = {}
        # set which contains the irrep pairs with empty bandlimited tensor
        # product
        self.empty_tensor_prod = set()
        self.dims = {}

        # Initialize activation function to ensure the parameters model a
        # true probability distribution
        L_out = L if L_out is None else L_out
        if group.order() > -1:
            rotation = getattr(group, "rotation_order", None)
            if rotation is None:
                rotation = group.order()
            L = min(L, rotation // 2)
            L_out = min(L_out, rotation // 2)
        self.L = L
        self.L_out = L_out
        try:
            self.in_irreps = group.bl_irreps(self.L)
            self.out_irreps = group.bl_irreps(self.L_out)
        except:
            self.in_irreps = group._irreps
            self.out_irreps = group._irreps

        self.act = gspaces.no_base_space(group)

        if self.group.continuous:
            N = group.bl_regular_representation(max(self.L, self.L_out)).size
        else:
            N = len(group.elements)

        if "D" in str(self.group) or (
            "SO" not in str(self.group) and "O" in str(self.group)
        ):
            N //= 2

        # Use 'regular' samping whenever possible.
        # types = ["regular", "thomson"]
        # temp
        if "(3)" in str(self.group):
            self.grid_type = "thomson"
            identity_grid = True
            # Compensate for not perfectly regular grid sampling.
            N *= 2
        elif "(2)" in str(self.group):
            self.grid_type = "regular"
            identity_grid = True
        else:
            self.grid_type = "regular"
            identity_grid = False

        self.N = N
        self.activation = FourierPointwise(
            self.act,
            irreps=self.in_irreps,
            out_irreps=self.out_irreps,
            channels=1,
            function="softmax",
            type=self.grid_type,
            identity_grid=identity_grid,
            normalize=False,
            N=self.N,
        )
        # Compute the initialization of fourier transform coefficients.
        self.in_size = 0
        fts = []

        for irrep in self.in_irreps:
            irrep = group.irrep(*irrep)
            d = irrep.size**2 // irrep.sum_of_squares_constituents
            ft = torch.ones(d) if irrep.is_trivial() else torch.zeros(d)
            self.in_size += d
            fts.append(ft)

        # Compute the indices of the output irreps in the normalized fourier coefficients.
        self.psi_inds = {}
        self.out_size = 0
        for irrep in self.out_irreps:
            irrep = group.irrep(*irrep)
            self.register_buffer(
                f"endomorphism_{irrep.id}",
                torch.from_numpy(irrep.endomorphism_basis()).float(),
            )
            d = irrep.size**2 // irrep.sum_of_squares_constituents
            self.psi_inds[irrep.id] = torch.tensor(
                range(self.out_size, self.out_size + d)
            )
            self.out_size += d
        self.fts_in = torch.nn.Parameter(torch.cat(fts), requires_grad=True)
        self._dev = self.fts_in.device

        # Register a flag which is set to True at init and when fts_in has been
        # updated. This will allow us to only recompute the normalized fourier
        # transform once forward pass.
        self.fts_in.register_hook(partial(set_grad_flag, self))
        set_grad_flag.recompute_ft[self] = True

    _cached_instances = {}

    def __call__(self, j_basis, j_cg):
        return self.forward(j_basis, j_cg)

    def kl_divergence(self, other):
        if other is not None:
            assert isinstance(other, IrrepsMapFourierBLact)

        p_ll = self.fts_in.flatten()
        p_distr = self.fts_out.detach()
        log_zp = self.activation.z
        min_len_p = min(p_ll.shape[0], p_distr.shape[0])
        p_prod = p_distr[:min_len_p] @ p_ll[:min_len_p] - log_zp

        if other is None:
            q_ll = []
            for irrep in self.in_irreps:
                irrep = self.group.irrep(*irrep)
                d = irrep.size**2 // irrep.sum_of_squares_constituents
                ft = torch.ones(d) if irrep.is_trivial() else torch.zeros(d)
                q_ll.append(ft)
            q_ll = torch.cat(q_ll).to(p_ll.device)
            log_zq = 1

        # TODO: Support different groups
        elif other.group != self.group:
            return torch.tensor(0)
        else:
            q_ll = other.fts_in.detach().flatten()
            # p_distr = other.fts_out.detach()
            min_len_q = min(q_ll.shape[0], p_distr.shape[0])
            log_zq = other.activation.z.detach()

        min_len_q = min(p_distr.shape[0], q_ll.shape[0])
        out = p_prod - p_distr[:min_len_q] @ q_ll[:min_len_q] + log_zq
        return out

    def inv_fourier(self, g):
        f = 0
        for psi in self.psi_inds:
            inds = self.psi_inds[psi]
            psi = self.group.irrep(*psi)
            d = psi.size**2 // psi.sum_of_squares_constituents
            ft = (
                self.fts_out[inds]
                .view(psi.size, psi.size // psi.sum_of_squares_constituents)
                .detach()
                .cpu()
                .numpy()
            )
            f += (
                np.sqrt(d)
                * (ft * psi(g)[:, : psi.size // psi.sum_of_squares_constituents]).sum()
            )
        return f

    def get_distribution(self, n=100, sphere=False, grad=False):
        n_theta = 4
        with torch.enable_grad() if grad else torch.no_grad():
            if hasattr(self, "inv_fourier_fn") and self.sampling_n == n:
                invfourier_fn = getattr(self, "inv_fourier_fn")
            else:
                grid = None
                if isinstance(self.group, SO3) or isinstance(self.group, O3):
                    angles = torch.cat(
                        [
                            torch.linspace(0, np.pi, n // 2),
                            torch.linspace(np.pi, 2 * np.pi, n // 2),
                        ]
                    )
                    phi = np.linspace(0, 2 * np.pi, n)
                    gamma = np.linspace(np.pi / n, np.pi, n)

                    phi = torch.cat(
                        [
                            torch.linspace(0, np.pi, n // 2),
                            torch.linspace(np.pi, 2 * np.pi, n // 2),
                        ]
                    )

                    gamma = torch.cat(
                        [
                            torch.linspace(0, 0.5 * np.pi, n // 2),
                            torch.linspace(0.5 * np.pi, np.pi, n // 2),
                        ]
                    )

                    gamma, phi = np.meshgrid(phi, gamma)

                    x = np.sin(gamma) * np.cos(phi)
                    y = np.sin(gamma) * np.sin(phi)
                    z = np.cos(gamma)

                    vectors = np.array([x.flatten(), y.flatten(), z.flatten()]).T

                    grid_3d_ids = [self.group.identity]
                    grid_3d = []
                    if isinstance(self.group, SO3):
                        for theta in np.linspace(np.pi / 4, np.pi, n_theta):
                            grid_3d += [
                                self.group.element(theta * vector, param="EV")
                                for vector in vectors
                            ]
                        grid = (
                            [
                                self.group.element((angle, 0, 0), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((0, angle, 0), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((0, 0, angle), param="XYZ")
                                for angle in angles
                            ]
                        )
                    else:
                        grid_3d_ids += [
                            self.group.element((1, np.array([0, 0, 0])), param="EV")
                        ]
                        for theta in np.linspace(np.pi / 4, np.pi, n_theta):
                            grid_3d += [
                                self.group.element((0, theta * vector), param="EV")
                                for vector in vectors
                            ]
                        for theta in np.linspace(np.pi / 4, np.pi, n_theta):
                            grid_3d += [
                                self.group.element((1, theta * vector), param="EV")
                                for vector in vectors
                            ]
                        grid = (
                            [
                                self.group.element((0, (angle, 0, 0)), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((0, (0, angle, 0)), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((0, (0, 0, angle)), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((1, (angle, 0, 0)), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((1, (0, angle, 0)), param="XYZ")
                                for angle in angles
                            ]
                            + [
                                self.group.element((1, (0, 0, angle)), param="XYZ")
                                for angle in angles
                            ]
                        )

                    self.inv_fourier_fn3d = InvFourier(
                        self.act,
                        irreps=self.out_irreps,
                        channels=1,
                        grid=grid_3d_ids + grid_3d,
                        type=self.grid_type,
                        parametrization="EV",
                        normalize=False,
                        N=n,
                    )

                    # self._samp_angles = angles_sphere
                    # self._samp_angles_alpha = angles_alpha

                    invfourier_fn = InvFourier(
                        self.act,
                        irreps=self.out_irreps,
                        channels=1,
                        grid=grid,
                        type=self.grid_type,
                        parametrization="XYZ",
                        normalize=False,
                        N=n,
                    )
                else:
                    invfourier_fn = InvFourier(
                        self.act,
                        irreps=self.out_irreps,
                        channels=1,
                        grid=grid,
                        type=self.grid_type,
                        normalize=False,
                        N=n,
                    )

                self.inv_fourier_fn = invfourier_fn
                self.sampling_n = n

            x = invfourier_fn.in_type(self.fts_out[None, :].detach().cpu())
            if hasattr(self, "inv_fourier_fn3d") and sphere:
                f_3d = self.inv_fourier_fn3d(x).flatten().numpy()
                factor = len(f_3d) // (n**2 * n_theta)
                grid_3d = self.inv_fourier_fn3d.grid[-(n**2) * n_theta * factor :]
                grid_3d_ids = self.inv_fourier_fn3d.grid[: -(n**2) * n_theta * factor]
                sphere_data = {
                    "f": f_3d[len(grid_3d_ids) :],
                    "f_id": f_3d[: len(grid_3d_ids)],
                    "grid": grid_3d,
                    "grid_id": grid_3d_ids,
                    "n": n,
                    "n_theta": n_theta,
                }
            else:
                sphere_data = None
            f = invfourier_fn(x).flatten().cpu().numpy()
            if isinstance(self.group, O2):
                elements = np.concatenate(
                    [
                        np.linspace(0, 2 * np.pi, n, endpoint=False),
                        np.linspace(2 * np.pi, 4 * np.pi, n, endpoint=False),
                    ]
                ).reshape(1, -1)
                f = f.reshape(1, -1)
            elif isinstance(self.group, SO2):
                elements = np.linspace(0, 2 * np.pi, n).reshape(1, -1)
                f = f.reshape(1, -1)
            elif isinstance(self.group, CyclicGroup):
                elements = np.linspace(
                    0, 2 * np.pi, self.group.order(), endpoint=False
                ).reshape(1, -1)
                f = f.reshape(1, -1)
            elif isinstance(self.group, DihedralGroup):
                elements = np.concatenate(
                    [
                        np.linspace(
                            0, 2 * np.pi, self.group.order() // 2, endpoint=False
                        ),
                        np.linspace(
                            2 * np.pi,
                            4 * np.pi,
                            self.group.order() // 2,
                            endpoint=False,
                        ),
                    ]
                ).reshape(1, -1)
                f = f.reshape(1, -1)
            elif isinstance(self.group, O3):
                elements = np.concatenate(
                    [
                        np.linspace(0, np.pi, n // 2),
                        np.linspace(np.pi, 2 * np.pi, n // 2),
                        np.linspace(2 * np.pi, 3 * np.pi, n // 2),
                        np.linspace(3 * np.pi, 4 * np.pi, n // 2),
                    ]
                    * 3
                ).reshape(3, -1)
                f = f.reshape(3, -1)
            elif isinstance(self.group, SO3):
                elements = np.concatenate(
                    [
                        np.linspace(0, np.pi, n // 2),
                        np.linspace(np.pi, 2 * np.pi, n // 2),
                    ]
                    * 3
                ).reshape(3, -1)
                f = f.reshape(3, -1)

            return f, elements, sphere_data

    def get_fourier_transforms(
        self, psis: List[IrreducibleRepresentation]
    ) -> List[torch.Tensor]:
        """
        Computes/retrieves the normalized fourier transform coefficients for
        each irrep in psis.

        Args:
            - psis (list[IrreducibleRepresentation]): list of irreps
        Returns:
            - fts_out_list (list[Torch.tensor]): list of normalized fourier coefficients
        """
        # Only recompute the normalization of the fts when update was performed
        # or device changed.
        if set_grad_flag.recompute_ft[self] or self._dev != self.fts_in.device:
            # Remove cached maps
            self.c = {}
            self.psi_fts = {}
            # Compute normalized fts
            fts_in = self.activation.in_type(self.fts_in.view(1, -1))
            fts_out = self.activation(fts_in)
            self.register_buffer("fts_out", fts_out.tensor.flatten())
            if hasattr(self.activation, "identity_val"):
                self.shift_loss = -self.activation.identity_val
            else:
                self.shift_loss = torch.tensor(0)
            set_grad_flag.recompute_ft[self] = False
            self._dev = self.fts_in.device
            # self.get_distribution()

        fts_out = self.fts_out
        # Gather fts of psis
        fts_out_list = []
        for psi in psis:
            if psi.id in self.psi_fts:
                full_fts = self.psi_fts[psi.id]
            else:
                endomorphism = getattr(self, f"endomorphism_{psi.id}")
                fts = fts_out[self.psi_inds[psi.id]].view(
                    psi.size, psi.size // psi.sum_of_squares_constituents
                )
                full_fts = torch.einsum("knm, ml -> nkl", endomorphism, fts).view(
                    endomorphism[0].shape
                )
                self.psi_fts[psi.id] = full_fts
            fts_out_list.append(full_fts)
        return fts_out_list

    def get_bandlimited_tp(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
        change_basis: torch.Tensor,
        change_basis_inv: torch.tensor,
    ) -> Tuple[List[IrreducibleRepresentation], torch.tensor, torch.tensor, np.array]:
        """Computes the bandlimited list of irreps in the tensor product of
        j_basis and j_cg. Filters out the columns and rows of the change of
        basis and inverse change of basis

        Args:
            - j_basis (IrreducibleRepresentation): irrep of the basis
            - j_cg (IrreducibleRepresentation): irrep of TODO
            - change_basis (torch.Tensor): change of basis matrix
            - change_basis_inv (torch.Tensor): inverse of change of basis matrix
        Returns:
            - psis (list[IrreducibleRepresentation])
            - change_basis (torch.tensor): masked change of basis
            - change_basis_inv (torch.tensor): masked inverse change of basis
            - change_basis_inds (np.array): mask used to mask the change of basis.
        """
        # Compute tensor product
        psis = [self.group.irrep(*irrep) for irrep in j_basis.tensor(j_cg).irreps]

        # Filter irreps based on maximum frequency
        psi_inds = [psi.id in self.group.bl_irreps(self.L_out) for psi in psis]

        # If all irreps are filtered.
        if sum(psi_inds) == 0:
            return None, None, None, None

        # Compute the corresponding boolean indices for the change of basis matrix
        sizes = [psi.size for psi in psis]
        change_basis_inds = [[psi_ind] * size for psi_ind, size in zip(psi_inds, sizes)]
        change_basis_inds = np.asarray(
            [ind for indlist in change_basis_inds for ind in indlist]
        )

        # Apply masking and return
        psis = [psi for i, psi in enumerate(psis) if psi_inds[i]]

        return (
            psis,
            change_basis[:, change_basis_inds],
            change_basis_inv[change_basis_inds],
            change_basis_inds,
        )

    def get_vars(
        self,
        j_basis: IrreducibleRepresentation,
        j_cg: IrreducibleRepresentation,
    ) -> Tuple[List[IrreducibleRepresentation], torch.tensor, torch.tensor]:
        """
        Computes (or obtains the cached) change of basis, change of basis inverse
        and bandlimited tensor product from j_basis and j_cg.

        Args:
            - j_basis (IrreducibleRepresentation): irrep of the basis
            - j_cg (IrreducibleRepresentation): irrep of the TODO

        Returns:
            - psis (list[IrreducibleRepresentation]): list of irreps in tensor product
            - change_basis (torch.tensor): change of basis
            - w (torch.tensor)
        """
        key = f"{j_basis.id}{j_cg.id}"
        # Empty bandlimited tensor product.
        if key in self.empty_tensor_prod:
            return None, None
        # Cache results.
        elif not hasattr(self, f"_change_of_basis_{key}"):
            j_tensor = j_basis.tensor(j_cg)
            change_basis, change_basis_inv = (
                torch.from_numpy(j_tensor.change_of_basis).float(),
                torch.from_numpy(j_tensor.change_of_basis_inv).float(),
            )

            # Compute bandlimited tensor product between irreppair
            (
                self.psis[key],
                change_basis,
                change_basis_inv,
                mask,
            ) = self.get_bandlimited_tp(j_basis, j_cg, change_basis, change_basis_inv)

            size_j_cg = j_cg.size
            size_j_basis = j_basis.size
            # Empty bandlimited tensor product.
            if mask is None:
                self.register_buffer(
                    f"_empty_c_{key}", torch.zeros(0, size_j_basis, size_j_cg)
                )
                self.empty_tensor_prod.add(key)

            self.register_buffer(f"_change_of_basis_{key}", change_basis)
            self.dims[key] = 0 if mask is None else int(sum(mask))

        # Obtain cached results
        change_basis = getattr(self, f"_change_of_basis_{key}")
        # w = getattr(self, f"_w_{key}")
        psis = self.psis[f"{key}"]

        return psis, change_basis

    def dim(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
    ) -> int:
        j_basis = self.group.irrep(*self.group.get_irrep_id(j_basis))
        j_cg = self.group.irrep(*self.group.get_irrep_id(j_cg))
        key = f"{j_basis.id}{j_cg.id}"
        if key not in self.dims:
            self(j_basis, j_cg)

        return self.dims[key]

    def forward(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
    ) -> torch.tensor:
        """
        Computes the mapping between j_basis and j_cg.

        Args:
            - j_basis (IrreducibleRepresentation): irrep of the basis
            - j_cg (IrreducibleRepresentation): irrep of the TODO

        Returns:
            - c_jj (torch.tensor): mapping between the irreppair.
        """
        j_basis = self.group.irrep(*self.group.get_irrep_id(j_basis))
        j_cg = self.group.irrep(*self.group.get_irrep_id(j_cg))
        psis, change_basis = self.get_vars(j_basis, j_cg)
        key = f"{j_basis.id}{j_cg.id}"
        # Empty bandlimited tensor product
        if psis is None:
            # print(j_basis, j_cg)
            c_jj = getattr(self, f"_empty_c_{key}")
            # print(c_jj)
        # Return cashed irrepsmapping if it is already generated on the current parameter update
        elif (
            not set_grad_flag.recompute_ft[self]
            and self._dev == self.fts_in.device
            and key in self.c
        ):
            c_jj = self.c[key]
        else:
            # print(
            #     not set_grad_flag.recompute_ft[self],
            #     self._dev == self.fts_in.device,
            #     key in self.c,
            # )
            fts = self.get_fourier_transforms(psis)

            psi_ft_block = torch.block_diag(*fts)
            p_jj = change_basis @ psi_ft_block

            size_j_cg = j_cg.size
            size_j_basis = j_basis.size

            p_jj = p_jj.view(size_j_basis, size_j_cg, change_basis.shape[1])
            c_jj = p_jj.permute(2, 0, 1)

            self.c[key] = c_jj
        return c_jj

    def __hash__(self):
        return (
            hash(self.layer_id)
            + hash(str(self.group))
            + hash(self.L * 1000)
            + hash(self.L_out * 1000000)
        )

    @classmethod
    def _generator(
        cls,
        layer_id: Union[int, None],
        group: Group,
        L: int = 2,
        L_out: Union[int, None] = None,
    ) -> "IrrepsMapFourierBLact":
        key = layer_id, str(group), L
        if key not in cls._cached_instances:
            cls._cached_instances[key] = IrrepsMapFourierBLact(
                layer_id, group, L=L, L_out=L_out
            )

        return cls._cached_instances[key]


class IrrepsMapFourierBL(torch.nn.Module):
    def __init__(self, layer_id: Union[int, None], group: Group):
        super(IrrepsMapFourierBL, self).__init__()
        self.layer_id = layer_id
        self.group = group
        self.fts = torch.nn.ParameterDict({})
        self.shared = True
        self.identifier = defaultdict(int)
        self.psis = {}
        self.empty_tensor_prod = set()

    _cached_instances = {}

    def __call__(self, j_basis, j_cg):
        return self.forward(j_basis, j_cg)

    def get_fourier_transforms(self, psis):
        fts = []
        for psi in psis:
            if str(psi.id) not in self.fts:
                d = psi.size
                self.fts[str(psi.id)] = torch.nn.Parameter(
                    (torch.ones(d, d) if psi.is_trivial() else torch.zeros(d, d)),
                    requires_grad=True,
                )
            fts.append(self.fts[str(psi.id)])
        return fts

    def get_bandlimited_tp(self, j_basis, j_cg, change_basis, change_basis_inv, L=4):
        psis = [
            self.group.irrep(*irrep)
            for irrep, _ in self.group._tensor_product_irreps(j_basis, j_cg)
        ]

        psi_inds = [
            True if not any(list(f > L for f in psi.id)) else False for psi in psis
        ]

        if sum(psi_inds) == 0:
            return None, None, None, None

        sizes = [psi.size for psi in psis]

        change_basis_inds = [[psi_ind] * size for psi_ind, size in zip(psi_inds, sizes)]

        change_basis_inds = np.asarray(
            [ind for indlist in change_basis_inds for ind in indlist]
        )

        psis = [psi for i, psi in enumerate(psis) if psi_inds[i]]

        return (
            psis,
            change_basis[:, change_basis_inds],
            change_basis_inv[change_basis_inds],
            change_basis_inds,
        )

    def get_vars(
        self,
        j_basis: IrreducibleRepresentation,
        j_cg: IrreducibleRepresentation,
    ):
        key = f"{j_basis.id}{j_cg.id}"
        if key in self.empty_tensor_prod:
            return None, None, None

        elif not hasattr(self, f"_w_{key}"):
            j_tensor = j_basis.tensor(j_cg)
            size_j_cg = j_cg.size
            size_j_basis = j_basis.size
            change_basis, change_basis_inv = (
                torch.from_numpy(j_tensor.change_of_basis).float(),
                torch.from_numpy(j_tensor.change_of_basis_inv).float(),
            )

            (
                self.psis[key],
                change_basis,
                change_basis_inv,
                mask,
            ) = self.get_bandlimited_tp(j_basis, j_cg, change_basis, change_basis_inv)

            if mask is None:
                self.register_buffer(
                    f"_c_{key}", torch.zeros(1, size_j_basis, size_j_cg)
                )
                self.empty_tensor_prod.add(key)

            self.register_buffer(f"_change_of_basis_{key}", change_basis)
            self.register_buffer(f"_change_of_basis_inv_{key}", change_basis_inv)

            w = (
                change_basis_inv @ torch.eye(size_j_basis).flatten()
                if j_basis == j_cg
                else torch.ones(size_j_basis * size_j_cg)[mask]
            )
            self.register_buffer(f"_w_{key}", w)

        change_basis = getattr(self, f"_change_of_basis_{key}")
        change_basis_inv = getattr(self, f"_change_of_basis_inv_{key}")
        w = getattr(self, f"_w_{key}")
        psis = self.psis[f"{key}"]

        return psis, change_basis, w

    def forward(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
    ) -> "torch.tensor":
        j_basis = self.group.irrep(*self.group.get_irrep_id(j_basis))
        j_cg = self.group.irrep(*self.group.get_irrep_id(j_cg))

        psis, change_basis, w = self.get_vars(j_basis, j_cg)
        if psis is None:
            c_jj = getattr(self, f"_c_{j_basis.id}{j_cg.id}")
        else:
            fts = self.get_fourier_transforms(psis)

            psi_ft_block = torch.block_diag(*fts)

            p_jj = change_basis @ psi_ft_block  # @ change_basis_inv

            size_j_cg = j_cg.size
            size_j_basis = j_basis.size

            p_jj = p_jj.view(size_j_basis, size_j_cg, change_basis.shape[1])
            # print(j_basis, j_cg)
            # print(psi_ft_block.detach().cpu().numpy())
            # print(change_basis.cpu().numpy())
            # print(p_jj.permute(2, 0, 1).detach().cpu().numpy())
            c_jj = (p_jj @ w)[None, :]

        return c_jj

    @classmethod
    def _generator(
        cls, layer_id: Union[int, None], basis: SteerableFiltersBasis
    ) -> "IrrepsMapFourierBL":
        key = layer_id, str(basis.group)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = IrrepsMapFourierBL(layer_id, basis.group)

        return cls._cached_instances[key]


class IrrepsMapFourier(torch.nn.Module):
    def __init__(self, layer_id: Union[int, None], group: Group):
        super(IrrepsMapFourier, self).__init__()
        self.layer_id = layer_id
        self.group = group
        self.fts = torch.nn.ParameterDict({})
        self.shared = True
        self.identifier = defaultdict(int)
        self.psis = {}

    _cached_instances = {}

    def __call__(self, j_basis, j_cg):
        return self.forward(j_basis, j_cg)

    def get_fourier_transforms(self, psis):
        fts = []
        for psi in psis:
            if str(psi.id) not in self.fts:
                d = psi.size
                self.fts[str(psi.id)] = torch.nn.Parameter(
                    (torch.ones(d, d) if psi.is_trivial() else torch.zeros(d, d)),
                    requires_grad=True,
                )

            fts.append(self.fts[str(psi.id)])

        return fts

    def get_vars(
        self,
        j_basis: IrreducibleRepresentation,
        j_cg: IrreducibleRepresentation,
    ):
        key = f"{j_basis.id}{j_cg.id}"
        if not hasattr(self, f"_change_of_basis_{j_basis.id}{j_cg.id}"):
            self.psis[key] = [
                self.group.irrep(*irrep)
                for irrep, _ in self.group._tensor_product_irreps(j_basis, j_cg)
            ]
            j_tensor = j_basis.tensor(j_cg)
            change_basis, change_basis_inv = (
                torch.from_numpy(j_tensor.change_of_basis).float(),
                torch.from_numpy(j_tensor.change_of_basis_inv).float(),
            )
            self.register_buffer(f"_change_of_basis_{key}", change_basis)
            self.register_buffer(f"_change_of_basis_inv_{key}", change_basis_inv)
            size_j_cg = j_cg.size
            size_j_basis = j_basis.size
            w = (
                change_basis_inv @ torch.eye(size_j_basis).flatten()
                if j_basis == j_cg
                else torch.ones(size_j_basis * size_j_cg)
            )
            self.register_buffer(f"_w_{key}", w)

        change_basis = getattr(self, f"_change_of_basis_{key}")
        change_basis_inv = getattr(self, f"_change_of_basis_inv_{key}")
        w = getattr(self, f"_w_{key}")
        psis = self.psis[f"{key}"]

        return psis, change_basis, w

    def forward(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
    ) -> "torch.tensor":
        j_basis = self.group.irrep(*self.group.get_irrep_id(j_basis))
        j_cg = self.group.irrep(*self.group.get_irrep_id(j_cg))

        psis, change_basis, w = self.get_vars(j_basis, j_cg)
        fts = self.get_fourier_transforms(psis)

        psi_ft_block = torch.block_diag(*fts)

        p_jj = change_basis @ psi_ft_block  # @ change_basis_inv

        size_j_cg = j_cg.size
        size_j_basis = j_basis.size

        p_jj = p_jj.reshape(size_j_basis, size_j_cg, size_j_basis * size_j_cg)
        # print(j_basis, j_cg)
        # print(psi_ft_block.detach().cpu().numpy())
        # print(change_basis.cpu().numpy())
        # print(p_jj.permute(2, 0, 1).detach().cpu().numpy())
        p_jj = (p_jj @ w)[None, :]

        return p_jj

    @classmethod
    def _generator(
        cls, layer_id: Union[int, None], basis: SteerableFiltersBasis
    ) -> "IrrepsMapFourier":
        key = layer_id, str(basis.group)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = IrrepsMapFourier(layer_id, basis.group)

        return cls._cached_instances[key]


class IrrepsMap:
    def __init__(self, layer_id: Union[int, None], group: Group):
        self.layer_id = layer_id
        self.group = group
        self.c = {}
        self.shared = True
        self.identifier = defaultdict(int)

    _cached_instances = {}

    def __call__(self, j_basis, j_cg):
        return self.forward(j_basis, j_cg)

    def forward(
        self,
        j_basis: Union[IrreducibleRepresentation, Tuple],
        j_cg: Union[IrreducibleRepresentation, Tuple],
    ) -> "torch.nn.Parameter":
        j_basis = self.group.get_irrep_id(j_basis)
        j_cg = self.group.get_irrep_id(j_cg)
        id_key = f"{(j_basis, j_cg)}"
        if not self.shared:
            self.identifier[id_key] += 1
        key = f"{self.layer_id} {(j_basis, j_cg)} {self.identifier[id_key]}"
        if key not in self.c:
            size_j_cg = self.group.irrep(*j_cg).size
            size_j_basis = self.group.irrep(*j_basis).size
            param = (
                torch.eye(size_j_basis, size_j_cg).reshape(1, size_j_basis, size_j_cg)
                if j_cg == j_basis
                else torch.zeros((1, size_j_basis, size_j_cg), dtype=torch.float32)
            )
            self.c[key] = torch.nn.Parameter(param)
        return self.c[key]

    @classmethod
    def _generator(
        cls, layer_id: Union[int, None], basis: SteerableFiltersBasis
    ) -> "IrrepsMap":
        key = layer_id, str(basis.group)

        if key not in cls._cached_instances:
            cls._cached_instances[key] = IrrepsMap(layer_id, basis.group)

        return cls._cached_instances[key]


def set_grad_flag(irrepsmap, _):
    set_grad_flag.recompute_ft[irrepsmap] = True


set_grad_flag.recompute_ft = {}
