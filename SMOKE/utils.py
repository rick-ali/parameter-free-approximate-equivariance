import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.utils import data
import wandb
import os
import matplotlib.pyplot as plt
from escnn2.kernels.irreps_mapping import IrrepsMapFourierBLact
from typing import Dict, List
from utils_repr import W as W_repr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def plot_signal(sphere_data: dict, label=None, f_max=None, f_min=None) -> None:
    # set up figure layouts
    _axis = dict(
        showbackground=False,
        showticklabels=False,
        showgrid=False,
        zeroline=True,
        title="",
        nticks=3,
    )

    _layout = dict(
        scene=dict(
            xaxis=dict(
                **_axis,
            ),
            yaxis=dict(
                **_axis,
            ),
            zaxis=dict(
                **_axis,
            ),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=1, r=1, t=1, b=1),
    )

    _cmap_bwr = [
        [0, "rgb(0,50,255)"],
        [0.5, "rgb(200,200,200)"],
        [1, "rgb(255,50,0)"],
    ]

    f, f_identity, n, grid, n_theta = (
        sphere_data["f"],
        sphere_data["f_id"],
        sphere_data["n"],
        sphere_data["grid"],
        sphere_data["n_theta"],
    )
    if f_max is None:
        f_max = max(np.max(f), np.max(f_identity))
    if f_min is None:
        f_min = min(np.min(f), np.min(f_identity))

    n_points = (n**2) * n_theta
    # n_points = (n**2) * len(alphas)

    fs = np.split(f, len(f) // n_points)
    grid = grid
    if isinstance(grid[0].to("EV"), tuple):
        grid_array = np.array([elem.to("EV")[1] for elem in grid])
    else:
        grid_array = np.array([elem.to("EV") for elem in grid])
    thetas = np.linalg.norm(grid_array, axis=1)
    unique_theta = np.unique(np.round(thetas, 3))
    # grid_array /= thetas[:, None]
    # x = torch.outer(torch.cos(gamma), torch.sin(beta))
    # y = torch.outer(torch.sin(gamma), torch.sin(beta))
    # z = torch.outer(torch.ones(n), torch.cos(beta))

    x, y, z = (
        grid_array[: n**2, 0] / unique_theta[0],
        grid_array[: n**2, 1] / unique_theta[0],
        grid_array[: n**2, 2] / unique_theta[0],
    )

    thetas = [0] + list(unique_theta)
    figs = []
    for j, f in enumerate(fs):
        frames = []
        for i, theta in enumerate(thetas):
            data = []
            if i == 0:
                f_theta = np.ones_like(f[: n**2].reshape(n, n)) * f_identity[j]
                x_max, y_max, z_max = 0, 0, 0
                x_min, y_min, z_min = 0, 0, 0
            else:
                f_theta = f[(i - 1) * (n**2) : i * (n**2)].reshape(n, n)
                max_ind = np.argmax(f_theta)
                min_ind = np.argmin(f_theta)
                x_max, y_max, z_max = x[max_ind], y[max_ind], z[max_ind]
                x_min, y_min, z_min = x[min_ind], y[min_ind], z[min_ind]

            data += [
                go.Surface(
                    x=x.reshape(n, n),
                    y=y.reshape(n, n),
                    z=z.reshape(n, n),
                    surfacecolor=f_theta,
                    colorscale=_cmap_bwr,
                    cmin=f_min - 1e-3,
                    cmax=f_max,
                    opacity=1,
                    name=f"Equivariance",
                    showlegend=True,
                ),
                go.Scatter3d(
                    x=[x_max * 0.9, x_max * 1.5],
                    y=[y_max * 0.9, y_max * 1.5],
                    z=[z_max * 0.9, z_max * 1.5],
                    mode="lines+text",
                    line=dict(color="purple", width=30),
                    name="Maximum equivariance",
                    showlegend=True,
                    # text=["", f"({x_max:.2f}, {y_max:.2f}, {z_max:.2f})"],
                    # textfont=dict(size=60, color="purple"),
                ),
                go.Scatter3d(
                    x=[x_min * 0.9, x_min * 1.5],
                    y=[y_min * 0.9, y_min * 1.5],
                    z=[z_min * 0.9, z_min * 1.5],
                    mode="lines+text",
                    line=dict(color="darkgreen", width=30),
                    name="Minimum equivariance",
                    showlegend=True,
                    # text=["", f"({x_min:.2f}, {y_min:.2f}, {z_min:.2f})"],
                    # textfont=dict(size=60, color="darkgreen"),
                ),
                go.Scatter3d(
                    x=[-x_max * 0.9, -x_max * 1.5],
                    y=[-y_max * 0.9, -y_max * 1.5],
                    z=[-z_max * 0.9, -z_max * 1.5],
                    mode="lines+text",
                    line=dict(color="purple", width=30, dash="longdash"),
                    name="Maximum equivariance complement",
                    showlegend=True,
                    # text=["", f"({x_max:.2f}, {y_max:.2f}, {z_max:.2f})"],
                    # textfont=dict(size=60, color="purple"),
                ),
                go.Scatter3d(
                    x=[-x_min * 0.9, -x_min * 1.5],
                    y=[-y_min * 0.9, -y_min * 1.5],
                    z=[-z_min * 0.9, -z_min * 1.5],
                    mode="lines+text",
                    line=dict(color="darkgreen", width=30, dash="longdash"),
                    name="Minimum equivariance complement",
                    showlegend=True,
                    # text=["", f"({x_min:.2f}, {y_min:.2f}, {z_min:.2f})"],
                    # textfont=dict(size=60, color="darkgreen"),
                ),
                go.Scatter3d(
                    x=[-1.5, 1.5],
                    y=[0, 0],
                    z=[0, 0],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="x-axis",
                    # showlegend=True,
                    text=["", "x"],
                    textfont=dict(size=60),
                ),
                go.Scatter3d(
                    x=[0, 0],
                    y=[-1.5, 1.5],
                    z=[0, 0],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="y-axis",
                    # showlegend=True,
                    text=["", "y"],
                    textfont=dict(size=60),
                ),
                go.Scatter3d(
                    x=[0, 0],
                    y=[0, 0],
                    z=[-1.5, 1.5],
                    mode="lines+text",
                    line=dict(color="black", width=10),
                    name="z-axis",
                    # showlegend=True,
                    text=["", "z"],
                    textfont=dict(size=60),
                ),
            ]
            frames.append(
                go.Frame(
                    data=data,
                    name=f"{theta/pi:.02f} \u03C0",  # pi
                )
            )
        # To show surface at figure initialization
        fig = go.Figure(layout=_layout, frames=frames)
        f_theta = np.ones_like(f[: n**2].reshape(n, n)) * f_identity[j]
        fig.add_trace(
            go.Surface(
                x=x.reshape(n, n),
                y=y.reshape(n, n),
                z=z.reshape(n, n),
                surfacecolor=f_theta,
                colorscale=_cmap_bwr,
                cmin=f_min - 1e-3,
                cmax=f_max,
                opacity=1,
                name=f"Equivariance",
                showlegend=True,
            )
        )

        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines+text",
                line=dict(color="purple", width=30),
                name="Maximum equivariance",
                showlegend=True,
                textfont=dict(size=60, color="purple"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[-0.1, 0.1],
                y=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines+text",
                line=dict(color="darkgreen", width=30),
                name="Minimum equivariance",
                showlegend=True,
                textfont=dict(size=60, color="darkgreen"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines",
                line=dict(color="purple", width=30, dash="longdash"),
                name="Maximum equivariance complement",
                showlegend=True,
                textfont=dict(size=60),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                y=[-0.1, 0.1],
                x=[-0.1, 0.1],
                z=[-0.1, 0.1],
                mode="lines",
                line=dict(color="darkgreen", width=30, dash="longdash"),
                name="Minimum equivariance complement",
                showlegend=True,
                textfont=dict(size=60),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[-1.5, 1.5],
                y=[0, 0],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="x-axis",
                # showlegend=True,
                text=["", "x"],
                textfont=dict(size=60),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[-1.5, 1.5],
                z=[0, 0],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="y-axis",
                # showlegend=True,
                text=["", "y"],
                textfont=dict(size=60),
            ),
        )
        fig.add_trace(
            go.Scatter3d(
                x=[0, 0],
                y=[0, 0],
                z=[-1.5, 1.5],
                mode="lines+text",
                line=dict(color="black", width=10),
                name="z-axis",
                # showlegend=True,
                text=["", "z"],
                textfont=dict(size=60),
            ),
        )
        sliders = [
            {
                "pad": {"b": 10, "t": 60},
                "len": 1.5,
                "x": 0.1,
                "y": 0,
                "steps": [
                    {
                        "args": [
                            [f.name],
                            {
                                "frame": {"duration": 0},
                                "mode": "immediate",
                                "fromcurrent": True,
                                "transition": {"duration": 0, "easing": "linear"},
                            },
                        ],
                        "label": str(f.name),
                        "method": "animate",
                    }
                    for f in fig.frames
                ],
            }
        ]

        fig.update_layout(
            sliders=sliders,
            legend=dict(
                yanchor="top", y=1.59, xanchor="left", x=0.01, title_text=label
            ),
        )
        figs.append(fig)
    return figs


def log_learned_equivariance(irrepmaps, epoch, run_name, sphere=False):
    layer_ids = set()
    layer_ids = set([irrepmap.layer_id for irrepmap in irrepmaps])
    if list(layer_ids) == []:
        return

    layer_offset = min(layer_ids)
    n = 100
    for irrepmap in irrepmaps:
        layer_id = irrepmap.layer_id
        (prob_fn, g_elements, data) = irrepmap.get_distribution(n=n, sphere=sphere)
        if data is not None and sphere:
            figs = plot_signal(data)
            for i, fig in enumerate(figs):
                try:
                    os.makedirs("temp/", exist_ok=True)
                except FileNotFoundError:
                    pass
                except FileExistsError:
                    pass
                reflection = " reflection" if i else ""
                table = wandb.Table(columns=["test plots"])
                path_to_plotly = f"temp/{layer_id}{i}.html"
                fig.write_html(path_to_plotly, auto_play=False)
                table.add_data(wandb.Html(path_to_plotly))
                wandb.log(
                    {
                        f"Learned degree of equivariance layer {layer_id - layer_offset} 3-Sphere{reflection} plot": table
                    },
                    step=epoch,
                )

                # reflection = " reflection" if i else ""
                # wandb.log(
                #     {
                #         f"Learned degree of equivariance layer {layer_id - layer_offset} 3-Sphere{reflection}": fig
                #     },
                #     step=epoch,
                # )
        plt.title(f"Degree of equivariance layer {layer_id - layer_offset}")
        plt.ylabel("likelihood of h")
        plt.xlabel("groupelement g")
        for angle_id in range(prob_fn.shape[0]):
            angle_name = f" angle: {angle_id}" if prob_fn.shape[0] > 1 else ""
            data = [
                [g_element, prob, layer_id, run_name, angle_id]
                for (g_element, prob) in zip(g_elements[angle_id], prob_fn[angle_id])
            ]

            table = wandb.Table(
                data=data,
                columns=[
                    "transformation element",
                    "equivariance degree",
                    "layer_id",
                    "run_name",
                    "angle_id",
                ],
            )
            wandb.log(
                {
                    f"Learned Degree of equivariance layer {layer_id - layer_offset}{angle_name}": wandb.plot.line(
                        table,
                        "transformation element",
                        "equivariance degree",
                        title=f"Degree of equivariance layer {layer_id - layer_offset}{angle_name}",
                    )
                },
                step=epoch,
            )
            plt.plot(g_elements[angle_id], prob_fn[angle_id], label=angle_name)

        if prob_fn.shape[0] > 1:
            plt.legend()

        plt.ylim([0, 1.5])

        wandb.log(
            {
                f"gradual Degree of equivariance layer {layer_id - layer_offset}": wandb.Image(
                    plt
                )
            },
            step=epoch,
        )
        plt.close()


def get_irrepsmaps(model):
    irrepmaps = list(
        filter(lambda x: isinstance(x, IrrepsMapFourierBLact), model.modules())
    )
    return irrepmaps


def get_kl_loss(irrepmaps):
    irrepmaps = {irrepmap.layer_id: irrepmap for irrepmap in irrepmaps}
    kl_loss_sum = 0
    kl_uniform = 0
    for layer_id, irrepmap in irrepmaps.items():
        if layer_id == 0 or (layer_id - 1) not in irrepmaps:
            kl_uniform += irrepmap.kl_divergence(None).squeeze() / len(irrepmaps)
        else:
            kl_loss_sum += irrepmap.kl_divergence(
                irrepmaps[layer_id - 1]
            ).squeeze() / len(irrepmaps)
        # print(kl_loss)
    return kl_loss_sum, kl_uniform


def get_shift_loss(irrepsmaps):
    loss = 0
    for irrepmap in irrepsmaps:
        if hasattr(irrepmap, "shift_loss"):
            loss += irrepmap.shift_loss.flatten()[0]

    return loss


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


class Dataset(data.Dataset):
    def __init__(
        self,
        input_length,
        mid,
        output_length,
        direc,
        task_list,
        sample_list,
        stack=False,
    ):
        self.input_length = input_length
        self.mid = mid
        self.output_length = output_length
        self.direc = direc
        self.task_list = task_list
        self.sample_list = sample_list
        self.stack = stack
        # print(task_list)
        try:
            self.data_lists = [
                torch.load(
                    self.direc + "/raw_data_" + str(idx[0]) + "_" + str(idx[1]) + ".pt"
                )
                for idx in task_list
            ]
        except:
            self.data_lists = [
                torch.load(self.direc + "/raw_data_" + str(idx) + ".pt")
                for idx in task_list
            ]

    def __len__(self):
        return len(self.task_list) * len(self.sample_list)

    def __getitem__(self, index):
        task_idx = index // len(self.sample_list)
        sample_idx = index % len(self.sample_list)
        y = self.data_lists[task_idx][
            (self.sample_list[sample_idx] + self.mid) : (
                self.sample_list[sample_idx] + self.mid + self.output_length
            )
        ]
        if not self.stack:
            x = self.data_lists[task_idx][
                (self.mid - self.input_length + self.sample_list[sample_idx]) : (
                    self.mid + self.sample_list[sample_idx]
                )
            ]
        else:
            x = self.data_lists[task_idx][
                (self.mid - self.input_length + self.sample_list[sample_idx]) : (
                    self.mid + self.sample_list[sample_idx]
                )
            ].reshape(-1, y.shape[-2], y.shape[-1])
        return x.float(), y.float()

def rotate_and_reorient(xx, k):
    B,C,H,W = xx.shape
    T = C // 2
    frames = xx.view(B, T, 2, H, W)
    # spatial rotation:
    frames = torch.rot90(frames, k=k, dims=(3,4))
    ux, uy = frames[:,:,0], frames[:,:,1]
    if   k==0: ux2,uy2 = ux,    uy
    elif k==1: ux2,uy2 = -uy,   ux
    elif k==2: ux2,uy2 = -ux,  -uy
    elif k==3: ux2,uy2 =  uy,  -ux
    frames = torch.stack([ux2, uy2], dim=2)
    return frames.view(B, C, H, W)

def augment_data(xx):
    """
    Augment each input with a random rotation.
    """
    covariates = np.random.randint(0, 4, size=xx.shape[0])
    xx_rotated = []
    for i, c in enumerate(covariates):
        xx_rotated.append(rotate_and_reorient(xx[i].unsqueeze(0), c))
    return torch.stack(xx_rotated, dim=0).squeeze(1), covariates

import torch

def apply_D4(xx, k, flip):
    """
    Apply g = (rotation by k*90° CCW, optional horizontal flip)
    to a batch xx of shape [B, 2, H, W] (one timestep, 2-vector channels).
    Returns the transformed [B,2,H,W].
    """
    # 1) Maybe flip horizontally (axis -1)
    if flip:
        xx = torch.flip(xx, dims=[-1])  # reflect across vertical center

    # 2) Rotate CCW by k*90° on the spatial grid
    xx = torch.rot90(xx, k=k, dims=(2,3))

    # 3) Reorient the vector components by R_g = (flip ? F : I) @ R_k
    #    Build R_k first:
    if   k == 0:  Rk = torch.tensor([[1,0],[0,1]], device=xx.device, dtype=xx.dtype)
    elif k == 1:  Rk = torch.tensor([[0,-1],[1, 0]], device=xx.device, dtype=xx.dtype)
    elif k == 2:  Rk = torch.tensor([[-1,0],[0,-1]], device=xx.device, dtype=xx.dtype)
    elif k == 3:  Rk = torch.tensor([[ 0,1],[-1,0]], device=xx.device, dtype=xx.dtype)

    if flip:
        F = torch.tensor([[-1,0],[0,1]], device=xx.device, dtype=xx.dtype)
        Rg = F @ Rk
    else:
        Rg = Rk

    # 4) Apply Rg to the 2-vector at every pixel:
    #    xx has shape [B,2,H,W]; we want Rg @ [u_x;u_y] at each (b,h,w).
    B, C, H, W = xx.shape
    flat = xx.reshape(B, 2, -1)          # [B,2, H*W]
    mixed = Rg @ flat                 # [2,2] @ [B,2,N] -> [B,2,N]
    return mixed.view(B, 2, H, W)

def rotate_and_reorientd4(xx, covariates):
    """
    xx:           [B, 20, H, W]  = 10×2 channels
    covariates:   list of (k,flip) for each sample in batch
    returns:      xx_trans of same shape
    """
    B, C, H, W = xx.shape
    T = C // 2
    out = torch.zeros_like(xx)

    for i, g in enumerate(covariates):
        k = g % 4
        flip = g >= 4
        # split into [2,H,W], apply D4, then re-stack
        frames = xx[i].view(T, 2, H, W)     # [T,2,H,W]
        transformed = apply_D4(frames, k, flip)  # [T,2,H,W]
        out[i] = transformed.view(2*T, H, W)
    return out

def augment_data_D4(xx):
    covs = np.random.randint(0, 8, size=xx.shape[0])
    xx_trans = rotate_and_reorientd4(xx, covs)
    return xx_trans, covs




def get_transformed_latent(latents, covariate):
        flattened_latents = [l.reshape(l.shape[0], -1) for l in latents]
        transformed = [torch.zeros_like(l, device=device)
                       for l in flattened_latents]
        # for each unique group element
        for g in np.unique(covariate).tolist():
            mask = covariate == g
            for idx, l in enumerate(flattened_latents):
                W_g = W_repr(g, d=l.size(-1))
                # apply linear transform on masked entries
                transformed[idx][mask] = F.linear(l[mask], W_g)
        return transformed

def get_transformation_loss(
        transformed_latents: List[torch.Tensor],
        latents2: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Sum MSE loss over corresponding latents.
        """
        losses = [F.mse_loss(t, l2)
                  for t, l2 in zip(transformed_latents, latents2)]
        return sum(losses)

def rotate_and_mix_latent(lat: torch.Tensor, covariate: torch.Tensor, reorient=False):
    """
    lat:        [B, C, H, W]  the features E(x)
    covariate:  [B] with values in {0,1,2,3} for rotation k*90°
    returns:    lat_trans: [B, C, H, W] implementing g·lat = W(g)[ rot(lat) ]
    """
    lat = lat[0]
    B, C, H, W = lat.shape
    lat_trans = torch.zeros_like(lat)

    # 1) rotate spatially just like for the input
    #    this does both pixel-rotation and vector reorientation if needed;
    #    here lat is a scalar field per channel, so no vector‐reorientation step
    #    just pure rotation:
    #    torch.rot90 rotates CCW by k*90° about dims (2,3)
    # we’ll do this per-sample because each may have a different k
    for k in np.unique(covariate):
        mask = (covariate == k)
        if not mask.any(): continue

        # a) spatially rotate only those batch entries
        lat_sub = lat[mask]                 # [n, C, H, W]
        if reorient:
            # reorient the latent vector
            lat_rot = rotate_and_reorient(lat_sub, k)
        else:
            lat_rot = torch.rot90(lat_sub, k=k, dims=(2,3))

        # b) channel‐mix them
        W_g = W_repr(k, d=C)  # [C,C]
        # flatten spatial dims for matmul
        mixed = torch.einsum('ij,bjhw->bihw', W_g, lat_rot)
        lat_trans[mask] = mixed

    return lat_trans

def rotate_latents(lat: torch.Tensor, covariate: torch.Tensor, reorient=False):
    """
    lat:        [B, C, H, W]  the features E(x)
    covariate:  [B] with values in {0,1,2,3} for rotation k*90°
    returns:    lat_trans: [B, C, H, W] implementing g·lat = W(g)[ rot(lat) ]
    """
    lat = lat[0]
    B, C, H, W = lat.shape
    lat_trans = torch.zeros_like(lat)

    # 1) rotate spatially just like for the input
    #    this does both pixel-rotation and vector reorientation if needed;
    #    here lat is a scalar field per channel, so no vector‐reorientation step
    #    just pure rotation:
    #    torch.rot90 rotates CCW by k*90° about dims (2,3)
    # we’ll do this per-sample because each may have a different k
    for k in np.unique(covariate):
        mask = (covariate == k)
        if not mask.any(): continue

        # a) spatially rotate only those batch entries
        lat_sub = lat[mask]                 # [n, C, H, W]
        if reorient:
            lat_rot = rotate_and_reorient(lat_sub, k)
        else:
            lat_rot = torch.rot90(lat_sub, k=k, dims=(2,3))

        lat_trans[mask] = lat_rot

    return lat_trans

def mix_channels(lat: torch.Tensor, covariate: torch.Tensor):
    """
    lat:        [B, C, H, W]  the features E(x)
    covariate:  [B] with values in {0,1,2,3} for rotation k*90°
    returns:    lat_trans: [B, C, H, W] implementing g·lat = W(g)[ rot(lat) ]
    """
    lat = lat[0]
    B, C, H, W = lat.shape
    lat_trans = torch.zeros_like(lat)

    # 1) rotate spatially just like for the input
    #    this does both pixel-rotation and vector reorientation if needed;
    #    here lat is a scalar field per channel, so no vector‐reorientation step
    #    just pure rotation:
    #    torch.rot90 rotates CCW by k*90° about dims (2,3)
    # we’ll do this per-sample because each may have a different k
    for k in np.unique(covariate):
        mask = (covariate == k)
        if not mask.any(): continue

        # a) spatially rotate only those batch entries
        lat_sub = lat[mask]                 # [n, C, H, W]
        W_g = W_repr(k, d=C)  # [C,C]
        mixed = torch.einsum('ij,bjhw->bihw', W_g, lat_sub)
        lat_trans[mask] = mixed

    return lat_trans

def d4_latent(lat: torch.Tensor, covariate: torch.Tensor):
    """
    lat:        [B, C, H, W]  the features E(x)
    covariate:  [B] {0:1, 1:r, 2:r^2, 3:r^3, 4:s, 5:r s, 6:r^2 s, 7:r^3 s}"""

    lat = lat[0]
    B, C, H, W = lat.shape
    lat_trans = torch.zeros_like(lat)

    for k in np.unique(covariate):
        mask = (covariate == k)
        if not mask.any(): continue
        lat_sub = lat[mask] # [n, C, H, W]

        if k >= 4:
            # horizontal flip 
            lat_sub = lat_sub.flip(3)
        lat_rot = torch.rot90(lat_sub, k=(k*90)%360, dims=(2,3))

        lat_trans[mask] = lat_rot

    return lat_trans




def train_epoch(
    train_loader,
    model,
    optimizer,
    loss_function,
    model_type,
    irrepmaps=None,
    loss_factors=[5, 3, 3],
    lambda_t = 0.05,
    latent_action=None,
    reorient=False,
):
    model.train()
    train_mse = []
    equiv_loss = []
    for xx, yy in train_loader:
        B, T, _, H, W = yy.shape
        xx, yy = xx.to(device), yy.to(device)

        loss_pred = 0.0
        # — Equivariance on a per-sample rotation —
        if lambda_t > 0:
            with torch.no_grad():
                xx_rot, covariates = augment_data(xx)      # each sample has its own k
                #xx_rot, covariates = augment_data_D4(xx)  # each sample has its own (k,flip)
                yy_rot = []
                for i, c in enumerate(covariates):
                   yy_rot.append(rotate_and_reorient(yy[i].view(T*2,H,W).unsqueeze(0), c).view(T,2,H,W))
                yy_rot = torch.stack(yy_rot, dim=0).squeeze(1)
                #yy_rot = rotate_and_reorientd4(yy.view(B,2*T,H,W), covariates).view(B,T,2,H,W)

            _, lat        = model(xx,     [3])
            y0_pred_rot, lat_rot    = model(xx_rot, [3])
            if latent_action == "flatten":
                lat_trans     = get_transformed_latent(lat, covariates)
                lat_trans    = [l.view(l.shape) for l in lat]
            elif latent_action == "channelsandgrid":
                lat_trans     = rotate_and_mix_latent(lat, covariates, reorient)
            elif latent_action == "grid":
                lat_trans     = rotate_latents(lat, covariates, reorient)
            elif latent_action == "channels":
                lat_trans     = mix_channels(lat, covariates)
            elif latent_action == "d4_grid":
                lat_trans     = d4_latent(lat, covariates)
            else:
                raise ValueError(f"Unknown latent action: {latent_action}")
            loss_equiv    = get_transformation_loss(lat_trans, lat_rot)
            yy_true_rot = yy_rot.transpose(0,1)[0]
            loss_pred    += 0.5*loss_function(y0_pred_rot, yy_true_rot)
        else:
            loss_equiv = torch.tensor(0.0)
            #D4---------------------
            #xx_rot, covariates = augment_data_D4(xx)  # each sample has its own (k,flip)
            #yy_rot = rotate_and_reorientd4(yy.view(B,2*T,H,W), covariates).view(B,T,2,H,W)
            #D4---------------------
            xx_rot, covariates = augment_data(xx)      # each sample has its own k
            #C4---------------------
            #xx_rot, covariates = augment_data_D4(xx)  # each sample has its own (k,flip)
            yy_rot = []
            for i, c in enumerate(covariates):
                yy_rot.append(rotate_and_reorient(yy[i].view(T*2,H,W).unsqueeze(0), c).view(T,2,H,W))
            yy_rot = torch.stack(yy_rot, dim=0).squeeze(1)
            #C4---------------------
            y0_pred_rot, _    = model(xx_rot, [3])
            yy_true_rot = yy_rot.transpose(0,1)[0]
            loss_pred    += 0.5*loss_function(y0_pred_rot, yy_true_rot)


        # — Autoregressive prediction loss —
        xx_roll = xx.clone()
        for i, y_true in enumerate(yy.transpose(0,1)):
            y_pred, _ = model(xx_roll, [4])
            if i == 0:
                loss_pred += 0.5*loss_function(y_pred, y_true)
            else:
                loss_pred += loss_function(y_pred, y_true)
            xx_roll    = torch.cat([xx_roll[:,2:], y_pred], dim=1)
        train_mse.append((loss_pred.item()/yy.shape[1]))
        equiv_loss.append(loss_equiv.item())

        # — Weight‐regularizers —
        loss = loss_pred + lambda_t * loss_equiv
        if model_type == "RSteer":
            wc = model.module.get_weight_constraint()
            loss += loss_function(wc, torch.zeros_like(wc))
        elif model_type == "ENCNN":
            kl_loss, kl_u = get_kl_loss(irrepmaps)
            shift_loss    = get_shift_loss(irrepmaps)
            loss += (
                loss_factors[0]*shift_loss +
                loss_factors[1]*kl_loss   +
                loss_factors[2]*kl_u
            )

        # — Backprop & step —
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_rmse = round(np.sqrt(np.mean(train_mse)), 5)
    equiv_loss = np.mean(equiv_loss)
    return train_rmse, equiv_loss
    


def eval_epoch(valid_loader, model, loss_function):
    model.eval()
    valid_mse = []
    preds = []
    trues = []
    with torch.no_grad():
        for xx, yy in valid_loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0
            ims = []
            for y in yy.transpose(0, 1):
                im, _ = model(xx)
                xx = torch.cat([xx[:, im.shape[1] :], im], 1)
                loss += loss_function(im, y)
                ims.append(im.cpu().data.numpy())
            ims = np.array(ims).transpose(1, 0, 2, 3, 4)
            preds.append(ims)
            trues.append(yy.cpu().data.numpy())
            valid_mse.append(loss.item() / yy.shape[1])
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        valid_rmse = round(np.sqrt(np.mean(valid_mse)), 5)
    return valid_rmse, preds, trues


def RPPConv_L2(mdl, conv_wd=1e-6, basic_wd=1e-6):
    conv_l2 = 0.0
    basic_l2 = 0.0
    for block in mdl.model:
        if hasattr(block, "conv"):
            conv_l2 += sum([p.pow(2).sum() for p in block.conv.parameters()])
        if hasattr(block, "linear"):
            basic_l2 += sum([p.pow(2).sum() for p in block.linear.parameters()])

    return conv_wd * conv_l2 + basic_wd * basic_l2


def RPPConv_L1(mdl, conv_wd=1e-6, basic_wd=1e-6):
    conv_l1 = 0.0
    basic_l1 = 0.0
    for block in mdl.model:
        if hasattr(block, "conv"):
            conv_l1 += sum([p.abs().sum() for p in block.conv.parameters()])
        if hasattr(block, "linear"):
            basic_l1 += sum([p.abs().sum() for p in block.linear.parameters()])

    return conv_wd * conv_l1 + basic_wd * basic_l1
