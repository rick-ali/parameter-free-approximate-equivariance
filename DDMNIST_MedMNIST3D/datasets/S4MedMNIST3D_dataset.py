from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as  TF
import medmnist
from medmnist import INFO
import pytorch_lightning as pl
import PIL
import torch
import random
import torch.utils.data as data
import torch.nn.functional as F
from utils.representations import OctahedralRegularRepresentation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils3D.utils import Transform3D


class PairedS4MedMNIST3D(Dataset):
    def __init__(self, data, split, test_all_rotations=False):
        """
        Args:
        data: MedMNIST dataset object. In particular, this class assumes that the dataset object has the following attributes:
            - imgs: List of PIL images
            - labels: List of labels
        transform: Transform to apply to the images
        Split: 'train', 'val', or 'test'
        x2_angle: Angle to rotate the second image by
        """
        self.data = data
        self.split = split
        self.test_all_rotations = test_all_rotations


    def __len__(self):
        """
        If self.test_all_rotations is True, for val and test we return the number of images in the dataset times the number of rotations
        Otherwise, we return the number of images
        """
        if self.test_all_rotations:
            if self.split == 'train':
                return len(self.data.imgs)
            else:
                return len(self.data.imgs) * len(self.rotation_indices)
        else:
            return len(self.data.imgs)
    
    def flip(self, R: torch.Tensor) -> int:
        """
        Decide whether to flip label or not based on the rotation matrix.
        If flip == -1, then the label is flipped.
        If flip == 1, then the label is not flipped.
        """
        e1 = torch.tensor([1, 0, 0], dtype=torch.float32)
        e2 = torch.tensor([0, 1, 0], dtype=torch.float32)
        e3 = torch.tensor([0, 0, 1], dtype=torch.float32)
        e1_rotated = R @ e1
        e2_rotated = R @ e2
        e3_rotated = R @ e3
        flip = e1_rotated[e1_rotated != 0] * e2_rotated[e2_rotated != 0] * e3_rotated[e3_rotated != 0]
        return flip.item()
    
    def flip_label(self, label: int, flip: int) -> int:
        """
        Flip the label based on the flip value.
        If flip == -1, then the label is flipped.
        If flip == 1, then the label is not flipped.
        """
        labels = {'0': 'liver',
        '1': 'kidney-right',
        '2': 'kidney-left',
        '3': 'femur-right',
        '4': 'femur-left',
        '5': 'bladder',
        '6': 'heart',
        '7': 'lung-right',
        '8': 'lung-left',
        '9': 'spleen',
        '10': 'pancreas'}
        flip_map = {
            0: 0,
            1: 2,
            2: 1,
            3: 4,
            4: 3,
            5: 5,
            6: 6,
            7: 8,
            8: 7,
            9: 9,
            10: 10
        }
        if flip == -1:
            return flip_map[label]
        else:
            return label
        
    def should_swap_labels(self, R, tol=1e-6):
        """
        Return True if we need to swap left/right organ labels under rotation R.
        We swap if either:
        - det(R) < 0  (improper rotation, mirror)
        - R is a 180° rotation about the z-axis (x→-x, y→-y, z→z)
        """
        # 1) chirality test
        if torch.det(R) < 0:
            return True

        # 2) 180° about z test: check axis images
        I = torch.eye(3, dtype=R.dtype, device=R.device)
        ex, ey, ez = I[:,0], I[:,1], I[:,2]
        ex_r, ey_r, ez_r = R @ ex, R @ ey, R @ ez

        is_half_turn_z = (
            torch.allclose(ex_r, -ex, atol=tol) and
            torch.allclose(ey_r, -ey, atol=tol) and
            torch.allclose(ez_r,  ez, atol=tol)
        )
        return is_half_turn_z

    def augment_3d_octahedral(self, img: torch.Tensor, label: int) -> Tuple[torch.Tensor, int]:
        """
        Apply a random octahedral group rotation to a 3D image.

        Args:
            img: Tensor of shape [1, D, H, W] or [C, D, H, W] with C=1 (assumes single-channel volume).
        Returns:
            aug: Rotated image tensor with same shape as input.
            idx: Index in 0..23 of the chosen rotation.
        """
        # Ensure img has shape [N, C, D, H, W]
        if img.dim() == 4:
            # assume [C, D, H, W]
            img = img.unsqueeze(0)  # [1, C, D, H, W]
        elif img.dim() == 5:
            pass  # already batched
        else:
            raise ValueError(f"Expected 4D or 5D tensor, got shape {img.shape}")

        N, C, D, H, W = img.shape
        if N != 1:
            raise ValueError("Batch size >1 not supported currently")

        device = img.device
        dtype = img.dtype

        # Initialize octahedral representation
        oct_rep = OctahedralRegularRepresentation(device=device, dtype=dtype)

        # Sample random rotation index
        idx = random.randrange(oct_rep.order)  # 0..23
        R = oct_rep.rotations[idx]  # 3x3 rotation matrix
        # Build 3x4 affine matrix for F.affine_grid
        # theta maps normalized coordinates: shape [N, 3, 4]
        # Since rotation about center in normalized coords, no translation needed
        theta = torch.zeros((1, 3, 4), device=device, dtype=dtype)
        theta[0, :3, :3] = R

        # if self.data.flag == 'organmnist3d':
        #     flip = self.flip(R)
        #     # flip = torch.linalg.det(R).item()
        #     # ex_rot = R @ torch.tensor([1, 0, 0], device=device, dtype=dtype)
        #     # flip = -1 if torch.allclose(ex_rot, torch.tensor([-1.,0.,0.])) else 1
        #     # flip = -1 if self.should_swap_labels(R) else 1
        #     label = self.flip_label(label, flip)

        # Generate grid
        grid = F.affine_grid(theta, size=(1, C, D, H, W), align_corners=True)
        # Sample
        aug = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Remove batch dim if needed
        aug = aug.squeeze(0)
        return aug, idx, label


    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train' or self.split == 'val':
            x1, y1 = self.data.__getitem__(idx)
            y1 = y1[0]
            augmented_X1, g1, y1 = self.augment_3d_octahedral(x1, y1)
            transformed_X2, covariate, y2 = self.augment_3d_octahedral(augmented_X1, y1)
        
        else:
            if self.test_all_rotations:
                img_idx = idx // 2
                x1, y1 = self.data.__getitem__(img_idx)
                flipping_idx = idx % 2

                flip_x1 = [True,False][flipping_idx]
                augmented_X1 = TF.hflip(x1) if flip_x1 else x1
                
                covariate = 1
                transformed_X2 = TF.hflip(augmented_X1)

            else:
                x1, y1 = self.data.__getitem__(idx)
                y1 = y1[0]
                augmented_X1 = x1
                transformed_X2, covariate, y2 = self.augment_3d_octahedral(augmented_X1, y1)

        return (augmented_X1, y1), (transformed_X2, y2), transformation_type, covariate



class S4MedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_flag, batch_size, resize, as_rgb, size, download):
        super().__init__()
        self.data_flag = data_flag
        self.batch_size = batch_size
        self.resize = resize
        self.as_rgb = as_rgb
        self.size = size
        self.download = download
        self.info = INFO[data_flag]
        self.DataClass = getattr(medmnist, self.info['python_class'])

    def setup(self, stage=None):
        
        # Original MedMNIST3D transform
        # shape_transform = True if self.data_flag in ['adrenalmnist3d', 'vesselmnist3d'] else False
        # train_transform = Transform3D(mul='random') if shape_transform else Transform3D()
        # eval_transform = Transform3D(mul='0.5') if shape_transform else Transform3D()
        # Approximate Equivariance paper transform
        train_transform = eval_transform = transforms.Compose(
            [
                lambda x: torch.FloatTensor(x),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        
        train_data = self.DataClass(split='train', transform=train_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        val_data = self.DataClass(split='val', transform=eval_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        test_data = self.DataClass(split='test', transform=eval_transform, download=self.download, as_rgb=self.as_rgb, size=self.size)

        self.train_dataset = PairedS4MedMNIST3D(train_data, split='train')
        self.val_dataset = PairedS4MedMNIST3D(val_data, split='val')
        self.test_dataset = PairedS4MedMNIST3D(test_data, split='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=48, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=48, shuffle=False, num_workers=15)
    

if __name__ == '__main__':
    
    
    data_module = S4MedMNISTDataModule('organmnist3d', 128, resize=False, as_rgb=True, size=28, download=False)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(len(train_loader), len(val_loader), len(test_loader))
    for batch in train_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    for batch in val_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    for batch in test_loader:
        (x1, y1), (x2, y2), transformation_type, covariate = batch
        print(x1.shape, y1.shape, x2.shape, y2.shape)#, transformation_type, covariate)
        break
    