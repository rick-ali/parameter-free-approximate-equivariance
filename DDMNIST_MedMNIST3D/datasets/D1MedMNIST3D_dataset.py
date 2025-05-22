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
from utils.representations import D1RegularRepresentation

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils3D.utils import Transform3D


class PairedD1MedMNIST3D(Dataset):
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
    

    def augment_3d_d1(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
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

        idx = random.randint(0, 1)

        if idx == 1:
            # R = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=torch.float32)
            # theta = torch.zeros((1, 3, 4), device=device, dtype=dtype)
            # theta[0, :3, :3] = R
            # # Generate grid
            # grid = F.affine_grid(theta, size=(1, C, D, H, W), align_corners=True)
            # # Sample
            # aug = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=True)

            # Flip image across the z-axis 
            # aug = img.flip(dims=[2])

            aug = -img
        if idx == 0:
            aug = img

        # Remove batch dim if needed
        aug = aug.squeeze(0)
        return aug, idx


    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train' or self.split == 'val':
            x1, y1 = self.data.__getitem__(idx)
            y1 = y1[0]
            augmented_X1, g1 = self.augment_3d_d1(x1)
            transformed_X2, covariate = self.augment_3d_d1(augmented_X1)
        
        else:
            x1, y1 = self.data.__getitem__(idx)
            y1 = y1[0]
            augmented_X1 = x1
            transformed_X2, covariate = self.augment_3d_d1(augmented_X1)

        return (augmented_X1, y1), (transformed_X2, y1), transformation_type, covariate



class D1MedMNISTDataModule(pl.LightningDataModule):
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

        self.train_dataset = PairedD1MedMNIST3D(train_data, split='train')
        self.val_dataset = PairedD1MedMNIST3D(val_data, split='val')
        self.test_dataset = PairedD1MedMNIST3D(test_data, split='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=48, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=48, shuffle=False, num_workers=15)
    

if __name__ == '__main__':
    
    
    data_module = D1MedMNISTDataModule('organmnist3d', 128, resize=False, as_rgb=True, size=28, download=False)
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
    