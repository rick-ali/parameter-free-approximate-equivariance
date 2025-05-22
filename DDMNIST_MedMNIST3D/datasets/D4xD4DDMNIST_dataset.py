from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as  TF
from torchvision.transforms.functional import InterpolationMode
import numpy as np
import medmnist
from medmnist import INFO
import pytorch_lightning as pl
import PIL
import torch
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from DDMNIST_dataset import DDMNIST

class PairedD4xD4DDMNIST(Dataset):
    def __init__(self, data, split, test_all_rotations=False, augment_test=False):
        """
        Args:
        data: DDMNIST dataset object. In particular, this class assumes that the dataset object has the following attributes:
            - imgs: List of PIL images
            - labels: List of labels
        transform: Transform to apply to the images
        Split: 'train', 'val', or 'test'
        x2_angle: Angle to rotate the second image by
        """
        self.data = data
        self.split = split
        self.test_all_rotations = test_all_rotations
        self.augment_test = augment_test


    def __len__(self):
        """
        If self.test_all_rotations is True, for val and test we return the number of images in the dataset times the number of rotations
        Otherwise, we return the number of images
        """
        return len(self.data.labels)
    

    def transform_d4(self, img: torch.Tensor, code: int) -> torch.Tensor:
        """
        Apply one of the 8 D4 transforms *plus* a random small rotation + correction:
          0: identity
          1: r   (90° CCW)
          2: r^2 (180°)
          3: r^3 (270° CCW)
          4: s   (horizontal flip)
          5: r s
          6: r^2 s
          7: r^3 s
        """
        x = img
        # 4) do the flip if needed
        if code >= 4:
            x = TF.hflip(x)

        # 1) sample a random “small” angle
        angle_small = np.random.uniform(-180, 180)

        # 2) apply that first
        x = TF.rotate(x, float(angle_small), InterpolationMode.BILINEAR)

        # 3) figure out the D4 code’s “net rotation”
        #    (0, 90, 180, or 270)
        if   code == 0: net_rot =   0
        elif code == 1: net_rot =  90
        elif code == 2: net_rot = 180
        elif code == 3: net_rot = 270
        elif code == 4: net_rot =   0
        elif code == 5: net_rot =  90
        elif code == 6: net_rot = 180
        elif code == 7: net_rot = 270
        else: 
            raise ValueError(f"Invalid D4 code: {code}")


        # 5) apply the “correction” rotation so that
        #    angle_small + (net_rot - angle_small) = net_rot
        correction = net_rot - angle_small
        x = TF.rotate(x, float(correction), InterpolationMode.BILINEAR)

        return x

    def augment_d4x_d4(self, img1: torch.Tensor, img2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Randomly sample an element of D4 x D4 and apply its action to two images separately.

        Returns:
            aug1: transformed img1.
            aug2: transformed img2.
            idx: flat index in 0..63 corresponding to (g,h).
        """
        # sample flat index
        idx = random.randint(0, 63)
        # decompose into (g,h)
        g, h = divmod(idx, 8)
        # apply D4 actions independently
        aug1 = self.transform_d4(img1, g)
        aug2 = self.transform_d4(img2, h)
        return aug1, aug2, idx
    
    

    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train' or self.split == 'val' or self.augment_test:
            img1, img2, y = self.data.__getitem__(idx)
            augmented_img1, augmented_img2, _ = self.augment_d4x_d4(img1, img2)
            #augmented_img1, _ = self.augment_image_c4(img1)
            #augmented_img2, _ = self.augment_image_c4(img2)
            #augmented_img1, augmented_img2  = img1, img2
            # angle_small = np.random.uniform(-180, 180)
            # augmented_img1 = TF.rotate(TF.rotate(img1, float(angle_small), InterpolationMode.BILINEAR), float(-angle_small), InterpolationMode.BILINEAR)
            # augmented_img2 = TF.rotate(TF.rotate(img2, float(angle_small), InterpolationMode.BILINEAR), float(-angle_small), InterpolationMode.BILINEAR)
            #augmented_img1, augmented_img2, _ = self.augment_d1x_d1(img1, img2)

            transformed_img1, transformed_img2, covariate = self.augment_d4x_d4(augmented_img1, augmented_img2)
            
            x1 = self.data._combine_images(augmented_img1[0], augmented_img2[0])
            x2 = self.data._combine_images(transformed_img1[0], transformed_img2[0])

            return (x1, y), (x2, y), transformation_type, covariate
        
        else:
            img1, img2, y = self.data.__getitem__(idx)
            transformed_img1, transformed_img2, covariate = self.augment_d4x_d4(img1, img2)
            # Add noise to img1 and img2 for consistency
            small_angle = np.random.uniform(-180, 180)
            img1 = TF.rotate(img1, float(small_angle), InterpolationMode.BILINEAR)
            img2 = TF.rotate(img2, float(small_angle), InterpolationMode.BILINEAR)
            img1 = TF.rotate(img1, float(-small_angle), InterpolationMode.BILINEAR)
            img2 = TF.rotate(img2, float(-small_angle), InterpolationMode.BILINEAR)
            
            x1 = self.data._combine_images(img1[0], img2[0])
            x2 = self.data._combine_images(transformed_img1[0], transformed_img2[0])

            return (x1, y), (x2, y), transformation_type, covariate


class D4xD4DDMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, augment_test=False):
        super().__init__()
        self.batch_size = batch_size
        self.augment_test = augment_test
        

    def setup(self, stage=None):
        train_data = DDMNIST(train=True, images_per_class=100)
        self.train_dataset = PairedD4xD4DDMNIST(train_data, split='train')

        val_data = DDMNIST(train=False, images_per_class=20)
        self.val_dataset = PairedD4xD4DDMNIST(val_data, split='val')

        test_data = DDMNIST(train=False, images_per_class=50)
        self.test_dataset = PairedD4xD4DDMNIST(test_data, split='test', augment_test=self.augment_test)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)
    

if __name__ == '__main__':
    data_module = D4xD4DDMNISTDataModule(256)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(f"Train dataset size: {len(data_module.train_dataset)}")
    print(f"Validation dataset size: {len(data_module.val_dataset)}")
    print(f"Test dataset size: {len(data_module.test_dataset)}")
    for i, ((x1, y1), (x2, y2), transformation_type, covariate) in enumerate(train_loader):
        #imshow(torchvision.utils.make_grid(x1))
        print(y1)
        #imshow(torchvision.utils.make_grid(x2))
        break
    