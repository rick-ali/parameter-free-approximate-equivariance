from typing import Tuple
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms.functional as  TF
from torchvision.transforms.functional import InterpolationMode
import medmnist
import numpy as np
from medmnist import INFO
import pytorch_lightning as pl
import PIL
import torch
import random
import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from DDMNIST_dataset import DDMNIST

class PairedC4xC4DDMNIST(Dataset):
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
    

    def augment_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
        code = random.randint(0, 3)
        x = img

        # Introduce noise
        angle_small = np.random.uniform(-180, 180)
        x = TF.rotate(x, float(angle_small), InterpolationMode.BILINEAR)

        if   code == 0: net_rot =   0
        elif code == 1: net_rot =  90
        elif code == 2: net_rot = 180
        elif code == 3: net_rot = 270

        correction = net_rot# - angle_small
        x = TF.rotate(x, correction, InterpolationMode.BILINEAR)
        
        return x, code
    

    

    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train' or self.split == 'val' or self.augment_test:
            img1, img2, y = self.data.__getitem__(idx)
            augmented_img1, _ = self.augment_image(img1)
            augmented_img2, _ = self.augment_image(img2)
            transformed_img1, covariate1 = self.augment_image(augmented_img1)
            transformed_img2, covariate2 = self.augment_image(augmented_img2)
            covariate = 4*covariate1 + covariate2
            
            x1 = self.data._combine_images(augmented_img1[0], augmented_img2[0])
            x2 = self.data._combine_images(transformed_img1[0], transformed_img2[0])

            return (x1, y), (x2, y), transformation_type, covariate
        
        else:
            img1, img2, y = self.data.__getitem__(idx)
            transformed_img1, covariate1 = self.augment_image(img1)
            transformed_img2, covariate2 = self.augment_image(img2)
            covariate = 4*covariate1 + covariate2

            x1 = self.data._combine_images(img1[0], img2[0])
            x2 = self.data._combine_images(transformed_img1[0], transformed_img2[0])

            return (x1, y), (x2, y), transformation_type, covariate


class C4xC4DDMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, augment_test=False):
        super().__init__()
        self.batch_size = batch_size
        self.augment_test = augment_test
        

    def setup(self, stage=None):
        train_data = DDMNIST(train=True, images_per_class=100)
        self.train_dataset = PairedC4xC4DDMNIST(train_data, split='train')

        val_data = DDMNIST(train=False, images_per_class=20)
        self.val_dataset = PairedC4xC4DDMNIST(val_data, split='val')

        test_data = DDMNIST(train=False, images_per_class=50)
        self.test_dataset = PairedC4xC4DDMNIST(test_data, split='test', augment_test=self.augment_test)


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)
    

if __name__ == '__main__':
    data_module = C4xC4DDMNISTDataModule(256)
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
    