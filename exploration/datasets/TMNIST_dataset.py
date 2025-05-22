import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as  TF
import random
import pandas as pd

class PairedTMNISTDataset(Dataset):
    def __init__(self, root, train, complex=False, digits=None, x2_transformation='font', x2_angle=None):
        self.train = train
        self.complex = complex
        self.x2_transformation = x2_transformation

        if x2_transformation == 'rotation':
            assert x2_angle is not None
            self.x2_angle = x2_angle

        if digits is None:
            digits = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.digits = digits
        
        df = pd.read_csv(root)

        self.datafont1 = {}
        self.datafont2 = {}

        for digit in digits:
            df1 = df[(df['labels'] == digit) & (df['names'] == 'IBMPlexSans-MediumItalic')]
            df2 = df[(df['labels'] == digit) & (df['names'] == 'Bahianita-Regular')]

            X1 = df1.iloc[:,2:]
            self.datafont1[digit] = torch.tensor(np.array(X1).reshape(1, 28, 28)) / 255.0

            X2 = df2.iloc[:,2:]
            self.datafont2[digit] = torch.tensor(np.array(X2).reshape(1, 28, 28)) / 255.0
    

    def __len__(self):
        # Potentially infinite as there are only 2*n_digits base images and the dataset consists in augmentations. 
        return 10000  # Like MNIST
    
    
    def affine_transform(self, image, angle, scale, shear, x_displacement=0, y_displacement=0):
        # x,y displacement is useless if we use a CNN since it's translation invariant
        return TF.affine(image, angle, [x_displacement, y_displacement], scale, shear)

    def font__getitem__(self, idx):
        base_digit = random.choice(self.digits)
        x1 = self.datafont1[base_digit]
        font_1 = 0
        x2 = self.datafont2[base_digit]
        font_2 = 1

        swap_font = random.randint(0, 1)
        if swap_font == 1:
            x1, x2 = x2, x1
            font_1, font_2 = font_2, font_1

        angle = random.uniform(-90, 90)
        scale = random.uniform(0.8, 1.2)
        shear = 0 #random.uniform(-25, 25)
        transformed_X1 = self.affine_transform(x1, angle=angle, scale=scale, shear=shear)
        transformed_X2 = self.affine_transform(x2, angle=angle, scale=scale, shear=shear)

        return (transformed_X1, font_1), (transformed_X2, font_2)

    def rotation__getitem__(self, idx):
        base_digit = random.choice(self.digits)

        if random.randint(0, 1) == 1:
            x1 = self.datafont1[base_digit]
            x2 = self.datafont1[base_digit]
        else:
            x1 = self.datafont2[base_digit]
            x2 = self.datafont2[base_digit]

        angle = random.uniform(-180, 180)
        scale = random.uniform(0.8, 1.2)
        shear = 0 #random.uniform(-25, 25)
        x2_angle = (angle + 180 + self.x2_angle) % 360 - 180
        transformed_X1 = self.affine_transform(x1, angle=angle, scale=scale, shear=shear)
        transformed_X2 = self.affine_transform(x2, angle=x2_angle, scale=scale, shear=shear)

        return (transformed_X1, torch.tensor(base_digit)), (transformed_X2, torch.tensor(base_digit))

    def __getitem__(self, idx):
        if self.x2_transformation == 'font':
            return self.font__getitem__(idx)
        elif self.x2_transformation == 'rotation':
            return self.rotation__getitem__(idx)



class PairedTMNISTDataModule(pl.LightningDataModule):
    def __init__(self, root='./dataset/TMNIST_Data.csv', batch_size=32, num_workers=4, transform=transforms.ToTensor(),
                 val_split=0.1, complex=False, digits=None, x2_transformation='font', x2_angle=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split  # Fraction of the training set to use for validation
        self.complex = complex
        self.root = root
        self.digits = digits
        self.x2_transformation = x2_transformation
        self.x2_angle = x2_angle

    def setup(self, stage=None):
        train_dataset = PairedTMNISTDataset(self.root, train=True, 
                                            digits=self.digits, 
                                            x2_transformation=self.x2_transformation, 
                                            x2_angle=self.x2_angle)

        train_size = int(0.8 * len(train_dataset))
        val_size = int(0.2 * len(train_dataset))
        #test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        self.test_dataset = PairedTMNISTDataset(self.root, train=False, 
                                                digits=self.digits,
                                                x2_transformation=self.x2_transformation,
                                                x2_angle=self.x2_angle)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True
        )


# Example usage
        
    
if __name__ == "__main__":
    transform = transforms.ToTensor()
    
    data_module = PairedTMNISTDataModule(batch_size=64, transform=transform, val_split=0.1)
    data_module.setup(stage="fit")
    print("train_dataloader_len = ", data_module.train_dataloader())#.__len__())
    print("test_dataloader_len = ", data_module.test_dataloader())#.__len__())

    # Print examples 
    index = 0
    for (x1, y1), (x2, y2) in data_module.train_dataloader():
        print(f"Batch {index+1}:")
        print(f"x1 labels: {y1.tolist()}, x2 labels: {y2.tolist()}")
        index += 1
        if index == 10:
            break 

