import pytorch_lightning as pl
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, IterableDataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as  TF
import random
import pandas as pd


class PairedMultiWTMNISTDataset(Dataset):
    def __init__(self, root, train, complex=False, x2_angle=None):
        self.train = train
        self.complex = complex

        self.x2_angle = x2_angle
        
        df = pd.read_csv(root)
        df1 = df[(df['labels'] == 5) & (df['names'] == 'IBMPlexSans-MediumItalic')]
        df2 = df[(df['labels'] == 5) & (df['names'] == 'Bahianita-Regular')]
        
        assert len(df1) == len(df2)

        X1 = df1.iloc[:,2:]
        X_data1 = torch.tensor(np.array(X1).reshape(len(df1), 1, 28, 28)) / 255.0
        y1 = df1['labels']
        y_data1 = torch.tensor(np.array(y1))

        X2 = df2.iloc[:,2:]
        X_data2 = torch.tensor(np.array(X2).reshape(len(df2), 1, 28, 28)) / 255.0
        y2 = df2['labels']
        y_data2 = torch.tensor(np.array(y2))

        self.data1 = (X_data1, y_data1)
        self.data2 = (X_data2, y_data2)

    def __len__(self):
        # Potentially infinite as there are only 2*n_digits base images and the dataset consists in augmentations. 
        return 10000  # Like MNIST
    
    
    def affine_transform(self, image, angle, scale, shear, x_displacement=0, y_displacement=0):
        # x,y displacement is useless if we use a CNN since it's translation invariant
        return TF.affine(image, angle, [x_displacement, y_displacement], scale, shear)


    def __getitem__(self, idx):
        x1 = self.data1[0]
        y1 = self.data1[1][0]
        y2 = self.data2[1][0]

        transformations = ['angle', 'scale', 'font']
        transformation = random.randint(0, len(transformations)-1)

        if transformation == 0:  # angle
            # angle = random.uniform(-10, 10)
            # x2 = self.affine_transform(x1, angle=angle, scale=1, shear=1)
            # covariate = torch.tensor(angle, dtype=torch.float32)
            #! note that angle is not the actual rotation angle
            #! This way W_angle^1 = rotate 5°
            angle = random.randint(1, 10)
            x2 = self.affine_transform(x1, angle= 0+angle*10, scale=1, shear=1)
            covariate = torch.tensor(angle, dtype=torch.int32)
        elif transformation == 1:  # scale
            # scale = random.uniform(0.7, 1.1)
            # x2 = self.affine_transform(x1, angle=0, scale=scale, shear=1)
            # covariate = torch.tensor(scale, dtype=torch.float32)
            #! note that scale is not the actual scaling
            #! This way W_scale^1 = scale by 0.9
            scale = random.randint(1, 4)
            x2 = self.affine_transform(x1, angle=0, scale=1-scale*0.1, shear=1)
            covariate = torch.tensor(scale, dtype=torch.int32)
        elif transformation == 2:  # font
            x2 = self.data2[0]
            covariate = torch.tensor(1., dtype=torch.int32)

        return (x1[0], y1), (x2[0], y2), transformation, covariate



class PairedMultiWTMNISTDataModule(pl.LightningDataModule):
    def __init__(self, root='./dataset/TMNIST_Data.csv', batch_size=32, num_workers=4, transform=transforms.ToTensor(), val_split=0.1, complex=False):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split  # Fraction of the training set to use for validation
        self.complex = complex
        self.root = root

    def setup(self, stage=None):
        train_dataset = PairedMultiWTMNISTDataset(self.root, train=True)

        train_size = int(0.8 * len(train_dataset))
        val_size = int(0.2 * len(train_dataset))
        #test_size = len(full_dataset) - train_size - val_size

        self.train_dataset, self.val_dataset = random_split(
            train_dataset, [train_size, val_size]
        )

        self.test_dataset = PairedMultiWTMNISTDataset(self.root, train=False)

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
    
    data_module = PairedMultiWTMNISTDataModule(batch_size=64, transform=transform, val_split=0.1)
    data_module.setup(stage="fit")
    print("train_dataloader_len = ", data_module.train_dataloader())#.__len__())
    print("test_dataloader_len = ", data_module.test_dataloader())#.__len__())

    # Print examples 
    index = 0
    for (x1, y1), (x2, y2), transformation in data_module.train_dataloader():
        print(f"Batch {index+1}:")
        print(x1, y1, x2, y2, transformation)
        exit()

