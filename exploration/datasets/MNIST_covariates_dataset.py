import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as  TF

class PairedMNISTDataset(Dataset):
    def __init__(self, root="./dataset", train=True, transform=None, x2_angle=None):
        # Load the full MNIST dataset
        self.data = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.x2_angle = x2_angle
        assert 360 % x2_angle == 0

    def __len__(self):
        return len(self.data)
    
    def affine_transform(self, image, angle, scale, shear, x_displacement=0, y_displacement=0):
        # x,y displacement is useless if we use a CNN since it's translation invariant
        return TF.affine(image, angle, [x_displacement, y_displacement], scale, shear)

    def __getitem__(self, idx):
        x1, y1 = self.data[idx]
        
        angle = 0#random.uniform(-180, 180)
        scale = 0#random.uniform(0.8, 1.2)
        shear = 0

        covariate = random.randint(1, int(360/self.x2_angle)-1)
        x2_angle = (angle + 180 + self.x2_angle*covariate) % 360 - 180
        transformed_X1 = self.affine_transform(x1, angle=angle, scale=scale, shear=shear)
        transformed_X2 = self.affine_transform(x1, angle=x2_angle, scale=scale, shear=shear)

        return (transformed_X1, y1), (transformed_X2, y1), 0, covariate



class PairedMNISTCovariatesDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, transform=transforms.ToTensor(), val_split=0.1, x2_angle=None):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split  # Fraction of the training set to use for validation
        self.x2_angle = x2_angle

    def setup(self, stage=None):
        full_train_dataset = PairedMNISTDataset(train=True, transform=self.transform, x2_angle=self.x2_angle)

        val_len = int(len(full_train_dataset) * self.val_split)
        train_len = len(full_train_dataset) - val_len

        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_len, val_len])

        self.test_dataset = PairedMNISTDataset(train=False, transform=self.transform, x2_angle=self.x2_angle)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
            shuffle=True, persistent_workers=True
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
    
    data_module = PairedMNISTCovariatesDataModule(batch_size=64, transform=transform, val_split=0.1)
    data_module.setup(stage="fit")

    # Print examples 
    index = 0
    for (x1, y1), (x2, y2) in data_module.train_dataloader():
        print(f"Batch {index+1}:")
        print(f"x1 labels: {y1.tolist()}, x2 labels: {y2.tolist()}")
        index += 1
        if index == 10:
            break 

    
    index = 0
    for (x1, y1), (x2, y2) in data_module.test_dataloader():
        print(f"Batch {index+1}:")
        print(f"x1 labels: {y1.tolist()}, x2 labels: {y2.tolist()}")
        index += 1
        
    


