import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as  TF
import matplotlib.pyplot as plt
import numpy as np

class PairedMNISTDataset(Dataset):
    def __init__(self, root="./dataset", train=True, transform=None, x2_angle=None):
        # Load the full MNIST dataset
        self.data = datasets.MNIST(root=root, train=train, download=True, transform=transforms.ToTensor())
        # #! CIFAR
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.5,), (0.5,))
        # ])
        # self.data = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
        self.x2_angle = x2_angle
        print(f"Using x2_angle = {x2_angle}")
        assert 360 % x2_angle == 0

    def __len__(self):
        return len(self.data)
    
    def affine_transform(self, image, angle, scale, shear, x_displacement=0, y_displacement=0):
        # x,y displacement is useless if we use a CNN since it's translation invariant
        return TF.affine(image, angle, [x_displacement, y_displacement], scale, shear, interpolation=TF.InterpolationMode.BILINEAR)


    def __getitem__(self, idx):
        #! Maybe I should give the network the full orbit?
        x1, y1 = self.data[idx]

        # Augment x1
        augment_reflection = random.choice([True, False])
        if augment_reflection:
            x1 = TF.hflip(x1)

        augment_angle = 0#random.uniform(-30, 30)
        scale = 1.0
        augmented_X1 = self.affine_transform(x1, angle=augment_angle, scale=scale, shear=0)

        # Transform x1 to get x2
        transformation_type = random.choice(['rotation', 'reflection'])
        if transformation_type == 'reflection':
            transformation_type = 1
            transformed_X2 = TF.hflip(augmented_X1)
            covariate = 1
        
        elif transformation_type == 'rotation':
            transformation_type = 0
            # Here introduces soft constraints if range = [1, 360/x2_angle], as we get W^N
            covariate = random.randint(1, int(360/self.x2_angle)-1)
            x2_angle = (augment_angle + 180 + self.x2_angle*covariate) % 360 - 180
            transformed_X2 = self.affine_transform(x1, angle=x2_angle, scale=scale, shear=0)
        
        return (augmented_X1, y1), (transformed_X2, y1), transformation_type, covariate



class PairedMultiWMNISTDataModule(pl.LightningDataModule):
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


def plot_pair_images(x1,x2, title="Title"):
    combined =np.hstack((x1, x2))
    # Plot the combined image
    plt.figure(figsize=(10, 5))  # Adjust figure size as needed
    plt.imshow(combined, cmap='gray')
    plt.axis('off')  # Turn off axis for better visualization
    plt.title(title)
    plt.show()

# Example usage
if __name__ == "__main__":
    transform = transforms.ToTensor()
    
    data_module = PairedMultiWMNISTDataModule(batch_size=64, transform=transform, val_split=0.1, x2_angle=60)
    data_module.setup(stage="fit")

    # Print examples 
    index = 0
    for (x1, y1), (x2, y2), type, covariate in data_module.train_dataloader():
        print(f"Batch {index+1}:")
        print(f"x1 labels: {y1.tolist()}, x2 labels: {y2.tolist()}")
        index += 1
        if index == 10:
            break 

    plot_pair_images(x1[0][0], x2[0][0], title=f"covariate={covariate[0]}")

        
    


