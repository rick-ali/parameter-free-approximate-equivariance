import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import random
import torchvision.transforms.functional as  TF
import matplotlib.pyplot as plt
import numpy as np
import medmnist
from medmnist import INFO, Evaluator

class PairedMNISTDataset(Dataset):
    def __init__(self, data, split, x2_angle=None):

        self.rotations = [i for i in range(int(360//x2_angle))]
        self.split = split
    
        self.data = data

        self.x2_angle = x2_angle
        print(f"Using x2_angle = {x2_angle}")
        assert 360 % x2_angle == 0

    def __len__(self):
        # if self.split == "train":
        #     return len(self.data)
        # else:
        #     return len(self.data) * len(self.rotations)
        return len(self.data)
    

    def affine_transform(self, image, angle, scale, shear, x_displacement=0, y_displacement=0):
        # x,y displacement is useless if we use a CNN since it's translation invariant
        return TF.affine(image, angle, [x_displacement, y_displacement], scale, shear, interpolation=TF.InterpolationMode.BILINEAR)


    def __getitem__(self, idx):
        
        if self.split == "train":
            x1, y1 = self.data[idx]
            numeral = y1
            augment_angle_covariate = random.randint(0, int(360/self.x2_angle)-1)
            augment_angle = augment_angle_covariate * self.x2_angle
            scale = 1.0
            augmented_X1 = self.affine_transform(x1, angle=augment_angle, scale=scale, shear=0)
            y1 = torch.tensor([numeral, augment_angle_covariate])
            
            transformation_type = 0
            # Here introduces soft constraints if range = [1, 360/x2_angle], as we get W^N
            covariate = random.randint(1, int(360/self.x2_angle)-1)
            x2_angle = (augment_angle_covariate+covariate)%int(360/self.x2_angle) * self.x2_angle
            target_angle = int(x2_angle/self.x2_angle)
            y2 = torch.tensor([numeral, target_angle])
            transformed_X2 = self.affine_transform(x1, angle=x2_angle, scale=scale, shear=0)
        else:
            img_idx = idx // len(self.rotations)
            x1, y1 = self.data[img_idx]

            rotation_idx = idx % len(self.rotations) 

            numeral = y1
            augment_angle_covariate = self.rotations[rotation_idx]
            augment_angle = augment_angle_covariate * self.x2_angle
            scale = 1.0
            augmented_X1 = self.affine_transform(x1, angle=augment_angle, scale=scale, shear=0)
            y1 = torch.tensor([numeral, augment_angle_covariate])
            
            transformation_type = 0
            covariate = random.randint(1, int(360/self.x2_angle)-1)
            x2_angle = (augment_angle_covariate+covariate)%int(360/self.x2_angle) * self.x2_angle
            target_angle = int(x2_angle/self.x2_angle)
            y2 = torch.tensor([numeral, target_angle])
            transformed_X2 = self.affine_transform(x1, angle=x2_angle, scale=scale, shear=0)

        return (augmented_X1, y1), (transformed_X2, y2), transformation_type, covariate



class PairedClassificationDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, transform=transforms.ToTensor(), val_split=0.1, x2_angle=None, dataset='CIFAR', data_flag='pathmnist'):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split  # Fraction of the training set to use for validation
        self.x2_angle = x2_angle
        self.dataset = dataset
        self.data_flag = data_flag

    def setup(self, stage=None):
        # Load full MNIST dataset (without splitting)
        if self.dataset == 'MNIST':
            full_train_dataset = datasets.MNIST(root="./dataset", train=True, transform=transforms.ToTensor(), download=True)
            full_test_dataset = datasets.MNIST(root="./dataset", train=False, transform=transforms.ToTensor(), download=True)
        elif self.dataset == 'CIFAR':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            full_train_dataset = datasets.CIFAR10(root="./dataset/CIFAR10", train=True, download=True, transform=transform)
            full_test_dataset = datasets.CIFAR10(root="./dataset/CIFAR10", train=True, download=False, transform=transform)
        elif self.dataset == 'MedMNIST':
            data_flag = self.data_flag
            info = INFO[data_flag]

            data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
            DataClass = getattr(medmnist, info['python_class'])
            full_train_dataset = DataClass(root=f'./dataset/medmnist/{data_flag}', split='train', transform=data_transform, download=True)
            full_test_dataset = DataClass(root=f'./dataset/medmnist/{data_flag}', split='test', transform=data_transform, download=True)
            

        val_len = int(len(full_train_dataset) * self.val_split)
        train_len = len(full_train_dataset) - val_len

        # Randomly split dataset
        train_subset, val_subset = torch.utils.data.random_split(
            full_train_dataset, [train_len, val_len]
        )

        # Create dataset instances
        self.train_dataset = PairedMNISTDataset(train_subset, split="train", x2_angle=self.x2_angle)
        self.val_dataset = PairedMNISTDataset(val_subset, split="val", x2_angle=self.x2_angle)
        self.test_dataset = PairedMNISTDataset(full_test_dataset, split="test", x2_angle=self.x2_angle)

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
    
    data_module = PairedClassificationDataModule(batch_size=64, transform=transform, val_split=0.1, x2_angle=120)
    data_module.setup(stage="fit")

    print(len(data_module.train_dataset))
    print(int(len(data_module.train_dataset)*0.1))
    print("\n")
    print(len(data_module.val_dataset))
    print(len(data_module.test_dataset))

        
    


