import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import random

class PairedMNISTDataset(Dataset):
    def __init__(self, root="./dataset", train=True, transform=transforms.ToTensor(), complex=False, n_digits=2, offset=1):
        # Load the full MNIST dataset
        full_dataset = datasets.MNIST(root=root, train=train, download=True, transform=transform)
        self.n_digits = n_digits
        self.offset = offset
        self.complex = complex
        
        self.data = {label : [(x, y) for x, y in full_dataset if y==label] for label in range(n_digits)}
        self.transform = transform

    def __len__(self):
        # The length is the smaller of the datasets to ensure balanced pairing
        return min(len(self.data[label]) for label in self.data)

    def __getitem__(self, idx):
        # Randomly pick a base digit and its paired digit with the offset
        base_digit = random.randint(0, self.n_digits - 1)
        paired_digit = (base_digit + self.offset) % self.n_digits  # Wrap-around pairing if out of range

        # Fetch images and labels for the pair
        x1, y1 = self.data[base_digit][idx % len(self.data[base_digit])]
        x2, y2 = self.data[paired_digit][idx % len(self.data[paired_digit])]

        if self.complex:
            x1 = x1.type(torch.complex64)
            x2 = x2.type(torch.complex64)
            y1 = y1.type(torch.complex64)
            y2 = y2.type(torch.complex64)

        return (x1, y1), (x2, y2)



class PairedMNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, num_workers=4, transform=transforms.ToTensor(), val_split=0.1, complex=False, n_digits=2, offset=1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.val_split = val_split  # Fraction of the training set to use for validation
        self.complex = complex
        self.n_digits = n_digits
        self.offset = offset

    def setup(self, stage=None):
        full_train_dataset = PairedMNISTDataset(train=True, transform=self.transform, n_digits=self.n_digits, offset=self.offset)

        val_len = int(len(full_train_dataset) * self.val_split)
        train_len = len(full_train_dataset) - val_len

        self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_len, val_len])

        self.test_dataset = PairedMNISTDataset(train=False, transform=self.transform, n_digits=self.n_digits, offset=self.offset)

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
    
    data_module = PairedMNISTDataModule(batch_size=64, transform=transform, val_split=0.1)
    data_module.setup(stage="fit")
    print("train_dataloader_len = ", data_module.train_dataloader().__len__())
    print("test_dataloader_len = ", data_module.test_dataloader().__len__())

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
        
    


