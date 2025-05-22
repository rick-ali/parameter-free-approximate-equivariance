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
from utils.representations import D8RegularRepresentation

class PairedD8MedMNIST2D(Dataset):
    def __init__(self, data, transform, split, test_all_rotations=False):
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
        self.transform = transform
        self.representation = D8RegularRepresentation(device='cpu')


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
    

    def augment_image(self, img: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Augments the input image using a random symmetry from D16, i.e. the dihedral group
        of an octagon. The convention is:
        - Indices 0 to 7: pure rotations by 0, 45, ..., 315 degrees.
        - Indices 8 to 15: reflection (horizontal flip) followed by a rotation by 0, 45, ..., 315 degrees.
        
        Returns:
        res: the augmented image
        choice: an integer (0-15) representing the chosen group element.
        """
        choice = random.randint(0, 15)
        
        if choice < 8:
            # Pure rotation: rotate by 45 * choice degrees.
            angle = 45 * choice
            res = TF.rotate(img, angle)
        else:
            # Reflection followed by rotation.
            # Here we define s as a horizontal flip.
            k = choice - 8
            # This corresponds to the group element r^k s, i.e. apply s then rotate by 45*k.
            res = TF.rotate(TF.hflip(img), 45 * k)
        
        return res, choice
    

    def __getitem__(self, idx):
        transformation_type = 0

        if self.split == 'train':
            x1, y1 = self.data.__getitem__(idx)
            augmented_X1, g_x1 = self.augment_image(x1)
            transformed_X2, g_x2 = self.augment_image(x1)
            covariate = self.representation.group_mult(g_x2, self.representation.d16_inverse(g_x1))
        
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
                augmented_X1, g_x1 = x1, 0
                transformed_X2, g_x2 = self.augment_image(x1)
                covariate = self.representation.group_mult(g_x2, self.representation.d16_inverse(g_x1))

        return (augmented_X1, y1), (transformed_X2, y1), transformation_type, covariate



class D8MedMNISTDataModule(pl.LightningDataModule):
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
        
        if resize:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224), interpolation=PIL.Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[.5], std=[.5])
            ])

    def setup(self, stage=None):
        train_data = self.DataClass(split='train', transform=self.transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        self.train_dataset = PairedD8MedMNIST2D(train_data, transform=self.transform, split='train')

        val_data = self.DataClass(split='val', transform=self.transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        self.val_dataset = PairedD8MedMNIST2D(val_data, transform=self.transform, split='val')

        test_data = self.DataClass(split='test', transform=self.transform, download=self.download, as_rgb=self.as_rgb, size=self.size)
        self.test_dataset = PairedD8MedMNIST2D(test_data, transform=self.transform, split='test')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=15)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=15)
    
if __name__ == '__main__':
    data_module = D8MedMNISTDataModule('pathmnist', 128, resize=False, as_rgb=True, size=28, download=False)
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
    