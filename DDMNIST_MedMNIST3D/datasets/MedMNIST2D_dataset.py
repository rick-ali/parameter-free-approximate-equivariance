from torch.utils.data import DataLoader
from torchvision import transforms
import medmnist
from medmnist import INFO
import pytorch_lightning as pl
import PIL



class MedMNISTDataModule(pl.LightningDataModule):
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
        self.train_dataset = self.DataClass(split='train', transform=self.transform,
                                            download=self.download, as_rgb=self.as_rgb, size=self.size)
        self.val_dataset = self.DataClass(split='val', transform=self.transform,
                                          download=self.download, as_rgb=self.as_rgb, size=self.size)
        self.test_dataset = self.DataClass(split='test', transform=self.transform,
                                           download=self.download, as_rgb=self.as_rgb, size=self.size)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)
        

if __name__ == '__main__':
    data_module = MedMNISTDataModule('pathmnist', 128, resize=False, as_rgb=False, size=28, download=False)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    print(len(train_loader), len(val_loader), len(test_loader))
    for x, y in train_loader:
        print(x.shape, y.shape)
        break