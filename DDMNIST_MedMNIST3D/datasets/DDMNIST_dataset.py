from unicodedata import digit
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

class Rotate90Transform:
    def __init__(self, angles=[0, 90, 180, 270]):
        self.angles = angles

    def __call__(self, x):
        angle_small = np.random.uniform(-180, 180)
        # im = Image.fromarray((x.numpy().squeeze() * 255).astype(np.uint8))
        # im.save(f"{angle_small}-before.png")
        x = TF.rotate(x, int(angle_small), InterpolationMode.BILINEAR)
        angle = np.random.choice(self.angles) - angle_small
        img = TF.rotate(x, int(angle), InterpolationMode.BILINEAR)
        # im = Image.fromarray((img.numpy().squeeze() * 255).astype(np.uint8))
        # im.save(f"{angle_small}-after.png")
        # print("hokee")
        return img


class DDMNIST(Dataset):
    def __init__(
        self,
        train=True,
        digit_transform=transforms.ConvertImageDtype(torch.float),
        number_transform=None,
        max_val=9,
        square=True,
        images_per_class=100,
        normalize=True,
    ):
        self.num_transform = number_transform
        self.digit_transform = digit_transform
        self.normalize = normalize
        self.train = train
        self.max_val = max_val
        self.square = square
        self.imgs_per_class = images_per_class
        self.mnist = datasets.MNIST(
            root=f"data/MNIST/{'train' if train else 'test'}",
            download=True,
            train=train,
        )

        self.img1, self.img2, self.labels = self._create_data()
        if self.normalize:
            mean, std = self._std()
            self._normalize = transforms.Normalize((mean,), (std,))

    def _create_data(self):
        im_1_indices, im_2_indices, targets = [], [], []
        for num in range(100):
            tens = num // 10
            digits = num % 10
            tens_inds = list((self.mnist.targets == tens).nonzero(as_tuple=True)[0])
            digit_inds = list((self.mnist.targets == digits).nonzero(as_tuple=True)[0])
            im_1_indices += list(np.random.choice(tens_inds, size=self.imgs_per_class))
            im_2_indices += list(np.random.choice(digit_inds, size=self.imgs_per_class))
            targets += [num] * self.imgs_per_class

        return (
            self.mnist.data[im_1_indices],
            self.mnist.data[im_2_indices],
            torch.LongTensor(targets),
        )

    def __getitem__(self, index):
        img1, img2 = (
            self.img1[index],
            self.img2[index],
        )
        img1 = self.digit_transform(img1[None, :, :])
        img2 = self.digit_transform(img2[None, :, :])
        #combined = self._combine_images(img1[0], img2[0])
        #return combined, self.labels[index]
        return img1, img2, self.labels[index]

    def _std(self):
        if self.square:
            return (
                0.5 * 0.1307,
                np.sqrt(0.3081),
            )
        else:
            return 0.1307, 0.3081

    def _combine_images(self, img1, img2):
        h, w = img1.shape
        if self.square:
            h_start = int(0.5 * h)
            h_end = int(1.5 * h)
            combined = torch.zeros((1, h * 2, w * 2), dtype=img1.dtype)
            combined[:, h_start:h_end, :w] = img1
            combined[:, h_start:h_end, w:] = img2
        else:
            combined = torch.zeros((1, h, 2 * w - 1), dtype=img1.dtype)
            combined[:, :, :w] = img1
            combined[:, :, w - 1 :] = img2

        if self.num_transform is not None:
            combined = self.num_transform(combined)
        if self.normalize:
            combined = self._normalize(combined)
        return combined

    def __len__(self):
        return len(self.labels)