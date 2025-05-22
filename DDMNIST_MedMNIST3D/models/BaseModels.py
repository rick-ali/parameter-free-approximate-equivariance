'''
Adapted from kuangliu/pytorch-cifar .
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import operator
import argparse

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1, num_classes=2, get_latent=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.get_latent = get_latent

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        latent = out.view(out.size(0), -1)
        out = self.linear(latent)
        if self.get_latent:
            return out, latent
        return out


def ResNet18(in_channels, num_classes, get_latent=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes, get_latent=get_latent)


def ResNet50(in_channels, num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, num_classes=num_classes)

# From https://github.com/QUVA-Lab/partial-escnn/blob/main/networks/networks_cnn.py#L728
# A Probabilistic Approach to Learning the Degree of Equivariance in Steerable CNNs
class ResBlock(torch.nn.Module):
    def __init__(self, block, skip):
        super(ResBlock, self).__init__()
        self.block = block
        self.skip = skip

    def forward(self, x):
        return self.block(x) + self.skip(x)
class CNN3DResnet(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, mnist_type="single", c=6, get_latent=False):
        self.get_latent = get_latent
        super(CNN3DResnet, self).__init__()

        if mnist_type == "double":
            w = h = d = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = d = 29
            padding_3 = (1, 1, 1)
            padding_4 = (2, 2, 2)
        c = c

        self.upsample = torch.nn.Upsample(size=(h, w, d))

        block_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, c, 7, stride=1, padding=2),
            torch.nn.BatchNorm3d(c),
            torch.nn.ELU(),
        )

        block_2 = torch.nn.Sequential(
            torch.nn.Conv3d(c, 2 * c, 5, stride=1, padding=2),
            torch.nn.BatchNorm3d(2 * c),
            torch.nn.ELU(),
        )

        skip_1 = torch.nn.Sequential(
            torch.nn.Conv3d(1 * n_channels, 2 * c, 7, padding=2)
        )

        self.resblock_1 = ResBlock(torch.nn.Sequential(block_1, block_2), skip_1)

        self.pool_1 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        block_3 = torch.nn.Sequential(
            torch.nn.Conv3d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
            torch.nn.BatchNorm3d(4 * c),
            torch.nn.ELU(),
        )

        pool_2 = torch.nn.AvgPool3d(5, stride=2, padding=1)

        block_4 = torch.nn.Sequential(
            torch.nn.Conv3d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        skip_2 = torch.nn.Sequential(
            torch.nn.AvgPool3d(5, stride=2, padding=0), torch.nn.Conv3d(2 * c, 6 * c, 3)
        )

        self.resblock_2 = ResBlock(
            torch.nn.Sequential(block_3, pool_2, block_4), skip_2
        )

        block_5 = torch.nn.Sequential(
            torch.nn.Conv3d(6 * c, 6 * c, 3, stride=1, padding=1),
            torch.nn.BatchNorm3d(6 * c),
            torch.nn.ELU(),
        )

        pool_3 = torch.nn.AvgPool3d(3, stride=1, padding=0)

        block_6 = torch.nn.Conv3d(6 * c, 8 * c, 1)

        skip_3 = torch.nn.Sequential(torch.nn.Conv3d(6 * c, 8 * c, 3))

        self.resblock_3 = ResBlock(
            torch.nn.Sequential(block_5, pool_3, block_6), skip_3
        )

        self.fully_net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(8 * c),
            torch.nn.ELU(),
            torch.nn.Linear(8 * c, n_classes),
        )
        self.in_type = lambda x: x

    def forward(self, x, latent_ids=None):
        x = self.upsample(x)
        # x = self.block_1(x)
        # x = self.block_2(x)
        x = self.resblock_1(x)
        x = self.pool_1(x)
        # x = self.block_3(x)
        # x = self.pool_2(x)
        # x = self.block_4(x)
        x = self.resblock_2(x)
        # x = self.block_5(x)
        # x = self.pool_3(x)
        # x = self.block_6(x)
        x = self.resblock_3(x)
        latents = x.reshape(x.shape[0], -1)
        x = self.fully_net(latents)

        if self.get_latent:
            return x, [latents]
        return x

    @property
    def network_name(self):
        return "CNN"



# DDMNIST CNN
# From https://github.com/QUVA-Lab/partial-escnn/blob/main/networks/networks_cnn.py#L524
class DDMNISTCNN(torch.nn.Module):
    def __init__(self, n_classes=10, n_channels=1, c = 6, mnist_type="single", get_latent=False):
        super(DDMNISTCNN, self).__init__()
        self.get_latent = get_latent
        self.n_classes = n_classes

        if mnist_type == "double":
            w = h = 57
            padding_3 = (2, 2)
            padding_4 = (0, 0)
        elif mnist_type == "single":
            w = h = 29
            padding_3 = (1, 1)
            padding_4 = (2, 2)
        

        self.upsample = torch.nn.Upsample(size=(h, w))
        self.dims = {}
        
        self.expanded_factor = 11 # it was an 8

        self.layers = torch.nn.ModuleList([
            self.upsample, # Layer 0
            torch.nn.Sequential(
                torch.nn.Conv2d(1 * n_channels, c, 7, stride=1, padding=2),
                torch.nn.BatchNorm2d(c),
                torch.nn.ELU(),
            ), # Layer 1 
            torch.nn.Sequential(
                torch.nn.Conv2d(c, 2 * c, 5, stride=1, padding=2),
                torch.nn.BatchNorm2d(2 * c),
                torch.nn.ELU(),
            ), # Layer 2 
            torch.nn.AvgPool2d(5, stride=2, padding=1), # Layer 3
            torch.nn.Sequential(
                torch.nn.Conv2d(2 * c, 4 * c, 3, stride=2, padding=padding_3),
                torch.nn.BatchNorm2d(4 * c),
                torch.nn.ELU(),
            ), # Layer 4
            torch.nn.AvgPool2d(5, stride=2, padding=1), # Layer 5
            torch.nn.Sequential(
                torch.nn.Conv2d(4 * c, 6 * c, 3, stride=2, padding=padding_4),
                torch.nn.BatchNorm2d(6 * c),
                torch.nn.ELU(),
            ), # Layer 6
            torch.nn.Sequential(
                torch.nn.Conv2d(6 * c, 6 * c, 3, stride=1, padding=1),
                torch.nn.BatchNorm2d(6 * c),
                torch.nn.ELU(),
            ), # Layer 7
            torch.nn.AvgPool2d(5, stride=1, padding=1), # Layer 8
            torch.nn.Conv2d(6 * c, self.expanded_factor * c, 1), # Layer 9 
            torch.nn.Flatten(), # Layer 10
            torch.nn.BatchNorm1d(self.expanded_factor * c), # Layer 11
            # torch.nn.ELU(), # Layer 12
            # torch.nn.Linear(8 * c, self.expanded_dim), # Layer 13
            torch.nn.ELU(), # Layer 12
            torch.nn.Linear(self.expanded_factor * c, n_classes), # Layer 13
        ])

        #self.dims, self.shapes = self.get_shapes()

    def forward(self, x, latent_ids=[-1]):
        latents = []
        for id, layer in enumerate(self.layers):
            x = layer(x)
            if id in latent_ids:
                latents.append(x.reshape(x.shape[0], -1))


        if self.get_latent:
            return x, latents

        return x

    @property
    def network_name(self):
        return "DDMNISTCNN"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DDMNIST CNN")
    parser.add_argument("--c", type=int, default=6, help="Number of channels")
    args = parser.parse_args()

    c = args.c

    model_6 = DDMNISTCNN(n_classes=100, n_channels=1, c=6, mnist_type="double", get_latent=True)
    model = DDMNISTCNN(n_classes=100, n_channels=1, c=c, mnist_type="double", get_latent=True)
    x = torch.rand(1, 1, 56, 56)
    out, latents = model(x)
    #print(f"Dimensions of the model with c={c}:\n{model.dims}\n")
    #print(f"Shapes of the model with c={c}:\n{model.shapes}\n")

    parameter_count_c = model.count_parameters()
    print(f"Total parameters for c={c}: {parameter_count_c}")
    
    parameter_count_6 = model_6.count_parameters()
    print(f"Total parameters for c=6: {parameter_count_6}")

    # Ratio of difference 
    difference = (parameter_count_c - parameter_count_6) / parameter_count_6
    print(f"Difference in parameters: {difference:.2%}")

    