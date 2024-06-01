from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(
            self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super().__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels):
        super(ResNet, self).__init__()
        self.device = device
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=2
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, 1024)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        layers.append(
            Block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


class ParallelResNet(nn.Module):
    def __init__(self, block, layers, layers_fft, num_classes, device='cpu'):
        super(ParallelResNet, self).__init__()
        self.device = device

        # Initialize two ResNet models
        self.resnet_filt = ResNet(block, layers, 3)
        self.resnet_fft = ResNet(block, layers_fft, 1)

        # Modify the first convolutional layer to accept different input channels
        self.resnet_filt.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_fft.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Define the combined fully connected layer
        self.fc_combined = nn.Linear(2048 * 2, num_classes)  # Concatenation of two 2048 features

        # Move models to the specified device
        self.to(self.device)

    def forward(self, x, y):
        # Process through the FFT branch (single channel input)
        out_fft = self.resnet_fft.conv1(x)
        out_fft = self.resnet_fft.bn1(out_fft)
        out_fft = self.resnet_fft.relu(out_fft)
        out_fft = self.resnet_fft.maxpool(out_fft)
        out_fft = self.resnet_fft.layer1(out_fft)
        out_fft = self.resnet_fft.layer2(out_fft)
        out_fft = self.resnet_fft.layer3(out_fft)
        out_fft = self.resnet_fft.layer4(out_fft)
        out_fft = self.resnet_fft.avgpool(out_fft)
        out_fft = out_fft.view(out_fft.size(0), -1)

        # Process through the noise-processed branch (3 channel input)
        out_noise_proc = self.resnet_filt.conv1(y)
        out_noise_proc = self.resnet_filt.bn1(out_noise_proc)
        out_noise_proc = self.resnet_filt.relu(out_noise_proc)
        out_noise_proc = self.resnet_filt.maxpool(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer1(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer2(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer3(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer4(out_noise_proc)
        out_noise_proc = self.resnet_filt.avgpool(out_noise_proc)
        out_noise_proc = out_noise_proc.view(out_noise_proc.size(0), -1)

        # Combine the outputs
        combined_out = torch.cat((out_fft, out_noise_proc), dim=1)

        # Final output through combined fully connected layer
        final_out = self.fc_combined(combined_out)
        return final_out