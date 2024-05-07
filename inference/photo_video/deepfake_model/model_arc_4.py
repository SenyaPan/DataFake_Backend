import torch
import torch.nn as nn
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
    def __init__(self, block, layers, image_channels, num_classes, device=device):
        super(ResNet, self).__init__()
        self.device = device
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.conv_fft = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
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
        self.fc = nn.Linear(4096, num_classes)

    def noise_proc(self, x):
        channels = x.size(1)
        kernel = torch.tensor([[[[-1, 2, -2, 2, -1],
                                 [2, -6, 8, -6, 2],
                                 [-2, 8, -12, 8, -2],
                                 [2, -6, 8, -6, 2],
                                 [-1, 2, -2, 2, -1]]]], dtype=torch.float)
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(self.device)
        processed_noise = F.conv2d(x, kernel, padding=1, groups=channels)
        return processed_noise

    def fft_proc(self, x):
        fft_out = torch.fft.rfft2(x, norm='ortho')
        fft_out = torch.sqrt(fft_out.real ** 2 + fft_out.imag ** 2)
        fft_out = F.adaptive_avg_pool3d(fft_out, (1, 224, 112))
        fft_out = fft_out.squeeze(2)
        fft_out = F.normalize(fft_out, dim=3)
        return fft_out

    def forward(self, x):
        conv_out = self.noise_proc(x)
        conv_out = self.conv1(conv_out)
        conv_out = self.bn1(conv_out)
        conv_out = self.relu(conv_out)
        conv_out = self.maxpool(conv_out)
        conv_out = self.layer1(conv_out)
        conv_out = self.layer2(conv_out)
        conv_out = self.layer3(conv_out)
        conv_out = self.layer4(conv_out)

        conv_out = self.avgpool(conv_out)
        conv_out = conv_out.reshape(conv_out.shape[0], -1)

        fft_out = self.fft_proc(x)
        fft_out = self.conv_fft(fft_out)
        fft_out = self.bn1(fft_out)
        fft_out = self.relu(fft_out)
        fft_out = self.maxpool(fft_out)
        fft_out = self.layer1(fft_out)
        fft_out = self.layer2(fft_out)
        fft_out = self.layer3(fft_out)
        fft_out = self.layer4(fft_out)

        fft_out = self.avgpool(fft_out)
        fft_out = fft_out.reshape(fft_out.shape[0], -1)

        combined_out = torch.cat((conv_out, fft_out), dim=1)
        combined_out = combined_out.view(-1, 4096)
        out = self.fc(combined_out)
        return out

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

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
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        self.in_channels = intermediate_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


def ResNet18(arr, img_channel=3, num_classes=2):
    return ResNet(Block, arr, img_channel, num_classes)
