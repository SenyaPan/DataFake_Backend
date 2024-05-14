import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class block(nn.Module):
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
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
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
    def __init__(self, block, layers, num_classes, device=device):
        super(ParallelResNet, self).__init__()
        self.device = device

        self.resnet_filt = ResNet(block, layers, 3)
        self.resnet_fft = ResNet(block, layers, 1)

        self.fc_combined = nn.Linear(2048, num_classes)

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
        # Проходим через обычную ветку ResNet
        out_fft = self.fft_proc(x.clone())
        out_fft = self.resnet_fft.conv1(out_fft)
        out_fft = self.resnet_fft.bn1(out_fft)
        out_fft = self.resnet_fft.relu(out_fft)
        out_fft = self.resnet_fft.maxpool(out_fft)
        out_fft = self.resnet_fft.layer1(out_fft)
        out_fft = self.resnet_fft.layer2(out_fft)
        out_fft = self.resnet_fft.layer3(out_fft)
        out_fft = self.resnet_fft.layer4(out_fft)
        out_fft = self.resnet_fft.avgpool(out_fft)
        out_fft = out_fft.reshape(out_fft.shape[0], -1)
        out_fft = self.resnet_fft.fc(out_fft)

        # Проходим через ветку с обработкой шума
        out_noise_proc = self.noise_proc(x.clone())
        out_noise_proc = self.resnet_filt.conv1(out_noise_proc)
        out_noise_proc = self.resnet_filt.bn1(out_noise_proc)
        out_noise_proc = self.resnet_filt.relu(out_noise_proc)
        out_noise_proc = self.resnet_filt.maxpool(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer1(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer2(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer3(out_noise_proc)
        out_noise_proc = self.resnet_filt.layer4(out_noise_proc)
        out_noise_proc = self.resnet_filt.avgpool(out_noise_proc)
        out_noise_proc = out_noise_proc.reshape(out_noise_proc.shape[0], -1)
        out_noise_proc = self.resnet_filt.fc(out_noise_proc)

        combined_out = torch.cat((out_fft, out_noise_proc), dim=1)
        final_out = self.fc_combined(combined_out)
        return final_out


model = ParallelResNet(block=block,
                       layers=[2, 1, 1, 1],
                       num_classes=2, device=device)