import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConvNet(nn.Module):
    def __init__(self, device):
        super(CustomConvNet, self).__init__()
        self.device = device

        self.conv_to_vector1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.conv_to_vector2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

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
        conv_out = self.conv_to_vector1(conv_out)

        fft_out = self.fft_proc(x)
        fft_out = self.conv_to_vector2(fft_out)

        conv_out = torch.flatten(conv_out, 1)
        fft_out = torch.flatten(fft_out, 1)
        combined_out = torch.cat((conv_out, fft_out), dim=1)

        out = self.classifier(combined_out)
        return out
