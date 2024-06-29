import torch
import torch.nn as nn
import numpy as np


class Resample(nn.Module):
    def __init__(self, size=256):
        super(Resample, self).__init__()
        self.size = size

    def forward(self, x):
        inds_1 = torch.LongTensor(np.linspace(0, x.size(2), self.size, endpoint=False)).to(
            device=x.device)
        inds_2 = torch.LongTensor(np.linspace(0, x.size(3), self.size, endpoint=False)).to(
            device=x.device)
        resample_x = x.index_select(2, inds_1)
        resample_x = resample_x.index_select(3, inds_2)
        return resample_x

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )
    def forward(self, x):
        return x + self.block(x)

class Res_module(nn.Module):
    def __init__(self, in_features):
        super(Res_module, self).__init__()

        self.block = nn.Sequential(
            ResidualBlock(in_features),
            ResidualBlock(in_features),
            ResidualBlock(in_features),
        )
    def forward(self, x):
        return x + self.block(x)

class ConditionNet(nn.Module):
    def __init__(self):
        super(ConditionNet, self).__init__()
        self.net = nn.Sequential(
            Resample(),
            nn.Conv2d(3, 16, (7, 7), (1, 1), (3, 3)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(16, 32, (4, 4), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            Res_module(32),
            nn.Conv2d(32, 128, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, (3, 3), (2, 2), (1, 1)),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 320, (1, 1), (1, 1), (0, 0)),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(320, 256, (1, 1), (1, 1), (0, 0)),
        )

    def forward(self, rgb):
        return self.net(rgb)