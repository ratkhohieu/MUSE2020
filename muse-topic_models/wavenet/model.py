import torch
import torchvision
import torch.nn as nn

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision.models import *


class Wave_Block(nn.Module):

    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()

        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1))
        dilation_rates = [2 ** i for i in range(dilation_rates)]
        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate * (kernel_size - 1)) / 2), dilation=dilation_rate))
            self.convs.append(nn.Conv1d(out_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.convs[0](x)
        res = x
        for i in range(self.num_rates):
            x = torch.tanh(self.filter_convs[i](x)) * torch.sigmoid(self.gate_convs[i](x))
            x = self.convs[i + 1](x)
            res = res + x
        return res


# detail
class wavenet(nn.Module):
    def __init__(self, inch=16, kernel_size=3, num_classes=3):
        super().__init__()

        self.wave_block1 = Wave_Block(inch, 16, 4, kernel_size)
        self.wave_block2 = Wave_Block(16, 32, 4, kernel_size)
        self.wave_block3 = Wave_Block(32, 64, 2, kernel_size)
        self.wave_block4 = Wave_Block(64, 128, 1, kernel_size)
        self.pool = nn.AdaptiveAvgPool1d(4)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.wave_block1(x)
        x = self.wave_block2(x)
        x = self.wave_block3(x)
        x = self.wave_block4(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
