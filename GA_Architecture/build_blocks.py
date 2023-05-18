import numpy as np
import random

import torch
import torch.nn as nn
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
from torchsummary import summary

def conv_block(in_channels, out_channels, kernel_size, *args, **kwargs):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    return nn.Sequential(OrderedDict({
        'conv': nn.Conv2d(in_channels, out_channels, kernel_size, padding='same'),
        'bn': nn.BatchNorm2d(out_channels),
        'act': nn.LeakyReLU()}))

def pool_block(type, kernel_size, *args, **kwargs):
    pooling = nn.ModuleDict([['avg', nn.AvgPool2d(kernel_size=kernel_size, padding='same', *args, **kwargs)],
                             ['max', nn.MaxPool2d(kernel_size=kernel_size, padding='same', *args, **kwargs)]])
    return pooling[type]

def concat_block(branch_1, branch_2):
    # N, C, H, W
    b1_shape = branch_1.shape[-1]
    b2_shape = branch_2.shape[-1]

    if b1_shape > b2_shape:
        size_down = int(b1_shape/b2_shape)
        maxpool = nn.MaxPool2d(kernel_size=size_down)
        branch_1 = maxpool(branch_1)
    else:
        size_down = int(b2_shape/b1_shape)
        maxpool = nn.MaxPool2d(kernel_size=size_down)
        branch_2 = maxpool(branch_2)
    return torch.cat((branch_1, branch_2), dim=1)

class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
    def forward(self, x):
       return self.relu(self.batchnorm(self.conv(x)))


# BUILDING BLOCKS FOR THE ARCHITECTURE
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super(conv_block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
       return self.leakyrelu(self.batchnorm(self.conv(x)))

class conv_nxn_subblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), *args, **kwargs):
        super().__init__()

        self.red_channels = int(in_channels/4)

        if kernel_size == (1,1):
            self.conv_nxn = conv_block(in_channels, out_channels, kernel_size=kernel_size)
        else:
            self.conv_nxn = nn.Sequential(conv_block(in_channels, self.red_channels, kernel_size=(1, 1)),
                                         conv_block(self.red_channels, out_channels, kernel_size=kernel_size, padding='same'))
    def forward(self, x):
        return self.conv_nxn(x)

class maxpool_subblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(1, 1), *args, **kwargs):
        super().__init__()
        self.padding = int(kernel_size[0]//2)
        self.maxpool = nn.Sequential(nn.MaxPool2d(kernel_size=kernel_size, stride=(1,1), padding=(self.padding, self.padding)),
                                     conv_block(in_channels, out_channels, kernel_size=(1, 1)))
    def forward(self, x):
        return self.maxpool(x)

class avgpool_subblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), *args, **kwargs):
        super().__init__()
        self.padding = int(kernel_size[0]//2)
        self.avgpool = nn.Sequential(nn.AvgPool2d(kernel_size=kernel_size, stride=(1,1), padding=(self.padding, self.padding)),
                                     conv_block(in_channels, out_channels, kernel_size=(1, 1)))

    def forward(self, x):
        return self.avgpool(x)

class identity_subblock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, blocks, output_channels, *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([*[subblock for subblock in blocks]])
        self.conv_1x1 = conv_block(in_channels, output_channels, kernel_size=(1,1))

    def forward(self, x):
        x_list = []
        for i,block in enumerate(self.blocks):
            # print(i, block(x).shape)
            x_list.append(block(x))
        x = torch.cat(x_list, dim=1)
        x = self.conv_1x1(x)
        return x
