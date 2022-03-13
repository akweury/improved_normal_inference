import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import math

"""
Normal Neuron Network
input: vertex (3,512,512)
output: normal  (3,512,512)

"""


class SVDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'NormalNN'
        kernel_5 = (5, 5)
        kernel_3 = (3, 3)
        padding_2 = (2, 2)
        padding_1 = (1, 1)

        self.conv1 = nn.Conv2d(3, 128, kernel_3, (1, 1), padding_1)
        self.conv2 = nn.Conv2d(128, 256, kernel_3, (1, 1), padding_1)
        self.conv3 = nn.Conv2d(256, 256, kernel_3, (1, 1), padding_1)
        self.conv4 = nn.Conv2d(256, 512, kernel_3, (1, 1), padding_1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 3)

    def forward(self, x0):
        x = self.conv1(x0)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv3(x)
        x = F.max_pool2d(x, 2, 2)

        x = self.conv4(x)
        x = F.max_pool2d(x, 2, 2)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x
