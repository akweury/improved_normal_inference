import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.UNet import UNet
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'noise_net'
        self.noise_net = UNet(1, 1)

    def forward(self, x0):
        xout = self.noise_net(x0)
        return xout
