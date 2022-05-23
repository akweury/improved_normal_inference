import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.DeGaResNet import DeGaResNet
from common.UNet import UNet
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'DeGaRes'

        # input confidence estimation network
        self.degares3_3 = DeGaResNet(3, 3)
        self.detailNet = UNet(3, 3)

    def forward(self, x):
        # general surface training
        xout = self.degares3_3(x)
        return xout
