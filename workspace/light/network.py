import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.ResNormalGuided import NormalGuided
from common.UNet import UNet
from help_funs import mu


class CNN(nn.Module):
    def __init__(self, c_num):
        super().__init__()
        # self.__name__ = 'light'

        # input confidence estimation network
        self.light3_3 = NormalGuided(3, 3, c_num)

    def forward(self, x):
        # general surface training
        x_light = x[:, 4:7, :, :]
        xout_light = self.light3_3(x_light)
        return xout_light
