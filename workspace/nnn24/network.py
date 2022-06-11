import sys
from os.path import dirname

import cv2

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.NormalNN24 import NormalNN24
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'nnn24'

        # input confidence estimation network
        self.conv24_3 = NormalNN24(24, 3)

    def forward(self, x0):
        # x0: vertex array
        # c0: confidence of each element in x0
        device = x0.get_device()
        # c0 = mu.binary(x0).to(device)
        xout = self.conv24_3(x0)

        out = torch.cat((xout, xout, xout), 1)
        return out
