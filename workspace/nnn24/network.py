import sys
from os.path import dirname

import cv2

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
# from common.NormalNN24 import NormalNN24
# from common.NormalNN24_2 import NormalNN24
from common.NormalNN24_4 import NormalNN24
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
        c0 = mu.binary(x0).to(device)
        xout = self.conv24_3(x0)

        # TODO: if any element of cout is less than a threshold epsilon, set it to 0.
        out = torch.cat((xout, c0, c0), 1)
        return out

    def minor_filter(self, tensor):
        eps = 0.01
        tensor[tensor < eps] = 0

        return tensor
