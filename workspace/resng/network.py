import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.ResNormalGuided import NormalGuided


class CNN(nn.Module):
    def __init__(self, c_num):
        super().__init__()
        self.__name__ = 'resng'

        # input confidence estimation network
        self.nconv3_3 = NormalGuided(3, 3, c_num)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]

        xout = self.nconv3_3(x_vertex, x_img)

        return xout
