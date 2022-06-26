import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn

from common.ResNormalGuided import NormalGuided


class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.__name__ = 'ag'
        self.net3_3 = NormalGuided(3, 3, channel_num)

    def forward(self, x):
        # x0: vertex array
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]

        xout = self.net3_3(x_vertex, x_img)
        return xout
