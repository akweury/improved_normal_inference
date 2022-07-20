import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.ResNormalGuided import NormalGuided


class CNN(nn.Module):
    def __init__(self, c_num, light_num):
        super().__init__()
        # self.__name__ = 'light'
        self.light_num = light_num

        # input confidence estimation network
        self.light3_3 = NormalGuided(3 * light_num, 3 * light_num, c_num)

    def forward(self, x):
        # general surface training
        x_light = x[:, 3 + self.light_num:3 + self.light_num + self.light_num * 3, :, :]
        xout_light = self.light3_3(x_light)
        return xout_light
