import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.NormalizedNNN import GCNN
from common.StandardNNN import NormalNN


class CNN(nn.Module):
    def __init__(self, c_num, light_num, net_type):
        super().__init__()
        # self.__name__ = 'light'
        self.light_num = light_num

        # input confidence estimation network
        if net_type == "gcnn":
            self.light3_3 = GCNN(3, 3, c_num)
        elif net_type == "cnn":
            self.light3_3 = NormalNN(3, 3, c_num)

    def forward(self, x):
        # general surface training
        x_light = x[:, 3 + self.light_num:6 + self.light_num, :, :]
        xout_light = self.light3_3(x_light)
        return xout_light
