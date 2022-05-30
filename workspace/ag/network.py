import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.AlbedoGatedNNN import AlbedoGatedNNN


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'ag'

        self.agconv3_3 = AlbedoGatedNNN(3, 3)

    def forward(self, x0):
        # x0: vertex array
        xout, x_rho, x_img = self.agconv3_3(x0)
        out = torch.cat((xout, x_rho, x_img), 1)
        return out
