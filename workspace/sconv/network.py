import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.StandardNNN import NormalNN


class CNN(nn.Module):
    def __init__(self, ch_size):
        super().__init__()
        self.__name__ = 'sconv'

        self.conv3_3 = NormalNN(3, 3, ch_size=ch_size)

    def forward(self, x0):
        xout = self.conv3_3(x0[:, :3, :, :])
        return xout
