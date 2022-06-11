import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.NormalizedNNN import NormalizedNNN


# from common.Nconv import NormalizedNet


class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.__name__ = 'nnnn'

        # input confidence estimation network
        self.nconv3_3 = NormalizedNNN(3, 3, channel_num)

    def forward(self, x0):
        # x0: vertex array
        xout = self.nconv3_3(x0[:, :3, :, :])
        return xout
