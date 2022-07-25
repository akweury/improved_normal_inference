import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.NormalizedNNN import GCNN, GCNN_NOC
from common.StandardNNN import NormalNN


# from common.Nconv import NormalizedNet


class CNN(nn.Module):
    def __init__(self, channel_num, net_type):
        super().__init__()
        self.__name__ = 'nnnn'

        if net_type == "gcnn":
            self.normal_net = GCNN(3, 3, channel_num)
        elif net_type == "cnn":
            self.normal_net = NormalNN(3, 3, channel_num)
        elif net_type == "gcnn_noc":
            self.normal_net = GCNN_NOC(3, 3, channel_num)

    def forward(self, x0):
        # x0: vertex array
        xout = self.normal_net(x0[:, :3, :, :])
        return xout
