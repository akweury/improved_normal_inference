import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.Nconv import NormalizedNet


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'nconv'

        # input confidence estimation network
        self.nconv3_3 = NormalizedNet(3, 3)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]

        mask = torch.sum(torch.abs(x[:, :3, :, :]), dim=1) > 0
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1).float()

        xout, cout = self.nconv3_3(x_vertex, x_img, mask)
        out = torch.cat((xout, cout, x_img), 1)
        return out
