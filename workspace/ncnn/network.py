import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.NormalGuided import NormalGuided
from common.NCNN import NCNN


class CNN(nn.Module):
    def __init__(self, channel_num=64):
        super().__init__()
        self.__name__ = 'ncnn'

        # input confidence estimation network
        self.lh = NCNN(3, 3, channel_num=channel_num)
        self.normal = NormalGuided(3, 3, channel_num=channel_num)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]
        x_light = x[:, 4:7, :, :]

        mask = torch.sum(torch.abs(x[:, :3, :, :]), dim=1) > 0
        c_in = mask.unsqueeze(1).float()

        l_out = self.lh(x_light, c_in)
        n_out = self.normal(x_vertex, x_img)

        out = torch.cat((n_out, l_out), 1)
        return out
