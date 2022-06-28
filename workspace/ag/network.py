import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.Layers import GConv
from common.ResNormalGuided import NormalGuided


class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.__name__ = 'ag'
        self.net3_3 = NormalGuided(3, 3, channel_num)
        self.albedo1 = GConv(6, 1, (3, 3), (1, 1), (1, 1))
        self.albedo2 = GConv(2, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x):
        # x0: vertex array
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]
        x_light = x[:, 4:7, :, :]
        input_mask = torch.sum(torch.abs(x_vertex), dim=1) > 0
        input_mask = input_mask.unsqueeze(1)
        x_normal_out = self.net3_3(x_vertex, x_img)

        x_g = self.albedo1(torch.cat((x_normal_out, x_light), 1))
        x_albedo = self.albedo2(torch.cat((x_g, x_img), 1))
        xout = torch.cat((x_normal_out, x_albedo, input_mask), 1)
        return xout
