import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.AlbedoGatedNNN import GNet2
import config
from help_funs import mu


class CNN(nn.Module):
    def __init__(self, channel_num, light_num, light_num_use):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        self.light_num = light_num
        self.light_num_use = light_num_use

        self.g_net = GNet2(3, 3, channel_num)
        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x0: vertex array
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3: 3 + self.light_num_use, :, :] / 255.0  # one image map
        x_light = x[:, 3 + self.light_num:3 + self.light_num + 3 * self.light_num_use, :, :]  # one light map

        # normal predict
        x_normal_out = self.g_net(x_vertex, x_light, x_img)

        return x_normal_out
