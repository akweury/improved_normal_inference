import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.ResNormalGuided import NormalGuided
from common.AlbedoGatedNNN import GNet
from common.VIL10Net import VIL10Net
import config
from help_funs import mu


# from common.NormalizedNNN import NormalizedNNN

class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        # self.light_net = NormalGuided(3, 3, channel_num)
        self.normal_net = VIL10Net(3, 3, channel_num)
        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x0: vertex array
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:8, :, :]
        x_light = x[:, 8:23, :, :]
        # albedo predict
        x_normal_out = self.normal_net(x_vertex, x_img, x_light)
        return x_normal_out
