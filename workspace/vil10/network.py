import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.VIL10Net import DownSampling
from common.Layers import GConv


# from common.NormalizedNNN import NormalizedNNN

class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        # self.light_net = NormalGuided(3, 3, channel_num)
        self.illusion_encoder = DownSampling(4, 3, channel_num)
        self.normal_net = DownSampling(3, 3, channel_num)
        self.max_pool = GConv(300, 32, (3, 3), (2, 2), (1, 1))

        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def forward(self, x, light_num):
        # x0: vertex array

        total_channel = light_num * 3

        x_light_feature = torch.zeros(size=(x.size(0), total_channel, x.size(2), x.size(3)))
        for i in range(light_num):
            img = x[:, 3 + i:4 + i, :, :]
            light = x[:, 3 + light_num + 3 * i:6 + light_num + 3 * i, :, :]
            x_light_out = self.illusion_encoder(torch.cat((img, light), 1))
            x_light_feature[:, 32 * i, 32 * (i + 1), :, :] = x_light_out
        # max pooling layer
        x_light_feature = self.max_pool(x_light_feature)

        x_vertex = x[:, :3, :, :]
        x_normal = self.normal_net(x_vertex, x_light_feature)

        return x_normal
