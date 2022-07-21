import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.VIL10Net import DownSampling, VILNet
from common.Layers import GConv


# from common.NormalizedNNN import NormalizedNNN

class CNN(nn.Module):
    def __init__(self, channel_num, light_num, light_num_use):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        self.light_num = light_num
        self.light_num_use = light_num_use
        # self.light_net = NormalGuided(3, 3, channel_num)
        self.illusion_encoder = DownSampling(4, 3, channel_num)
        self.max_pool = GConv(self.light_num * self.channel_num, self.channel_num, (3, 3), (1, 1), (1, 1))
        self.normal_net = VILNet(3, 3, channel_num)

        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def forward(self, x):

        # extractor
        total_channel = self.light_num * self.channel_num
        x_light_feature_cat = torch.zeros(size=(x.size(0), total_channel, x.size(2) // 8, x.size(3) // 8)).to(x.device)
        for i in range(self.light_num_use):
            img = x[:, 3 + i:4 + i, :, :]
            light = x[:, 3 + self.light_num + 3 * i:6 + self.light_num + 3 * i, :, :]

            # encode image and light
            x_light_out = self.illusion_encoder(torch.cat((img, light), 1))

            # concat features for 10 image and light maps
            x_light_feature_cat[:, self.channel_num * i:self.channel_num * (i + 1), :, :] = x_light_out

        # max pooling layer
        x_light_feature = self.max_pool(x_light_feature_cat)

        # normal estimation
        x_vertex = x[:, :3, :, :]
        x_normal = self.normal_net(x_vertex, x_light_feature)

        return x_normal
