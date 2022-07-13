import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.ResNormalGuided import NormalGuided
from common.AlbedoGatedNNN import GNet
import config


# from common.NormalizedNNN import NormalizedNNN

class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        self.light_net = NormalGuided(3, 3, channel_num)
        self.g_net = GNet(3, 3, channel_num)
        self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def init_net(self):
        light_source_net = NormalGuided(3, 3, self.channel_num)
        light_checkpoint = torch.load(config.light_3_32)

        light_source_net.load_state_dict(light_checkpoint['model'].light3_3.state_dict())
        light_source_net_dict = light_source_net.state_dict()
        light_net_dict = self.light_net.state_dict()
        light_source_net_dict = {k: v for k, v in light_source_net_dict.items() if
                                 k in light_net_dict and v.size() == light_net_dict[k].size()}
        light_net_dict.update(light_source_net_dict)
        self.light_net.load_state_dict(light_net_dict)

    def forward(self, x):
        # x0: vertex array
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]
        x_light = x[:, 4:7, :, :]
        input_mask = torch.sum(torch.abs(x_vertex), dim=1) > 0
        input_mask = input_mask.unsqueeze(1)

        # light inpaint
        x_light_out = self.light_net(x_light)

        # albedo predict
        x_g_out = self.g_net(x_vertex, x_light_out, x_img)

        # x_normal_out = (x_img / x_albedo_out) / x_light_out

        xout = torch.cat((x_light_out, x_g_out), 1)
        return xout
