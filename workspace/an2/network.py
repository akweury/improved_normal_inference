import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.NormalizedNNN import GCNN
from common.AlbedoGatedNNN import GNet, GNetF3F, GNetF3B, GNetF2F, GNetF2B, GNetF1B, GNetF1F
import config
from help_funs import mu

class CNN(nn.Module):
    def __init__(self, channel_num, light_num, light_num_use, net_type):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        self.light_num = light_num
        self.light_num_use = light_num_use
        # self.light_net = NormalGuided(3, 3, channel_num)
        # self.normal_net = GCNN(3, 3, channel_num)

        if net_type == "gnet-f4":
            self.g_net = GNet(3, 3, channel_num)
        elif net_type == "gnet-f3f":
            self.g_net = GNetF3F(3, 3, channel_num)
        elif net_type == "gnet-f3b":
            self.g_net = GNetF3B(3, 3, channel_num)
        elif net_type == "gnet-f2f":
            self.g_net = GNetF2F(3, 3, channel_num)
        elif net_type == "gnet-f2b":
            self.g_net = GNetF2B(3, 3, channel_num)
        elif net_type == "gnet-f1f":
            self.g_net = GNetF1F(3, 3, channel_num)
        elif net_type == "gnet-f1b":
            self.g_net = GNetF1B(3, 3, channel_num)

        self.last_conv = nn.Conv2d(3, 3, (1, 1), (1, 1), (0, 0))
        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    def init_net(self):
        normal_source_net = GNet(3, 3, self.channel_num)
        normal_checkpoint = torch.load(config.an2_trip_net_remote)
        normal_source_net.load_state_dict(normal_checkpoint['model'].g_net.state_dict())
        normal_source_net_dict = normal_source_net.state_dict()
        normal_net_dict = self.g_net.state_dict()
        normal_source_net_dict = {k: v for k, v in normal_source_net_dict.items() if
                                  k in normal_net_dict and v.size() == normal_net_dict[k].size()}
        normal_net_dict.update(normal_source_net_dict)
        self.g_net.load_state_dict(normal_net_dict)

    def forward(self, x):
        # x0: vertex array
        self.light_num = 1
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3: 3 + self.light_num_use, :, :] / 255.0  # one image map
        x_light = x[:, 3 + self.light_num:3 + self.light_num + 3 * self.light_num_use, :, :]  # one light map

        # normal predict
        x_normal_out = self.g_net(x_vertex, x_light, x_img)

        return x_normal_out
