import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.ResNormalGuided import NormalGuided
from common.AlbedoGatedNNN import GNet
from common.I5Net import I5Net
import config
from help_funs import mu


# from common.NormalizedNNN import NormalizedNNN

class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        # self.__name__ = 'albedoGated'
        self.channel_num = channel_num
        # self.light_net = NormalGuided(3, 3, channel_num)
        self.normal_net = I5Net(5, 3, channel_num)
        # self.remove_grad()

    def remove_grad(self):
        for param in self.light_net.parameters():
            param.requires_grad = False

    # def init_net(self):
    #     light_source_net = NormalGuided(3, 3, self.channel_num)
    #     light_checkpoint = torch.load(config.light_3_32)
    #
    #     light_source_net.load_state_dict(light_checkpoint['model'].light3_3.state_dict())
    #     light_source_net_dict = light_source_net.state_dict()
    #     light_net_dict = self.g_net.state_dict()
    #
    #     light_source_net_dict = mu.change_light_dict_name(light_source_net_dict, light_net_dict, "l_")
    #
    #     light_source_net_dict = {k: v for k, v in light_source_net_dict.items() if
    #                              k in light_net_dict and v.size() == light_net_dict[k].size()}
    #
    #     light_net_dict.update(light_source_net_dict)
    #     self.g_net.load_state_dict(light_net_dict)
    #
    #     normal_source_net = NormalGuided(3, 3, self.channel_num)
    #     normal_checkpoint = torch.load(config.gcnn_3_32)
    #     normal_source_net.load_state_dict(normal_checkpoint['model'].nconv3_3.state_dict())
    #     normal_source_net_dict = normal_source_net.state_dict()
    #     normal_net_dict = self.g_net.state_dict()
    #     normal_source_net_dict = {k: v for k, v in normal_source_net_dict.items() if
    #                               k in normal_net_dict and v.size() == normal_net_dict[k].size()}
    #     normal_net_dict.update(normal_source_net_dict)
    #     self.g_net.load_state_dict(normal_net_dict)

    def forward(self, x):
        # x0: vertex array

        x_img = x[:, 3:8, :, :]
        # albedo predict
        x_normal_out = self.normal_net(x_img)
        return x_normal_out
