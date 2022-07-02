import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import torch
import torch.nn as nn
from common.Layers import GConv
from common.ResNormalGuided import NormalGuided
from common.NormalizedNNN import NormalizedNNN
import config

class CNN(nn.Module):
    def __init__(self, channel_num):
        super().__init__()
        self.__name__ = 'ag'
        self.channel_num = channel_num
        self.net3_3 = NormalizedNNN(3, 3, channel_num)
        self.albedo1 = GConv(6, 1, (3, 3), (1, 1), (1, 1))
        self.albedo2 = GConv(2, 1, (3, 3), (1, 1), (1, 1))

    def init_net(self, model_name):
        net_dict = self.net3_3.state_dict()
        source_net = NormalizedNNN(3, 3, self.channel_num)
        checkpoint = torch.load(eval(f"config.{model_name}"))

        source_net.load_state_dict(checkpoint['model'].nconv3_3)
        source_net_dict = source_net.state_dict()
        source_net_dict = {k: v for k, v in source_net_dict.items() if k in net_dict and v.size() == net_dict[k].size()}
        net_dict.update(source_net_dict)
        self.net3_3.load_state_dict(net_dict)

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
