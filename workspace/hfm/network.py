import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.HFMNet import fconv_ms, map_conv
import torch
import numpy as np


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'hfm'

        # input confidence estimation network
        self.net3_3 = fconv_ms(input_channel1=1, input_channel2=3, output_channel=3, track_running_static=True)
        self.conf_net = map_conv(input_channel=4, output_channel=1, track_running_static=True)
        # self.net3_3 = NormalizedNNN(3, 3, channel_num)

    def forward(self, x0):
        # x0: vertex array
        vertex = x0[:, :3, :, :]
        img = x0[:, 3:4, :, :]

        mask = torch.sum(torch.abs(vertex[:, :3, :, :]), dim=1) > 0
        conf_net_input = torch.cat((vertex, mask[:, np.newaxis, :, :]), dim=1)
        vertex_conf = self.conf_net(conf_net_input)
        output_f, output_f1, output_f2, output_f3, output_d = self.net3_3(img, vertex, vertex_conf)

        return output_f
