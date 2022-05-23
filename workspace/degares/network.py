import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.DeGaResNet import DeGaResNet
from common.UNet import UNet
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'DeGaRes'

        # input confidence estimation network
        self.degares3_3 = DeGaResNet(3, 3)
        self.detailNet = UNet(3, 3)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]  # vertex
        x_img = x[:, 3:4, :, :]  # gray scale image
        # general surface training
        xout = self.degares3_3(x_vertex, x_img)

        mask_sharp_part = torch.zeros(size=(xout.size(0), 512, 512), dtype=torch.bool).to(x.device)
        # detail edge training
        for i in range(x.size(0)):
            mask_sharp_part[i, :, :] = ~mu.hpf_torch(xout[i, :, :, :])

        mask_sharp_part = mask_sharp_part.unsqueeze(1).repeat(1, 3, 1, 1)
        x_edge = torch.zeros(size=xout.size()).to(x.device)
        x_edge[mask_sharp_part] = x[:, :3, :, :][mask_sharp_part]
        x_out_sharp = self.detailNet(x_edge)

        xout = torch.cat((xout, x_out_sharp, mask_sharp_part), 1)

        return xout