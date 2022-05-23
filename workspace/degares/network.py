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
        # general surface training
        xout = self.degares3_3(x)

        mask_sharp_part = torch.zeros(size=(x.size(0), 512, 512), dtype=torch.bool).to(x.device)
        # detail edge training
        for i in range(x.size(0)):
            mask_sharp_part[i, :, :] = ~mu.hpf_torch(x[i, 3:, :, :])

        mask_sharp_part = mask_sharp_part.unsqueeze(1).repeat(1, 3, 1, 1)
        x_out_sharp = self.detailNet(x[:, :3, :, :][mask_sharp_part], mask_sharp_part)

        return torch.cat((xout, x_out_sharp, mask_sharp_part), 1)
