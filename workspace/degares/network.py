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
        xout_base = self.degares3_3(x)

        # sharp edge training
        mask_sharp_part_strict = torch.zeros(size=(x.size(0), 512, 512)).to(x.device)
        mask_sharp_part_extended = torch.zeros(size=(x.size(0), 512, 512)).to(x.device)
        for i in range(x.size(0)):
            mask_sharp_part_strict[i, :, :], mask_sharp_part_extended[i, :, :] = mu.hpf_torch(xout_base[i, :, :, :])
        mask_sharp_part_strict = mask_sharp_part_strict == 255
        mask_sharp_part_strict = mask_sharp_part_strict.unsqueeze(1).repeat(1, 3, 1, 1)

        mask_sharp_part_extended = mask_sharp_part_extended == 255
        mask_sharp_part_extended = mask_sharp_part_extended.unsqueeze(1).repeat(1, 3, 1, 1)

        # feed extended sharp part to model for training purpose
        x1 = torch.zeros(size=mask_sharp_part_extended.size()).to(x.device)
        x1[mask_sharp_part_extended] = x[:, :3, :, :][mask_sharp_part_extended]
        x_out_sharp = self.detailNet(x1)

        # only use the strict sharp part
        x_out_sharp[~mask_sharp_part_strict] = 0

        # set sharp edge in base to 0
        xout_base = xout_base.permute(0, 2, 3, 1)
        xout_base[(torch.sum(x_out_sharp, dim=1) > 0)] = 0
        xout_base = xout_base.permute(0, 3, 1, 2)

        return torch.cat((xout_base, x_out_sharp, x1), 1)
