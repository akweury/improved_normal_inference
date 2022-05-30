import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.AlbedoGatedNNN import AlbedoGatedNNN
from common.AlbedoGatedNNN import GConv
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'ag'

        self.agconv3_3 = AlbedoGatedNNN(3, 3)
        self.active = nn.LeakyReLU(0.01)
        self.albedoInpainting1 = GConv(1, 1, (3, 3), (1, 1), (1, 1))
        self.albedoInpainting2 = GConv(1, 1, (3, 3), (1, 1), (1, 1))
        self.albedoInpainting3 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))
        self.lsInpainting1 = GConv(3, 3, (3, 3), (1, 1), (1, 1))
        self.lsInpainting2 = GConv(3, 3, (3, 3), (1, 1), (1, 1))
        self.lsInpainting3 = nn.Conv2d(3, 3, (3, 3), (1, 1), (1, 1))

    def forward(self, xin):
        x0 = xin[:, :3, :, :]
        x_img0 = xin[:, 3:4, :, :]
        light_source = xin[:, 4:7, :, :]

        # x0: vertex array
        x_normal_out = self.agconv3_3(x0)

        # light source inpainting
        L = mu.vertex2light_direction_tensor(x0, light_source)
        L = self.lsInpainting1(L)

        # rho inpainting
        rho = x_img0 / (torch.sum(x_normal_out * L, dim=1, keepdim=True) + 1e-20)
        x_rho_out = self.albedoInpainting1(rho)
        x_img_out = x_rho_out * (torch.sum(x_normal_out * L, dim=1, keepdim=True))

        out = torch.cat((x_normal_out, x_rho_out, x_img_out), 1)
        return out
