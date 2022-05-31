import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.AlbedoGatedNNN import AlbedoGatedNNN
from common.AlbedoGatedNNN import GConv


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
        light_direction = xin[:, 4:7, :, :]

        # x0: vertex array
        x_normal_out = self.agconv3_3(x0)

        # light source inpainting
        L = self.lsInpainting1(light_direction)
        L = self.lsInpainting2(L)
        L = self.active(self.lsInpainting3(L))

        # rho inpainting
        G = torch.abs(torch.sum(x_normal_out * L, dim=1, keepdim=True))
        rho = x_img0 / (G + 1e-20)
        rho = rho / rho.max()
        rho = self.albedoInpainting1(rho)
        rho = self.albedoInpainting2(rho)
        x_rho_out = self.active(self.albedoInpainting3(rho))

        out = torch.cat((x_normal_out, x_rho_out, L), 1)
        return out
