import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.AlbedoGatedNNN import AlbedoGatedNNN
from common.AlbedoGatedNNN import GConv


class LightNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.__name__ = 'lignet'
        kernel_size = (3, 3)
        padding_size = (1, 1)
        stride = (1, 1)

        channel_size_1 = 16
        self.active_leaky_relu = nn.LeakyReLU(0.01)
        self.lsInpainting1 = GConv(in_ch, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting2 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting3 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting4 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting5 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting6 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting7 = GConv(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting8 = nn.Conv2d(channel_size_1, channel_size_1, kernel_size, stride, padding_size)
        self.lsInpainting9 = nn.Conv2d(channel_size_1, out_ch, kernel_size, stride, padding_size)
        self.merge = nn.Conv2d(out_ch * 2, 3, kernel_size, stride, padding_size)
        self.prod1 = nn.Conv2d(3, 1, kernel_size, stride, padding_size)
        self.prod2 = nn.Conv2d(1, 1, kernel_size, stride, padding_size)

    def forward(self, lin, nin):
        L = self.lsInpainting1(lin)
        L = self.lsInpainting2(L)
        L = self.lsInpainting3(L)
        L = self.lsInpainting4(L)
        L = self.lsInpainting5(L)
        L = self.lsInpainting6(L)
        L = self.lsInpainting7(L)
        L = self.active_leaky_relu(self.lsInpainting8(L))
        xout_light = self.lsInpainting9(L)

        scaleProd = self.active_leaky_relu(self.merge(torch.cat((xout_light, nin), 1)))
        scaleProd = self.active_leaky_relu(self.prod1(scaleProd))
        scaleProd = self.prod2(scaleProd)
        return scaleProd


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'ag'

        self.agconv3_3 = AlbedoGatedNNN(3, 3)
        self.active_leaky_relu = nn.LeakyReLU(0.01)
        self.active_sigmoid = nn.Sigmoid()

        self.lightInpainting = LightNet(3, 3)

        self.nl_layer = nn.Conv2d(6, 1, (3, 3), (1, 1), (1, 1))
        self.rho_layer = GConv(2, 1, (3, 3), (1, 1), (1, 1))

        self.albedoInpainting1 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))
        self.albedoInpainting2 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))
        self.albedoInpainting3 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, xin):
        # x0: vertex array
        x0 = xin[:, :3, :, :]
        x_normal_out = self.agconv3_3(x0)

        # light source inpainting
        light_direction = xin[:, 3:6, :, :]

        scaleProd = self.lightInpainting(light_direction, x_normal_out)

        out = torch.cat((x_normal_out, scaleProd), 1)
        return out
