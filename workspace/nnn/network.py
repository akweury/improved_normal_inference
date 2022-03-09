import sys
from os.path import dirname

import cv2

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.NormalNN import NormalNN
from help_funs import mu
from pncnn.common.nconv import NConvUNet
from pncnn.common.unet import UNetSP


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'nnn'

        # input confidence estimation network
        # self.conf_estimator = UNetSP(1, 1)
        self.conv3_3 = NormalNN(3, 3)
        self.nconv3_3 = NConvUNet(3, 3)
        self.nconv1_1 = NConvUNet(1, 1)

        self.var_estimator3_3 = UNetSP(3, 3)
        self.var_estimator1_1 = UNetSP(1, 1)

    def forward(self, x0, cpu):
        # x0: vertex array
        # c0: confidence of each element in x0

        # c0 = self.conf_estimator(x0) #  estimate the input confidence

        device = "cpu" if cpu else x0.get_device()
        c0 = mu.binary(x0).to(device)

        # out = self.conv(x0)

        # channels seperate training
        out_0, cout_0 = self.nconv1_1(x0[:, 0:1, :, :], c0[:, 0:1, :, :], cpu)  # estimated value of depth
        out_1, cout_1 = self.nconv1_1(x0[:, 1:2, :, :], c0[:, 1:2, :, :], cpu)  # estimated value of depth
        out_2, cout_2 = self.nconv1_1(x0[:, 2:3, :, :], c0[:, 2:3, :, :], cpu)  # estimated value of depth
        xout = torch.cat((out_0, out_1, out_2), 1)

        cout_0 = self.var_estimator1_1(cout_0)
        cout_1 = self.var_estimator1_1(cout_1)
        cout_2 = self.var_estimator1_1(cout_2)
        cout = torch.cat((cout_0, cout_1, cout_2), 1)

        # # channel combine training
        # xout, cout = self.nconv3_3(x0, c0, cpu)
        # cout = self.var_estimator3_3(cout)

        # TODO: if any element of cout is less than a threshold epsilon, set it to 0.

        return torch.cat((xout, cout, c0), 1)

    def minor_filter(self, tensor):
        eps = 0.01
        tensor[tensor < eps] = 0

        return tensor
