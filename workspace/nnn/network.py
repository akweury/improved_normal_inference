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
        self.conv3_3 = NormalNN(3, 3)
        # self.conv1_1 = NormalNN(1, 1)

    def forward(self, x0):
        # x0: vertex array
        # c0: confidence of each element in x0

        # c0 = self.conf_estimator(x0) #  estimate the input confidence

        device = x0.get_device()

        # c0 = self.conf_estimator3_3(x0).to(device)  # estimate the input confidence
        c0 = mu.binary(x0).to(device)
        xout = self.conv3_3(x0)
        # xout_0 = self.conv1_1(x0[:, 0:1, :, :])
        # xout_1 = self.conv1_1(x0[:, 1:2, :, :])
        # xout_2 = self.conv1_1(x0[:, 2:3, :, :])
        # xout = torch.cat((xout_0, xout_1, xout_2), 1)
        # xout = self.conv3_3(x0)
        # xout, cout = self.nconv3_3(x0, c0)
        # cout = self.var_estimator3_3(c0)

        # TODO: if any element of cout is less than a threshold epsilon, set it to 0.
        out = torch.cat((xout, c0, c0), 1)
        return out

    def minor_filter(self, tensor):
        eps = 0.01
        tensor[tensor < eps] = 0

        return tensor
