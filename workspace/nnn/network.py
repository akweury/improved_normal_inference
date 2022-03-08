import sys
from os.path import dirname

import cv2

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.NormalNN import NormalNN
from help_funs import mu
from pncnn.common.nconv import NConvUNet


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'nnn'

        # input confidence estimation network
        # self.conf_estimator = UNetSP(1, 1)
        self.conv = NormalNN(3, 3)
        self.nconv = NConvUNet(3, 3)
        # self.var_estimator = UNetSP(3, 3)
        # self.var_estimator = UNetSP(3, 3)

    def forward(self, x0, cpu):
        # x0: vertex array
        # c0: confidence of each element in x0

        # c0 = self.conf_estimator(x0) #  estimate the input confidence
        c0 = mu.binary(x0)

        # out = self.conv(x0)
        out, cout = self.nconv(x0, c0, cpu)  # estimated value of depth

        # conv layer with kernal 1
        # cout = self.var_estimator(cout)  #
        # out = torch.cat((xout, cout, c0), 1)

        # TODO: if any element of cout is less than a threshold epsilon, set it to 0.
        cout = self.minor_filter(cout)
        return out, cout, c0

    def minor_filter(self, tensor):
        eps = 0.01
        tensor[tensor < eps] = 0

        return tensor
