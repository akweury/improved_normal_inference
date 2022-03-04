########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################
import os, sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn

from pncnn.common.unet import UNetSP
from pncnn.common.nconv import NConvUNet
from help_funs.mu import binary


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'pncnn'

        # input confidence estimation network
        # self.conf_estimator = UNetSP(1, 1)
        self.nconv = NConvUNet(3, 3)
        self.var_estimator = UNetSP(3, 3)
        # self.var_estimator = UNetSP(3, 3)

    def forward(self, x0):
        # binary input
        # c0 = self.conf_estimator(x0) #  estimate the input confidence
        c0 = binary(x0)  # estimate the input confidence
        xout, cout = self.nconv(x0, c0, cpu=True)  # estimated value of depth

        # conv layer with kernal 1

        cout = self.var_estimator(cout)  #
        out = torch.cat((xout, cout, c0), 1)
        return out
