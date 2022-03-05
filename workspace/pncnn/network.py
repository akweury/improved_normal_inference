########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################
import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn

from pncnn.common.unet import UNetSP
from pncnn.common.nconv import NConvUNet
from help_funs import mu


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
        # x0: vertex array
        # c0: confidence of each element in x0

        # c0 = self.conf_estimator(x0) #  estimate the input confidence
        c0 = mu.binary(x0)

        _, normal_knn_8bit = mu.vertex2normal(mu.tenor2numpy(x0), k_idx=2)

        output_1 = [mu.normalize2_8bit(mu.tenor2numpy(x0)), mu.normalize2_8bit(mu.tenor2numpy(c0)), normal_knn_8bit]
        mu.show_horizontal(output_1, "input x0-c0-normal_knn")

        xout, cout = self.nconv(x0, c0, cpu=True)  # estimated value of depth

        # conv layer with kernal 1

        cout = self.var_estimator(cout)  #
        out = torch.cat((xout, cout, c0), 1)
        return out
