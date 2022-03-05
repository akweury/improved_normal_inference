########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################
import sys
from os.path import dirname

import cv2

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
        xout, cout = self.nconv(x0, c0)  # estimated value of depth

        # conv layer with kernal 1
        cout = self.var_estimator(cout)  #
        out = torch.cat((xout, cout, c0), 1)

        x0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(x0))
        mu.addText(x0_normalized_8bit, "x0")
        c0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(c0))
        mu.addText(c0_normalized_8bit, "c0")
        _, normal_knn_8bit = mu.vertex2normal(mu.tenor2numpy(x0), k_idx=2)
        mu.addText(normal_knn_8bit, "normal(knn)")
        output_img = cv2.hconcat([x0_normalized_8bit, c0_normalized_8bit, normal_knn_8bit])

        return out, output_img
