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
from common.SVDNet import SVDNet
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'svdn'

        self.svdNN = SVDNet()

    def forward(self, x0):
        out = self.svdNN(x0)
        return out
