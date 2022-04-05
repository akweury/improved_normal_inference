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
from common.NormalNN24 import NormalNN24


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'nnnx'

        self.svdNN = SVDNet()
        self.nnn24 = NormalNN24(24, 3)

    def forward(self, x0):
        # out = self.svdNN(x0)
        best_8neighbor_vectors = self.svdNN(x0)
        out = self.nnn24(best_8neighbor_vectors)
        return out
