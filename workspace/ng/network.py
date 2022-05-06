import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
import torch.nn as nn
from common.NormalGuided import NormalGuided
from help_funs import mu


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'ng'

        # input confidence estimation network
        self.nconv3_3 = NormalGuided(3, 3)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]
        x_img = x[:, 3:4, :, :]
        device = x_vertex.get_device()
        c0 = mu.binary(x_vertex).to(device)

        xout, cout = self.nconv3_3(x_vertex, x_img, c0)
        # TODO: if any element of cout is less than a threshold epsilon, set it to 0.
        out = torch.cat((xout, cout, x_img), 1)
        return out

    def minor_filter(self, tensor):
        eps = 0.01
        tensor[tensor < eps] = 0

        return tensor
