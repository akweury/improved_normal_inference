import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch.nn as nn
from common.NormalizedNNN import GCNN


class CNN(nn.Module):
    def __init__(self, c_num):
        super().__init__()
        self.__name__ = 'vertex'

        # input confidence estimation network
        self.vertex_inpainting = GCNN(3, 3, c_num)

    def forward(self, x):
        # x0: vertex array
        # c0: confidence of each element in x0
        x_vertex = x[:, :3, :, :]

        xout = self.vertex_inpainting(x_vertex)

        return xout
