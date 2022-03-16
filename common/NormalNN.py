import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import math

"""
Normal Neuron Network
input: vertex (3,512,512)
output: normal  (3,512,512)

"""


class NormalNN(nn.Module):
    def __init__(self, in_ch, out_ch, num_channels=3):
        super().__init__()
        self.__name__ = 'NormalNN'
        kernel_5 = (5, 5)
        kernel_3 = (3, 3)
        padding_2 = (2, 2)
        padding_1 = (1, 1)
        self.le_relu = nn.LeakyReLU(0.1)
        self.dconv1 = nn.Conv2d(in_ch, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv2 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv3 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv4 = nn.Conv2d(in_ch * num_channels, 2 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv5 = nn.Conv2d(2 * in_ch * num_channels, 4 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv6 = nn.Conv2d(4 * in_ch * num_channels, 8 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv7 = nn.Conv2d(8 * in_ch * num_channels, 8 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv1 = nn.Conv2d(16 * in_ch * num_channels, 4 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv2 = nn.Conv2d(8 * in_ch * num_channels, 1 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv3 = nn.Conv2d(2 * in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv4 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv5 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.conv1 = nn.Conv2d(in_ch * num_channels, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x0):
        x1 = self.le_relu(self.dconv1(x0))  # (1,9,512,512)
        x1 = self.le_relu(self.dconv2(x1))  # (1,9,512,512)
        x1 = self.le_relu(self.dconv3(x1))  # (1,9,512,512)

        # Downsample 1
        ds = 2
        x1_ds, idx = F.max_pool2d(x1, ds, ds, return_indices=True)  # (1,9,256,256)
        x1_ds /= 4

        x2_ds = self.le_relu(self.dconv4(x1_ds))  # (1,18,256,256)
        x2_ds = self.le_relu(self.dconv5(x2_ds))  # (1,18,256,256)

        # Downsample 2
        ds = 2
        x2_dss, idx = F.max_pool2d(x2_ds, ds, ds, return_indices=True)  # (1,18,128,128)
        x2_dss /= 4

        x3_ds = self.le_relu(self.dconv6(x2_dss))  # (1,18,128,128)

        # Downsample 3
        ds = 2
        x3_dss, idx = F.max_pool2d(x3_ds, ds, ds, return_indices=True)  # (1,18,64,64)
        x3_dss /= 4

        x4_ds =self.le_relu( self.dconv7(x3_dss))  # (1,18,64,64)

        # Upsample 1
        x4 = F.interpolate(x4_ds, x3_ds.size()[2:], mode='nearest')  # (1,18,128,128)

        x34_ds =self.le_relu( self.uconv1(torch.cat((x3_ds, x4), 1)))  # (1, 9, 128, 128)

        # Upsample 2
        x34 = F.interpolate(x34_ds, x2_ds.size()[2:], mode='nearest')
        x23_ds = self.le_relu(self.uconv2(torch.cat((x2_ds, x34), 1)))  # (1, 9, 256, 256)

        # # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')  # (1, 9, 512, 512)
        xout = self.le_relu(self.uconv3(torch.cat((x23, x1), 1)))  # (1, 9, 512, 512)

        xout = self.conv1(xout)  # (1, 3, 512, 512)
        return xout

