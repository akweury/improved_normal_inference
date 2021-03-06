import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd

from common.ResNet import ResNet
from common.ResNet import Bottleneck
from common.NCNN import NConv2d


# Normalized Convolution Layer
class NConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(NConv, self).__init__(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, False, (0, 0),
                                    groups, bias, padding_mode='zeros')

        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.nconv = NConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.active = nn.ReLU()

    def forward(self, x, cin=None, n=False):
        if n:
            # Normalized Convolution
            x, cout = self.nconv(x, cin)
        else:
            x = self.conv(x)

        x = self.active(x)
        return x


class NormalizedNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.__name__ = 'ncnn'
        kernel_down = (3, 3)
        kernel_down_2 = (5, 5)
        kernel_up = (3, 3)
        kernel_up_2 = (5, 5)
        padding_down = (1, 1)
        padding_down_2 = (2, 2)
        padding_down_3 = (4, 4)
        padding_down_4 = (8, 8)
        padding_down_5 = (16, 16)
        padding_up = (1, 1)
        padding_up_2 = (2, 2)
        stride = (1, 1)
        stride_2 = (2, 2)

        dilate1 = (2, 2)
        dilate2 = (4, 4)
        dilate3 = (8, 8)
        dilate4 = (16, 16)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()
        # self.active = nn.ReLU()

        self.epsilon = 1e-20
        channel_size_1 = 32
        channel_size_2 = 64

        self.resnet50 = ResNet(Bottleneck, layers=[1, 3, 4, 5])
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.
        # self.dconv1 = NConv2d(in_ch, channel_size_1, kernel_down, stride=stride, padding=padding_down)
        self.dconv1 = NConv(in_ch, channel_size_1, kernel_down, stride=stride, padding=padding_down)
        self.dconv2 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = NConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.dilated1 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down_2, dilation=dilate1)
        self.dilated2 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down_3, dilation=dilate2)
        self.dilated3 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down_4, dilation=dilate3)
        self.dilated4 = NConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down_5, dilation=dilate4)

        self.uconv1 = NConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = NConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = NConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv4 = NConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.conv1 = NConv(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = NConv(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x1, x_img_1, cin):
        # x1, c1 = self.dconv1(x1, cin)
        x1 = self.dconv1(x1, cin, n=False)
        x1 = self.dconv2(x1)
        x1 = self.dconv3(x1)

        x_img = self.resnet50(x_img_1)

        # Downsample 1
        x2 = self.dconv4(x1)
        x2 = self.dconv2(x2)
        x2 = self.dconv3(x2)

        # Downsample 2
        x3 = self.dconv4(x2)
        x3 = self.dconv2(x3)
        x3 = self.dconv3(x3)

        # Downsample 3
        x4 = self.dconv4(x3)
        x4 = self.dconv2(x4)
        x4 = self.dconv3(x4)

        # dilated conv
        x4 = self.dilated1(x4)
        x4 = self.dilated2(x4)
        x4 = self.dilated3(x4)
        x4 = self.dilated4(x4)
        x4 = self.dconv2(x4)
        x4 = self.dconv3(x4)

        # merge image feature and vertex feature
        x4 = torch.cat((x4, x_img), 1)

        # Upsample 1
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x3_mus = self.uconv1(x3_us)
        x3 = self.uconv2(torch.cat((x3, x3_mus), 1))

        # Upsample 2
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x2 = self.uconv3(torch.cat((x2, x2_us), 1))

        # # Upsample 3
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        x1 = self.uconv4(torch.cat((x1, x1_us), 1))

        xout = self.conv1(x1)  # 512, 512
        xout = self.conv2(xout)

        return xout, cin
