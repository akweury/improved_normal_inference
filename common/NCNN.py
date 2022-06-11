import torch
import torch.nn as nn
import torch.nn.functional as F

from common.Layers import NConv2d


class NCNN(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        self.__name__ = 'ncnn'
        kernel_down = (3, 3)
        kernel_down_2 = (5, 5)
        kernel_up = (3, 3)
        kernel_up_2 = (5, 5)
        padding_down = (1, 1)
        padding_down_2 = (2, 2)
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
        self.active_img = nn.LeakyReLU(0.01)

        self.epsilon = 1e-20
        channel_size_2 = channel_num * 2
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        # branch 1
        self.dconv1 = NConv2d(in_ch, channel_num, kernel_down, stride, padding_down)
        self.dconv2 = NConv2d(channel_num, channel_num, kernel_down, stride, padding_down)
        self.dconv3 = NConv2d(channel_num, channel_num, kernel_down, stride, padding_down)
        self.dconv4 = NConv2d(channel_num, channel_num, kernel_down, stride_2, padding_down)

        self.uconv1 = NConv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv2 = NConv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv3 = NConv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv4 = NConv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)

        self.conv1 = nn.Conv2d(channel_num, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x0, cin):
        x1, c1 = self.dconv1(x0, cin)
        x1, c1 = self.dconv2(x1, c1)
        x1, c1 = self.dconv3(x1, c1)

        # Downsample 1
        x2, c2 = self.dconv4(x1, c1)
        x2, c2 = self.dconv2(x2, c2)
        x2, c2 = self.dconv3(x2, c2)

        # Downsample 2
        x3, c3 = self.dconv4(x2, c2)
        x3, c3 = self.dconv2(x3, c3)
        x3, c3 = self.dconv3(x3, c3)

        # Downsample 3
        x4, c4 = self.dconv4(x3, c3)
        x4, c4 = self.dconv2(x4, c4)
        x4, c4 = self.dconv3(x4, c4)

        # Upsample 1
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        c3_us = F.interpolate(c4, c3.size()[2:], mode='nearest')  # 128,128
        x3, c3 = self.uconv1(torch.cat((x3, x3_us), 1), torch.cat((c3, c3_us), 1))

        # Upsample 2
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        c2_us = F.interpolate(c3, c2.size()[2:], mode='nearest')
        x2, c2 = self.uconv2(torch.cat((x2, x2_us), 1), torch.cat((c2, c2_us), 1))

        # # Upsample 3
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        c1_us = F.interpolate(c2, c1.size()[2:], mode='nearest')  # 512, 512
        x1, c1 = self.uconv3(torch.cat((x1, x1_us), 1), torch.cat((c1, c1_us), 1))

        xout = self.active_g(self.conv1(x1))  # 512, 512
        xout = self.active_g(self.conv2(xout))

        return xout
