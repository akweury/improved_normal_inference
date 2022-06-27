import torch
import torch.nn as nn
import torch.nn.functional as F

from common.Layers import GConv


class FGCNN(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        self.__name__ = 'fgcnn'
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

        # self.epsilon = 1e-20
        channel_size_1 = channel_num
        channel_size_2 = channel_num * 2
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        # intro
        self.init1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.init2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.init3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)

        # downsampling 1
        self.d1l1 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
        self.d1l2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d1l3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d1l4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        #
        # # downsampling 2
        self.d2l1 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
        self.d2l2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d2l3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d2l4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        #
        # # downsampling 3
        self.d3l1 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
        self.d3l2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d3l3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d3l4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)

        # # downsampling 4
        self.d4l1 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
        self.d4l2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d4l3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d4l4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)

        # # downsampling 5
        self.d5l1 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
        self.d5l2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d5l3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.d5l4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)

        self.uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv4 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv5 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        # # upsampling 1
        # self.u1l1 = Transp(channel_size_4, channel_size_3, kernel_up, stride_2, padding_up)
        # self.u1l2 = GConv(channel_size_3, channel_size_3, kernel_up, stride, padding_up)
        # self.u1l3 = GConv(channel_size_3, channel_size_3, kernel_up, stride, padding_up)
        #
        # # upsampling 2
        # self.u2l1 = Transp(channel_size_3, channel_size_1, kernel_up, stride_2, padding_up)
        # self.u2l2 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)
        # self.u2l3 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)
        #
        # # upsampling 3
        # self.u3l1 = Transp(channel_size_1, channel_size_1, kernel_up, stride_2, padding_up)
        # self.u3l2 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)
        # self.u3l3 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)

        # outro
        self.out1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.out2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, xin):
        x1 = self.init1(xin)
        x1 = self.d1l2(x1)
        x1 = self.d1l3(x1)

        # Downsample 1
        x2 = self.d1l1(x1)
        x2 = self.d1l2(x2)
        x2 = self.d1l3(x2)
        x2 = self.d1l4(x2)

        # Downsample 2
        x3 = self.d2l1(x2)
        x3 = self.d2l2(x3)
        x3 = self.d2l3(x3)
        x3 = self.d2l4(x3)

        # Downsample 3
        x4 = self.d3l1(x3)
        x4 = self.d3l2(x4)
        x4 = self.d3l3(x4)
        x4 = self.d3l4(x4)

        # Downsample 4
        x5 = self.d4l1(x4)
        x5 = self.d4l2(x5)
        x5 = self.d4l3(x5)
        x5 = self.d4l4(x5)

        # Downsample 5
        x6 = self.d5l1(x5)
        x6 = self.d5l2(x6)
        x6 = self.d5l3(x6)
        x6 = self.d5l4(x6)

        # Upsample 1
        x5_us = F.interpolate(x6, x5.size()[2:], mode='nearest')
        x5 = self.uconv1(torch.cat((x5, x5_us), 1))

        # Upsample 2
        x4_us = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x4 = self.uconv1(torch.cat((x4, x4_us), 1))

        # Upsample 3
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x3 = self.uconv1(torch.cat((x3, x3_us), 1))

        # Upsample 4
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x2 = self.uconv2(torch.cat((x2, x2_us), 1))

        # # Upsample 5
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        x1 = self.uconv3(torch.cat((x1, x1_us), 1))

        # # Upsample 1
        # x3 = self.u1l1(x4)
        # x3 = self.u1l2(x3)
        # # x3 = self.u1l3(x3)
        #
        # # Upsample 2
        # x2 = self.u2l1(x3)
        # x2 = self.u2l2(x2)
        # # x2 = self.u2l3(x2)
        #
        # # # Upsample 3
        # x1 = self.u3l1(x2)
        # x1 = self.u3l2(x1)
        # # x1 = self.u3l3(x1)

        xout = self.out1(x1)  # 512, 512
        xout = self.out2(xout)

        return xout
