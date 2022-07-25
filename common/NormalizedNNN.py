import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


# Normalized Convolution Layer
class GConv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(GConv, self).__init__(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, False, (0, 0),
                                    groups, bias, padding_mode='zeros')

        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()

    def forward(self, x):
        # Normalized Convolution
        x_g = self.active_g(self.conv_g(x))
        x_f = self.active_f(self.conv_f(x))
        x = x_f * x_g
        return x


class GCNN(nn.Module):
    def __init__(self, in_ch, out_ch, c_num):
        super().__init__()
        kernel_down = (3, 3)
        kernel_up = (3, 3)
        padding_down = (1, 1)
        padding_up = (1, 1)
        stride = (1, 1)
        stride_2 = (2, 2)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()
        # self.active_img = nn.LeakyReLU(0.01)

        self.epsilon = 1e-20
        channel_size_1 = c_num
        channel_size_2 = c_num * 2
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        # self.resnet50 = ResNet(Bottleneck, layers=[1, 3, 4, 5])

        # branch 1
        self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x1):
        x1 = self.dconv1(x1)
        x1 = self.dconv2(x1)
        x1 = self.dconv3(x1)

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

        # Upsample 3
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x3 = self.uconv1(torch.cat((x3, x3_us), 1))

        # Upsample 4
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x2 = self.uconv2(torch.cat((x2, x2_us), 1))

        # Upsample 5
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        x1 = self.uconv3(torch.cat((x1, x1_us), 1))

        xout = self.conv1(x1)  # 512, 512
        xout = self.conv2(xout)

        return xout


class GCNN_NOC(nn.Module):
    def __init__(self, in_ch, out_ch, c_num):
        super().__init__()
        kernel_down = (3, 3)
        kernel_up = (3, 3)
        padding_down = (1, 1)
        padding_up = (1, 1)
        stride = (1, 1)
        stride_2 = (2, 2)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()
        # self.active_img = nn.LeakyReLU(0.01)

        self.epsilon = 1e-20
        channel_size_1 = c_num
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        # branch 1
        self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.uconv1 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_1, channel_size_1, kernel_up, stride, padding_up)

        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x1):
        x1 = self.dconv1(x1)
        x1 = self.dconv2(x1)
        x1 = self.dconv3(x1)

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

        # Upsample 3
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x3 = self.uconv1(x3_us)

        # Upsample 4
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x2 = self.uconv2(x2_us)

        # Upsample 5
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        x1 = self.uconv3(x1_us)

        xout = self.conv1(x1)  # 512, 512
        xout = self.conv2(xout)

        return xout
