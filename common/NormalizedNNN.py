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

        self.conv_g = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_f = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()

    def forward(self, x):
        # Normalized Convolution
        x_g = self.active_g(self.conv_g(x))
        x_f = self.active_f(self.conv_f(x))
        x = x_f * x_g
        return x


class NormalizedNNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.__name__ = 'nnnn'
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
        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()
        # self.active = nn.ReLU()

        self.epsilon = 1e-20
        channel_size_1 = 12
        channel_size_2 = 24
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def c_avg(self, cout, weight):
        # Propagate confidence
        # cout = denom.sum(dim=1, keepdim=True)
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)

        s = torch.sum(k, dim=-1, keepdim=True)
        # s = torch.sum(k)

        cout = cout / s
        cout = cout.view(sz)
        return cout

    def forward(self, x0, c0):
        x1 = self.dconv1(x0)
        x1 = self.dconv2(x1)
        x1 = self.dconv3(x1)

        # Downsample 1
        # ds = 2
        # x1_ds, idx = F.max_pool2d(x1, ds, ds, return_indices=True)  # 256,256
        x1_ds = self.dconv4(x1)
        x2 = self.dconv2(x1_ds)
        x2 = self.dconv3(x2)

        # Downsample 2
        # ds = 2
        # x2_ds, idx = F.max_pool2d(x2, ds, ds, return_indices=True)  # 128,128
        x2_ds = self.dconv4(x2)
        x3 = self.dconv2(x2_ds)
        x3 = self.dconv3(x3)

        # Downsample 3
        # ds = 2
        # x3_ds, idx = F.max_pool2d(x3, ds, ds, return_indices=True)  # 64,64
        x3_ds = self.dconv4(x3)
        x4 = self.dconv2(x3_ds)
        x4 = self.dconv3(x4)

        # Upsample 1
        x4_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x5 = self.uconv1(torch.cat((x3, x4_us), 1))

        # Upsample 2
        x5_us = F.interpolate(x5, x2.size()[2:], mode='nearest')
        x6 = self.uconv2(torch.cat((x2, x5_us), 1))

        # # Upsample 3
        x6_us = F.interpolate(x6, x1.size()[2:], mode='nearest')  # 512, 512
        x7 = self.uconv3(torch.cat((x1, x6_us), 1))

        xout = self.conv1(x7)  # 512, 512
        xout = self.conv2(xout)
        return xout, c0
