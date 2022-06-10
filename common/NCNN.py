import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


class NCNN(nn.Module):
    def __init__(self, in_ch, out_ch, num_channels=64):
        super().__init__()
        self.__name__ = 'NCNN'

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.nconv1 = NConv2d(in_ch, num_channels, (3, 3))
        self.nconv2 = NConv2d(num_channels, num_channels, (3, 3))
        self.nconv3 = NConv2d(num_channels, num_channels, (3, 3))
        self.nconv4 = NConv2d(num_channels, num_channels, (3, 3))
        self.nconv5 = NConv2d(num_channels, num_channels, (3, 3))

        self.conv1 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))
        self.conv3 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))
        self.conv4 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))
        self.conv5 = nn.Conv2d(num_channels, num_channels, (3, 3), padding=(1, 1))

        self.conv6 = nn.Conv2d(num_channels, out_ch, (1, 1))

    def forward(self, x0, c0):
        x1, c1 = self.nconv1(x0, c0)
        x1 = self.relu(x1)

        x1, c1 = self.nconv2(x1, c1)
        x1 = self.relu(x1)
        x1, c1 = self.nconv3(x1, c1)
        x1 = self.relu(x1)
        x1, c1 = self.nconv4(x1, c1)
        x1 = self.relu(x1)
        x1, c1 = self.nconv5(x1, c1)
        x1 = self.relu(x1)

        x2 = self.relu(self.conv1(x1))
        x2 = self.relu(self.conv2(x2))
        x2 = self.relu(self.conv3(x2))
        x2 = self.relu(self.conv4(x2))
        x2 = self.relu(self.conv5(x2))

        xout = self.sigmoid(self.conv6(x2))

        return xout, c1


# Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(1, 1), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, (0, 0),
                                      groups, bias, padding_mode='zeros')

        self.eps = 1e-20
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, data, conf):
        # Normalized Convolution
        # denom = F.conv2d(conf, self.weight, None, self.stride,
        #                  self.padding, self.dilation, self.groups)
        # nomin = F.conv2d(data * conf, self.weight, None, self.stride,
        #                  self.padding, self.dilation, self.groups)
        channel_num = data.size(1)
        conf = conf.repeat(1, channel_num, 1, 1)
        denom = self.conv(conf)
        nomin = self.conv(data * conf)

        nconv = nomin / (denom + self.eps)

        # Add bias
        nconv += self.bias.view(1, self.bias.size(0), 1, 1).expand_as(nconv)
        conf = torch.sum(conf, dim=1, keepdim=True)
        # Propagate confidence
        cout = F.max_pool2d(conf, self.kernel_size, self.stride, self.padding)

        return nconv, cout
