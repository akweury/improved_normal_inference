import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd


class Conv(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True, active_function="LeakyReLU"):
        # Call _ConvNd constructor
        super(Conv, self).__init__(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, False, (0, 0),
                                   groups, bias, padding_mode='zeros')

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)

        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.active_name = active_function
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.bn1(self.conv(x))

        if self.active_name == "LeakyReLU":
            return self.active_LeakyReLU(x)
        elif self.active_name == "Sigmoid":
            return self.active_Sigmoid(x)
        elif self.active_name == "ReLU":
            return self.active_ReLU(x)
        elif self.active_name == "Tanh":
            return self.active_Tanh(x)
        elif self.active_name == "":
            return self.conv(x)
        else:
            raise ValueError


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


def conv1x1(in_planes: int, out_planes: int, stride) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


def gconv1x1(in_planes: int, out_planes: int, stride) -> GConv:
    """1x1 convolution"""
    return GConv(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


class GTransp(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(GTransp, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, False, (0, 0),
                                      groups, bias, padding_mode='zeros')

        self.conv_g = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(1, 1))
        self.conv_f = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(1, 1))

        self.active_f = nn.LeakyReLU(0.01)
        self.active_g = nn.Sigmoid()
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Normalized Convolution
        x_g = self.active_g(self.conv_g(x))
        x_f = self.active_f(self.conv_f(x))
        x = x_f * x_g
        return x


def gtransp1x1(in_planes: int, out_planes: int, stride) -> GTransp:
    """1x1 convolution"""
    return GTransp(in_planes, out_planes, kernel_size=(1, 1), stride=stride, bias=False)


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
        self.active_LeakyReLU = nn.LeakyReLU(0.01)

    def forward(self, data, conf):

        channel_num = data.size(1)
        if conf.size(1) == 1:
            conf = conf.repeat(1, channel_num, 1, 1)
        else:
            conf_0 = conf[:, :1, :, :].repeat(1, channel_num // 2, 1, 1)
            conf_1 = conf[:, 1:2, :, :].repeat(1, channel_num // 2, 1, 1)
            conf = torch.cat((conf_0, conf_1), 1)

        denom = self.conv(conf)
        nomin = self.conv(data * conf)

        nconv = nomin / (denom + self.eps)

        # Add bias
        nconv += self.bias.view(1, self.bias.size(0), 1, 1).expand_as(nconv)
        # Propagate confidence
        cout = F.max_pool2d(conf, self.kernel_size, self.stride, self.padding)
        mask = torch.sum(cout, dim=1) > 0
        cout = cout.permute(0, 2, 3, 1)
        cout[mask] = 1
        cout = cout.permute(0, 3, 1, 2)

        nconv = self.active_LeakyReLU(nconv)
        return nconv, cout


class Transp(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(Transp, self).__init__(in_channels, out_channels, kernel_size,
                                     stride, padding, dilation, False, (0, 0),
                                     groups, bias, padding_mode='zeros')

        self.main = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding=(1, 1))
        self.active = nn.LeakyReLU(0.01)
        self.bn1 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Transposed 2d layer
        x = self.main(x)
        x = self.bn1(x)
        x = self.active(x)
        return x


class ResidualBlock(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(ResidualBlock, self).__init__(in_channels, out_channels, kernel_size,
                                            stride, padding, dilation, False, (0, 0),
                                            groups, bias, padding_mode='zeros')

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, dilation=dilation)

        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.isDown = downsample
        self.downsample = nn.Sequential(
            conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active_ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.isDown is not None:
            identity = self.downsample(x)
        out += identity

        # out = self.active_ReLU(out)

        return out


class GRB(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, downsample, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(GRB, self).__init__(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, False, (0, 0),
                                  groups, bias, padding_mode='zeros')

        self.conv1 = GConv(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv2 = GConv(out_channels, out_channels, kernel_size, (1, 1), padding, dilation=dilation)

        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.isDown = downsample
        self.downsample = nn.Sequential(
            gconv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active_ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.isDown:
            identity = self.downsample(x)
        out += identity

        # out = self.active_ReLU(out)

        return out


class TRB(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, upsample, stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):
        # Call _ConvNd constructor
        super(TRB, self).__init__(in_channels, out_channels, kernel_size,
                                  stride, padding, dilation, False, (0, 0),
                                  groups, bias, padding_mode='zeros')

        self.conv1 = GTransp(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation)
        self.conv2 = GConv(out_channels, out_channels, kernel_size, (1, 1), padding, dilation=dilation)

        self.active_LeakyReLU = nn.LeakyReLU(0.01)
        self.active_ReLU = nn.ReLU()
        self.active_Sigmoid = nn.Sigmoid()
        self.active_Tanh = nn.Tanh()
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.isUp = upsample
        self.upsample = nn.Sequential(
            gtransp1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.active_ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.isUp:
            identity = self.upsample(x)
        out += identity

        # out = self.active_ReLU(out)

        return out
