import torch.nn as nn
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
        self.active_name = active_function

    def forward(self, x):
        # Normalized Convolution
        if self.active_name == "LeakyReLU":
            return self.active_LeakyReLU(self.conv(x))
        elif self.active_name == "Sigmoid":
            return self.active_Sigmoid(self.conv(x))
        elif self.active_name == "ReLU":
            return self.active_ReLU(self.conv(x))
        else:
            raise ValueError
