import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Normal Neuron Network
input: vertex (3,512,512)
output: normal  (3,512,512)

"""


class NormalNN24(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.__name__ = 'NormalNN24_4'
        kernel_down = (3, 3)
        kernel_down_2 = (5, 5)
        kernel_up = (3, 3)
        kernel_up_2 = (5, 5)
        padding_down = (1, 1)
        padding_down_2 = (2, 2)
        padding_up = (1, 1)
        padding_up_2 = (2, 2)

        self.active = nn.LeakyReLU(0.01)
        # self.active = nn.Tanh()
        # self.active = nn.ReLU()

        channel_size_1 = 32
        channel_size_2 = 64

        self.dconv1 = nn.Conv2d(in_ch, channel_size_1, kernel_down, (1, 1), padding_down)

        self.dconv2 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)

        self.dconv3 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)

        # self.dconv4 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)
        #
        # self.dconv5 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)
        #
        # self.dconv6 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)
        #
        # self.dconv7 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)

        self.uconv1 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.uconv2 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.uconv3 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.uconv4 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x0):
        x1 = self.active(self.dconv1(x0))  # 512,512
        x1 = self.active(self.dconv2(x1))  # 512,512
        x1 = self.active(self.dconv3(x1))  # 512,512

        # Downsample 1
        ds = 2
        x1_ds, idx = F.max_pool2d(x1, ds, ds, return_indices=True)  # 256,256
        x1_ds /= 4

        x2_ds = self.active(self.dconv2(x1_ds))  # 256,256
        x2_ds = self.active(self.dconv3(x2_ds))  # 256,256

        # Downsample 2
        ds = 2
        x2_ds, idx = F.max_pool2d(x2_ds, ds, ds, return_indices=True)  # 128,128
        x2_ds /= 4

        x3_ds = self.active(self.dconv2(x2_ds))  # 128,128
        x3_ds = self.active(self.dconv3(x3_ds))  # 128,128

        # Downsample 3
        ds = 2
        x3_ds, idx = F.max_pool2d(x3_ds, ds, ds, return_indices=True)  # 64,64
        x3_ds /= 4

        x4_ds = self.active(self.dconv2(x3_ds))  # 64,64
        x4_ds = self.active(self.dconv3(x4_ds))  # 64,64

        # Downsample 4
        ds = 2
        x4_ds, idx = F.max_pool2d(x4_ds, ds, ds, return_indices=True)  # 32,32
        x4_ds /= 4

        x5_ds = self.active(self.dconv2(x4_ds))  # 64,64
        x5_ds = self.active(self.dconv3(x5_ds))  # 64,64

        # Upsample 1
        x5 = F.interpolate(x5_ds, x4_ds.size()[2:], mode='nearest')  # 128,128

        x45_ds = self.active(self.uconv1(torch.cat((x4_ds, x5), 1)))  # 128, 128

        # Upsample 2
        x4 = F.interpolate(x45_ds, x3_ds.size()[2:], mode='nearest')  # 128,128

        x34_ds = self.active(self.uconv2(torch.cat((x3_ds, x4), 1)))  # 128, 128

        # Upsample 3
        x34 = F.interpolate(x34_ds, x2_ds.size()[2:], mode='nearest')
        x23_ds = self.active(self.uconv3(torch.cat((x2_ds, x34), 1)))  # 256, 256

        # # Upsample 4
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')  # 512, 512
        xout = self.active(self.uconv4(torch.cat((x23, x1), 1)))  # 512, 512

        # xout = self.active(self.conv1(xout))
        xout = self.conv1(xout)  # 512, 512
        xout = self.conv2(xout)
        return xout
