import torch
import torch.nn as nn

from common.Layers import GConv


class AlbedoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'albedoNet'

        self.albedo1 = GConv(6, 1, (3, 3), (1, 1), (1, 1))
        self.albedo2 = GConv(2, 1, (3, 3), (1, 1), (1, 1))
        self.albedo3 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x_normal_out, x_light, x_img):
        x_g = self.albedo1(torch.cat((x_normal_out, x_light), 1))
        x_albedo = self.albedo2(torch.cat((x_g, x_img), 1))
        x_albedo = self.albedo3(x_albedo)

        return x_albedo

#
# class AlbedoGatedNNN(nn.Module):
#     def __init__(self, in_ch, out_ch, channel_num):
#         super().__init__()
#         self.__name__ = 'ag'
#         kernel_down = (3, 3)
#         kernel_up = (3, 3)
#         padding_down = (1, 1)
#         padding_up = (1, 1)
#         stride = (1, 1)
#         stride_2 = (2, 2)
#
#         dilate1 = (2, 2)
#         dilate2 = (4, 4)
#         dilate3 = (8, 8)
#         dilate4 = (16, 16)
#
#         channel_size_1 = channel_num
#         channel_size_2 = channel_num * 2
#         # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.
#         self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
#         self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
#         self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
#         self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)
#
#         # self.dilated1 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down, dilate1)
#         # self.dilated2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down, dilate2)
#         # self.dilated3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down, dilate3)
#         # self.dilated4 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down, dilate4)
#
#         self.uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
#         self.uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
#         self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
#
#         self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
#         self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))
#
#     def forward(self, xin):
#         x1 = self.dconv1(xin)
#         x1 = self.dconv2(x1)
#         x1 = self.dconv3(x1)
#
#         # Downsample 1
#         x2 = self.dconv4(x1)
#         x2 = self.dconv2(x2)
#         x2 = self.dconv3(x2)
#
#         # Downsample 2
#         x3 = self.dconv4(x2)
#         x3 = self.dconv2(x3)
#         x3 = self.dconv3(x3)
#
#         # Downsample 3
#         x4 = self.dconv4(x3)
#         x4 = self.dconv2(x4)
#         x4 = self.dconv3(x4)
#
#         # # dilated conv
#         # x4 = self.dilated1(x4)
#         # x4 = self.dilated2(x4)
#         # x4 = self.dilated3(x4)
#         # x4 = self.dilated4(x4)
#         # x4 = self.dconv2(x4)
#         # x4 = self.dconv3(x4)
#
#         # Upsample 1
#         x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
#         x3 = self.uconv1(torch.cat((x3, x3_us), 1))
#
#         # Upsample 2
#         x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
#         x2 = self.uconv2(torch.cat((x2, x2_us), 1))
#
#         # # Upsample 3
#         x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
#         x1 = self.uconv3(torch.cat((x1, x1_us), 1))
#
#         xout = self.conv1(x1)  # 512, 512
#         # xout_normal = self.conv2(xout)
#
#         return xout
