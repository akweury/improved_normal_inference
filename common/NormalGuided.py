import torch
import torch.nn as nn
import torch.nn.functional as F

from common.Layers import GConv


class NormalGuided(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        self.__name__ = 'ng'
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
        self.dconv1 = GConv(in_ch, channel_num, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_num, channel_num, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_num, channel_num, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_num, channel_num, kernel_down, stride_2, padding_down)

        self.uconv1 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv4 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv5 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv6 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv7 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv8 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv9 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.uconv10 = GConv(channel_size_2, channel_num, kernel_up, stride, padding_up)

        self.conv1 = nn.Conv2d(channel_size_2, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

        # branch 2
        self.img_dconv1 = nn.Conv2d(1, channel_num, kernel_down, stride, padding_down)
        self.img_dconv2 = nn.Conv2d(channel_num, channel_num, kernel_down, stride, padding_down)
        self.img_dconv3 = nn.Conv2d(channel_num, channel_num, kernel_down, stride, padding_down)
        self.img_dconv4 = nn.Conv2d(channel_num, channel_num, kernel_down, stride_2, padding_down)

        self.img_uconv1 = nn.Conv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.img_uconv2 = nn.Conv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.img_uconv3 = nn.Conv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.img_uconv4 = nn.Conv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)
        self.img_uconv5 = nn.Conv2d(channel_size_2, channel_num, kernel_up, stride, padding_up)

    def forward(self, x1, x_img_1):
        x1 = self.dconv1(x1)
        x1 = self.dconv2(x1)
        x1 = self.dconv3(x1)

        x_img_1 = self.active_img(self.img_dconv1(x_img_1))
        x_img_1 = self.active_img(self.img_dconv2(x_img_1))
        x_img_1 = self.active_img(self.img_dconv3(x_img_1))

        # Downsample 1
        x2 = self.dconv4(x1)
        x2 = self.dconv2(x2)
        x2 = self.dconv3(x2)

        x_img_2 = self.active_img(self.img_dconv4(x_img_1))
        x_img_2 = self.active_img(self.img_dconv2(x_img_2))
        x_img_2 = self.active_img(self.img_dconv3(x_img_2))

        # Downsample 2
        x3 = self.dconv4(x2)
        x3 = self.dconv2(x3)
        x3 = self.dconv3(x3)

        x_img_3 = self.active_img(self.img_dconv4(x_img_2))
        x_img_3 = self.active_img(self.img_dconv2(x_img_3))
        x_img_3 = self.active_img(self.img_dconv3(x_img_3))

        # Downsample 3
        x4 = self.dconv4(x3)
        x4 = self.dconv2(x4)
        x4 = self.dconv3(x4)

        x_img_4 = self.active_img(self.img_dconv4(x_img_3))
        x_img_4 = self.active_img(self.img_dconv2(x_img_4))
        x_img_4 = self.active_img(self.img_dconv3(x_img_4))

        # Downsample 4
        x5 = self.dconv4(x4)
        x5 = self.dconv2(x5)
        x5 = self.dconv3(x5)

        x_img_5 = self.active_img(self.img_dconv4(x_img_4))
        x_img_5 = self.active_img(self.img_dconv2(x_img_5))
        x_img_5 = self.active_img(self.img_dconv3(x_img_5))

        # Downsample 5
        x6 = self.dconv4(x5)
        x6 = self.dconv2(x6)
        x6 = self.dconv3(x6)

        x_img_6 = self.active_img(self.img_dconv4(x_img_5))
        x_img_6 = self.active_img(self.img_dconv2(x_img_6))
        x_img_6 = self.active_img(self.img_dconv3(x_img_6))

        # merge image feature and vertex feature
        x6 = torch.cat((x6, x_img_6), 1)

        # Upsample 1
        x5_us = F.interpolate(x6, x5.size()[2:], mode='nearest')  # 128,128
        x5_mus = self.uconv1(x5_us)
        x5 = self.uconv2(torch.cat((x5, x5_mus), 1))

        x5_img_us = F.interpolate(x_img_6, x_img_5.size()[2:], mode='nearest')  # 128,128
        x_img_5 = self.img_uconv1(torch.cat((x_img_5, x5_img_us), 1))

        # merge image feature and vertex feature
        x5 = torch.cat((x5, x_img_5), 1)

        # Upsample 2
        x4_us = F.interpolate(x5, x4.size()[2:], mode='nearest')
        x4_mus = self.uconv3(x4_us)
        x4 = self.uconv4(torch.cat((x4, x4_mus), 1))

        x4_img_us = F.interpolate(x_img_5, x_img_4.size()[2:], mode='nearest')
        x_img_4 = self.img_uconv2(torch.cat((x_img_4, x4_img_us), 1))

        # merge image feature and vertex feature
        x4 = torch.cat((x4, x_img_4), 1)

        # Upsample 3
        x3_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        x3_mus = self.uconv5(x3_us)
        x3 = self.uconv6(torch.cat((x3, x3_mus), 1))

        x3_img_us = F.interpolate(x_img_4, x_img_3.size()[2:], mode='nearest')  # 128,128
        x_img_3 = self.img_uconv3(torch.cat((x_img_3, x3_img_us), 1))

        # merge image feature and vertex feature 2
        x3 = torch.cat((x3, x_img_3), 1)

        # Upsample 4
        x2_us = F.interpolate(x3, x2.size()[2:], mode='nearest')
        x2_mus = self.uconv7(x2_us)
        x2 = self.uconv8(torch.cat((x2, x2_mus), 1))

        x2_img_us = F.interpolate(x_img_3, x_img_2.size()[2:], mode='nearest')  # 128,128
        x_img_2 = self.img_uconv4(torch.cat((x_img_2, x2_img_us), 1))

        # merge image feature and vertex feature 3
        x2 = torch.cat((x2, x_img_2), 1)

        # # Upsample 5
        x1_us = F.interpolate(x2, x1.size()[2:], mode='nearest')  # 512, 512
        x1_mus = self.uconv9(x1_us)
        x1 = self.uconv10(torch.cat((x1, x1_mus), 1))

        x1_img_us = F.interpolate(x_img_2, x_img_1.size()[2:], mode='nearest')  # 128,128
        x_img_1 = self.img_uconv5(torch.cat((x_img_1, x1_img_us), 1))

        # merge image feature and vertex feature 3
        x1 = torch.cat((x1, x_img_1), 1)

        xout = self.conv1(x1)  # 512, 512
        xout = self.conv2(xout)

        return xout
