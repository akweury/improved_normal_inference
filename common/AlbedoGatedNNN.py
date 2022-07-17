import torch
import torch.nn as nn
import torch.nn.functional as F

from common.Layers import GConv


class AlbedoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'albedoNet'
        self.sigmoid = nn.Sigmoid()
        self.albedo1 = GConv(6, 1, (3, 3), (1, 1), (1, 1))
        self.albedo2 = GConv(2, 1, (3, 3), (1, 1), (1, 1))
        self.albedo3 = nn.Conv2d(1, 1, (3, 3), (1, 1), (1, 1))

    def forward(self, x_normal_out, x_light, x_img):
        x_g = self.albedo1(torch.cat((x_normal_out, x_light), 1))
        x_albedo = self.albedo2(torch.cat((x_g, x_img), 1))
        x_albedo = self.albedo3(x_albedo)

        return x_albedo


#
class GNet2(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        # self.__name__ = 'gnet'
        kernel_down = (3, 3)
        kernel_up = (3, 3)
        padding_down = (1, 1)
        padding_up = (1, 1)
        stride = (1, 1)
        stride_2 = (2, 2)

        channel_size_1 = channel_num
        channel_size_2 = channel_num * 2
        channel_size_3 = channel_num * 3
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.
        self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.l_dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.i_dconv1 = GConv(1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.uconv3_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv4_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv4 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv5_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv5 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.l_uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.l_uconv4 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.l_uconv5 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.i_uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.i_uconv4 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.i_uconv5 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.v_conv1 = GConv(channel_size_1 * 3, channel_size_1, kernel_up, stride, padding_up)
        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, vertex, light, img):
        v1 = self.dconv1(vertex)
        v1 = self.dconv2(v1)
        v1 = self.dconv3(v1)

        l1 = self.l_dconv1(light)
        l1 = self.l_dconv2(l1)
        l1 = self.l_dconv3(l1)

        i1 = self.i_dconv1(img)
        i1 = self.i_dconv2(i1)
        i1 = self.i_dconv3(i1)

        # Downsample 1
        v2 = self.dconv4(v1)
        v2 = self.dconv2(v2)
        v2 = self.dconv3(v2)

        l2 = self.l_dconv4(l1)
        l2 = self.l_dconv2(l2)
        l2 = self.l_dconv3(l2)

        i2 = self.i_dconv4(i1)
        i2 = self.i_dconv2(i2)
        i2 = self.i_dconv3(i2)

        # Downsample 2
        v3 = self.dconv4(v2)
        v3 = self.dconv2(v3)
        v3 = self.dconv3(v3)

        l3 = self.l_dconv4(l2)
        l3 = self.l_dconv2(l3)
        l3 = self.l_dconv3(l3)

        i3 = self.i_dconv4(i2)
        i3 = self.i_dconv2(i3)
        i3 = self.i_dconv3(i3)

        # Downsample 3
        v4 = self.dconv4(v3)
        v4 = self.dconv2(v4)
        v4 = self.dconv3(v4)

        l4 = self.l_dconv4(l3)
        l4 = self.l_dconv2(l4)
        l4 = self.l_dconv3(l4)

        i4 = self.i_dconv4(i3)
        i4 = self.i_dconv2(i4)
        i4 = self.i_dconv3(i4)

        # Upsample 1
        x4_cat = torch.cat((v4, l4, i4), 1)
        x3_cat = F.interpolate(x4_cat, v3.size()[2:], mode='nearest')  # 128,128
        x3 = self.uconv3_1(x3_cat)
        v3 = self.uconv3(torch.cat((x3, v3), 1))

        l3_us = F.interpolate(l4, l3.size()[2:], mode='nearest')  # 128,128
        l3 = self.l_uconv4(torch.cat((l3, l3_us), 1))

        i3_us = F.interpolate(i4, i3.size()[2:], mode='nearest')  # 128,128
        i3 = self.i_uconv4(torch.cat((i3, i3_us), 1))

        # Upsample 2
        x3_cat = torch.cat((v3, l3, i3), 1)
        x2_cat = F.interpolate(x3_cat, v2.size()[2:], mode='nearest')  # 128,128
        x2 = self.uconv4_1(x2_cat)
        v2 = self.uconv4(torch.cat((x2, v2), 1))

        l2_us = F.interpolate(l3, l2.size()[2:], mode='nearest')  # 128,128
        l2 = self.l_uconv4(torch.cat((l2, l2_us), 1))

        i2_us = F.interpolate(i3, i2.size()[2:], mode='nearest')  # 128,128
        i2 = self.i_uconv4(torch.cat((i2, i2_us), 1))

        # Upsample 3
        x2_cat = torch.cat((v2, l2, i2), 1)
        x1_cat = F.interpolate(x2_cat, v1.size()[2:], mode='nearest')  # 128,128
        x1 = self.uconv5_1(x1_cat)
        v1 = self.uconv5(torch.cat((x1, v1), 1))

        l1_us = F.interpolate(l2, l1.size()[2:], mode='nearest')  # 128,128
        l1 = self.l_uconv5(torch.cat((l1, l1_us), 1))

        i1_us = F.interpolate(i2, i1.size()[2:], mode='nearest')  # 128,128
        i1 = self.i_uconv5(torch.cat((i1, i1_us), 1))

        x0_cat = torch.cat((v1, l1, i1), 1)
        x0 = self.v_conv1(x0_cat)  # 512, 512
        x0 = self.conv1(x0)
        xout = self.conv2(x0)

        return xout


class GNet(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        # self.__name__ = 'gnet'
        kernel_down = (3, 3)
        kernel_up = (3, 3)
        padding_down = (1, 1)
        padding_up = (1, 1)
        stride = (1, 1)
        stride_2 = (2, 2)

        channel_size_1 = channel_num
        channel_size_2 = channel_num * 2
        channel_size_3 = channel_num * 3
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.
        self.dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.l_dconv1 = GConv(in_ch, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.l_dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.i_dconv1 = GConv(1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv2 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv3 = GConv(channel_size_1, channel_size_1, kernel_down, stride, padding_down)
        self.i_dconv4 = GConv(channel_size_1, channel_size_1, kernel_down, stride_2, padding_down)

        self.uconv1_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3_1 = GConv(channel_size_3, channel_size_1, kernel_up, stride, padding_up)
        self.uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.l_uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.l_uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.l_uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.i_uconv1 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.i_uconv2 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)
        self.i_uconv3 = GConv(channel_size_2, channel_size_1, kernel_up, stride, padding_up)

        self.v_conv1 = GConv(channel_size_1 * 3, channel_size_1, kernel_up, stride, padding_up)
        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, vertex, light, img):
        v1 = self.dconv1(vertex)
        v1 = self.dconv2(v1)
        v1 = self.dconv3(v1)

        l1 = self.l_dconv1(light)
        l1 = self.l_dconv2(l1)
        l1 = self.l_dconv3(l1)

        i1 = self.i_dconv1(img)
        i1 = self.i_dconv2(i1)
        i1 = self.i_dconv3(i1)

        # Downsample 1
        v2 = self.dconv4(v1)
        v2 = self.dconv2(v2)
        v2 = self.dconv3(v2)

        l2 = self.l_dconv4(l1)
        l2 = self.l_dconv2(l2)
        l2 = self.l_dconv3(l2)

        i2 = self.i_dconv4(i1)
        i2 = self.i_dconv2(i2)
        i2 = self.i_dconv3(i2)

        # Downsample 2
        v3 = self.dconv4(v2)
        v3 = self.dconv2(v3)
        v3 = self.dconv3(v3)

        l3 = self.l_dconv4(l2)
        l3 = self.l_dconv2(l3)
        l3 = self.l_dconv3(l3)

        i3 = self.i_dconv4(i2)
        i3 = self.i_dconv2(i3)
        i3 = self.i_dconv3(i3)

        # Downsample 3
        v4 = self.dconv4(v3)
        v4 = self.dconv2(v4)
        v4 = self.dconv3(v4)

        l4 = self.l_dconv4(l3)
        l4 = self.l_dconv2(l4)
        l4 = self.l_dconv3(l4)

        i4 = self.i_dconv4(i3)
        i4 = self.i_dconv2(i4)
        i4 = self.i_dconv3(i4)

        # Upsample 1
        x4_cat = torch.cat((v4, l4, i4), 1)
        x3_cat = F.interpolate(x4_cat, v3.size()[2:], mode='nearest')  # 128,128
        x3 = self.uconv1_1(x3_cat)
        v3 = self.uconv1(torch.cat((x3, v3), 1))

        l3_us = F.interpolate(l4, l3.size()[2:], mode='nearest')  # 128,128
        l3 = self.l_uconv1(torch.cat((l3, l3_us), 1))

        i3_us = F.interpolate(i4, i3.size()[2:], mode='nearest')  # 128,128
        i3 = self.i_uconv1(torch.cat((i3, i3_us), 1))

        # Upsample 2
        x3_cat = torch.cat((v3, l3, i3), 1)
        x2_cat = F.interpolate(x3_cat, v2.size()[2:], mode='nearest')  # 128,128
        x2 = self.uconv2_1(x2_cat)
        v2 = self.uconv2(torch.cat((x2, v2), 1))

        l2_us = F.interpolate(l3, l2.size()[2:], mode='nearest')  # 128,128
        l2 = self.l_uconv2(torch.cat((l2, l2_us), 1))

        i2_us = F.interpolate(i3, i2.size()[2:], mode='nearest')  # 128,128
        i2 = self.i_uconv2(torch.cat((i2, i2_us), 1))

        # Upsample 3
        x2_cat = torch.cat((v2, l2, i2), 1)
        x1_cat = F.interpolate(x2_cat, v1.size()[2:], mode='nearest')  # 128,128
        x1 = self.uconv3_1(x1_cat)
        v1 = self.uconv3(torch.cat((x1, v1), 1))

        l1_us = F.interpolate(l2, l1.size()[2:], mode='nearest')  # 128,128
        l1 = self.l_uconv3(torch.cat((l1, l1_us), 1))

        i1_us = F.interpolate(i2, i1.size()[2:], mode='nearest')  # 128,128
        i1 = self.i_uconv3(torch.cat((i1, i1_us), 1))

        x0_cat = torch.cat((v1, l1, i1), 1)
        x0 = self.v_conv1(x0_cat)  # 512, 512
        x0 = self.conv1(x0)
        xout = self.conv2(x0)

        return xout
