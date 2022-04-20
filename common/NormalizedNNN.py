import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.active = nn.LeakyReLU(0.01)
        self.active_last = nn.Tanh()
        # self.active = nn.ReLU()

        self.epsilon = 1e-20
        channel_size_1 = 6
        channel_size_2 = 12
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        # self.dconv = nn.Conv2d(in_ch, in_ch, kernel_down, (1, 1), padding_down)
        self.cconv1 = nn.Conv2d(in_ch, in_ch, kernel_down, (1, 1), padding_down)
        self.cconv2 = nn.Conv2d(in_ch, in_ch, kernel_down, (1, 1), padding_down)
        self.cconv3 = nn.Conv2d(in_ch, in_ch, kernel_down, (1, 1), padding_down)

        self.dconv1 = nn.Conv2d(in_ch, channel_size_1, kernel_down, (1, 1), padding_down)

        self.dconv2 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)

        self.dconv3 = nn.Conv2d(channel_size_1, channel_size_1, kernel_down, (1, 1), padding_down)

        self.uconv1 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.uconv2 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

        self.uconv3 = nn.Conv2d(channel_size_2, channel_size_1, kernel_up, (1, 1), padding_up)

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
        # x0 = self.cconv1(x0)
        # c0 = self.cconv1(c0)
        # x0 = x0 / (c0 + self.epsilon)
        # c0 = self.c_avg(c0, self.cconv1.weight)

        x1 = self.dconv1(x0)  # 512,512
        c1 = self.dconv1(c0)
        x1 = x1 / (c1 + self.epsilon)
        c1 = self.c_avg(c1, self.dconv1.weight)

        x1 = self.dconv2(x1)  # 512,512
        c1 = self.dconv2(c1)
        x1 = x1 / (c1 + self.epsilon)
        c1 = self.c_avg(c1, self.dconv2.weight)

        x1 = self.dconv3(x1)  # 512,512
        c1 = self.dconv3(c1)
        x1 = x1 / (c1 + self.epsilon)
        c1 = self.c_avg(c1, self.dconv3.weight)

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)  # 256,256

        x1_ds = torch.zeros(c1_ds.size()).to(x0.get_device())
        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c1_ds /= 4

        x2 = self.dconv2(x1_ds)  # 256,256
        c2 = self.dconv2(c1_ds)
        x2 = x2 / (c2 + self.epsilon)
        c2 = self.c_avg(c2, self.dconv2.weight)

        x2 = self.active(self.dconv3(x2))  # 256,256
        c2 = self.dconv2(c2)
        x2 = x2 / (c2 + self.epsilon)
        c2 = self.c_avg(c2, self.dconv3.weight)

        # Downsample 2
        ds = 2
        c2_ds, idx = F.max_pool2d(c2, ds, ds, return_indices=True)  # 128,128
        x2_ds = torch.zeros(c2_ds.size()).to(x0.get_device())
        for i in range(x2_ds.size(0)):
            for j in range(x2_ds.size(1)):
                x2_ds[i, j, :, :] = x2[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_ds /= 4

        x3 = self.dconv2(x2_ds)  # 128,128
        c3 = self.dconv2(c2_ds)
        x3 = x3 / (c3 + self.epsilon)
        c3 = self.c_avg(c3, self.dconv2.weight)

        x3 = self.dconv3(x3)  # 128,128
        c3 = self.dconv3(c3)
        x3 = x3 / (c3 + self.epsilon)
        c3 = self.c_avg(c3, self.dconv3.weight)

        # Downsample 3
        ds = 2
        c3_ds, idx = F.max_pool2d(c3, ds, ds, return_indices=True)  # 64,64
        x3_ds = torch.zeros(c3_ds.size()).to(x0.get_device())
        for i in range(x3_ds.size(0)):
            for j in range(x3_ds.size(1)):
                x3_ds[i, j, :, :] = x3[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_ds /= 4

        x4 = self.dconv2(x3_ds)  # 64,64
        c4 = self.dconv2(c3_ds)
        x4 = x4 / (c4 + self.epsilon)
        c4 = self.c_avg(c4, self.dconv2.weight)

        x4 = self.dconv3(x4)  # 64,64
        c4 = self.dconv3(c4)
        x4 = x4 / (c4 + self.epsilon)
        c4 = self.c_avg(c4, self.dconv3.weight)

        # Upsample 1
        x4_us = F.interpolate(x4, x3.size()[2:], mode='nearest')  # 128,128
        c4_us = F.interpolate(c4, c3.size()[2:], mode='nearest')  # 128,128

        x5 = self.uconv1(torch.cat((x3, x4_us), 1))  # 128, 128
        c5 = self.uconv1(torch.cat((c3, c4_us), 1))
        x5 = x5 / (c5 + self.epsilon)
        c5 = self.c_avg(c5, self.uconv1.weight)

        # Upsample 2
        x5_us = F.interpolate(x5, x2.size()[2:], mode='nearest')
        c5_us = F.interpolate(c5, c2.size()[2:], mode='nearest')

        x6 = self.uconv2(torch.cat((x2, x5_us), 1))  # 256, 256
        c6 = self.uconv2(torch.cat((c2, c5_us), 1))
        x6 = x6 / (c6 + self.epsilon)
        c6 = self.c_avg(c6, self.uconv2.weight)

        # # Upsample 3
        x6_us = F.interpolate(x6, x1.size()[2:], mode='nearest')  # 512, 512
        c6_us = F.interpolate(c6, c1.size()[2:], mode='nearest')  # 512, 512

        x7 = self.uconv3(torch.cat((x1, x6_us), 1))
        c7 = self.uconv3(torch.cat((c1, c6_us), 1))
        x7 = x7 / (c7 + self.epsilon)
        c7 = self.c_avg(c7, self.uconv3.weight)

        # xout = self.conv1(x7)  # 512, 512
        # xout = self.conv2(xout)
        return x7, c7, x7
