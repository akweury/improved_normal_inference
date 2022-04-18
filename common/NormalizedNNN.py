import torch
import torch.nn as nn
import torch.nn.functional as F
from common.nconv import NConv2d


class NormalizedNNN(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.__name__ = 'NormalizedNNN'
        kernel_down = (3, 3)
        kernel_up = (3, 3)
        padding_down = (1, 1)
        padding_up = (1, 1)
        self.active = nn.LeakyReLU(0.01)
        self.active_last = nn.Tanh()
        # self.active = nn.ReLU()
        pos_fn = "SoftPlus"
        channel_size_1 = 32
        channel_size_2 = 64

        self.dnconv1 = NConv2d(in_ch, channel_size_1, kernel_down, pos_fn, 'k', padding=padding_down)
        self.dnconv2 = NConv2d(channel_size_1, channel_size_1, kernel_down, pos_fn, 'k', padding=padding_down)
        self.dnconv3 = NConv2d(channel_size_1, channel_size_1, kernel_down, pos_fn, 'k', padding=padding_down)

        self.unconv1 = NConv2d(channel_size_2, channel_size_1, kernel_up, pos_fn, 'k', padding=padding_up)
        self.unconv2 = NConv2d(channel_size_2, channel_size_1, kernel_up, pos_fn, 'k', padding=padding_up)
        self.unconv3 = NConv2d(channel_size_2, channel_size_1, kernel_up, pos_fn, 'k', padding=padding_up)

        self.conv1 = nn.Conv2d(channel_size_1, out_ch, (1, 1), (1, 1), (0, 0))
        self.conv2 = nn.Conv2d(out_ch, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x0, c0):
        x1, c1 = self.dnconv1(x0, c0)  # 512,512
        x1, c1 = self.dnconv2(x1, c1)  # 512,512
        x1, c1 = self.dnconv3(x1, c1)  # 512,512

        # Downsample 1
        ds = 2
        c1_ds, idx = F.max_pool2d(c1, ds, ds, return_indices=True)  # 256,256
        x1_ds = torch.zeros(c1_ds.size()).to(x0.get_device())

        for i in range(x1_ds.size(0)):
            for j in range(x1_ds.size(1)):
                x1_ds[i, j, :, :] = x1[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c1_ds /= 4

        x1_ds, c1_ds = self.dnconv2(x1_ds, c1_ds)  # 256,256
        x1_ds, c1_ds = self.dnconv3(x1_ds, c1_ds)  # 256,256

        # Downsample 2
        ds = 2
        c2_ds, idx = F.max_pool2d(c1_ds, ds, ds, return_indices=True)  # 128,128
        x2_ds = torch.zeros(c2_ds.size()).to(x0.get_device())

        for i in range(x2_ds.size(0)):
            for j in range(x2_ds.size(1)):
                x2_ds[i, j, :, :] = x1_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c2_ds /= 4

        x2_ds, c2_ds = self.dnconv2(x2_ds, c2_ds)  # 128,128
        x2_ds, c2_ds = self.dnconv3(x2_ds, c2_ds)  # 128,128

        # Downsample 3
        ds = 2
        c3_ds, idx = F.max_pool2d(c2_ds, ds, ds, return_indices=True)  # 64,64

        x3_ds = torch.zeros(c3_ds.size()).to(x0.get_device())

        for i in range(x3_ds.size(0)):
            for j in range(x3_ds.size(1)):
                x3_ds[i, j, :, :] = x2_ds[i, j, :, :].view(-1)[idx[i, j, :, :].view(-1)].view(idx.size()[2:])
        c3_ds /= 4

        x3_ds, c3_ds = self.dnconv2(x3_ds, c3_ds)  # 64,64
        x3_ds, c3_ds = self.dnconv3(x3_ds, c3_ds)  # 64,64

        # Upsample 1
        x4 = F.interpolate(x3_ds, x3_ds.size()[2:], mode='nearest')  # 128,128
        c4 = F.interpolate(c3_ds, c3_ds.size()[2:], mode='nearest')
        x34_ds, c34_ds = self.unconv1(torch.cat((x3_ds, x4), 1), torch.cat((c3_ds, c4), 1))  # 128, 128

        # Upsample 2
        x34 = F.interpolate(x34_ds, x2_ds.size()[2:], mode='nearest')
        c34 = F.interpolate(c34_ds, c2_ds.size()[2:], mode='nearest')
        x23_ds, c23_ds = self.unconv2(torch.cat((x2_ds, x34), 1), torch.cat((c2_ds, c34), 1))  # 256, 256

        # # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')  # 512, 512
        c23 = F.interpolate(c23_ds, c0.size()[2:], mode='nearest')
        xout, cout = self.unconv3(torch.cat((x1, x23), 1), torch.cat((c23, c1), 1))  # 512, 512

        xout = self.conv1(xout)  # 512, 512
        xout = self.conv2(xout)
        return xout, cout
