import torch.nn as nn

from common.Layers import GRB, TRB, ResidualBlock, Conv


class FGCNN(nn.Module):
    def __init__(self, in_ch, out_ch, channel_num):
        super().__init__()
        self.__name__ = 'fgcnn'
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

        # self.epsilon = 1e-20
        channel_size_1 = channel_num
        channel_size_2 = channel_num * 2
        channel_size_3 = channel_num * 2
        channel_size_4 = channel_num * 2
        # https://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/PIRODDI1/NormConv/node2.html#:~:text=The%20idea%20of%20normalized%20convolution,them%20is%20equal%20to%20zero.

        self.init1 = GRB(in_ch, channel_size_1, kernel_down, True, stride, padding_down)
        self.init2 = GRB(channel_size_1, channel_size_1, kernel_down, False, stride, padding_down)
        self.init3 = GRB(channel_size_1, channel_size_1, kernel_down, False, stride, padding_down)

        self.d1l1 = GRB(channel_size_1, channel_size_2, kernel_down, True, stride_2, padding_down)
        self.d1l2 = GRB(channel_size_2, channel_size_2, kernel_down, False, stride, padding_down)
        self.d1l3 = GRB(channel_size_2, channel_size_2, kernel_down, False, stride, padding_down)
        self.d1l4 = GRB(channel_size_2, channel_size_2, kernel_down, False, stride, padding_down)

        self.d2l1 = GRB(channel_size_2, channel_size_3, kernel_down, True, stride_2, padding_down)
        self.d2l2 = GRB(channel_size_3, channel_size_3, kernel_down, False, stride, padding_down)
        self.d2l3 = GRB(channel_size_3, channel_size_3, kernel_down, False, stride, padding_down)
        self.d2l4 = GRB(channel_size_3, channel_size_3, kernel_down, False, stride, padding_down)

        self.d3l1 = GRB(channel_size_3, channel_size_4, kernel_down, True, stride_2, padding_down)
        self.d3l2 = GRB(channel_size_4, channel_size_4, kernel_down, False, stride, padding_down)
        self.d3l3 = GRB(channel_size_4, channel_size_4, kernel_down, False, stride, padding_down)
        self.d3l4 = GRB(channel_size_4, channel_size_4, kernel_down, False, stride, padding_down)

        self.u1l1 = TRB(channel_size_4, channel_size_3, kernel_up, True, stride_2, padding_up)
        self.u1l2 = GRB(channel_size_3, channel_size_3, kernel_up, False, stride, padding_up)
        self.u1l3 = GRB(channel_size_3, channel_size_3, kernel_up, False, stride, padding_up)

        self.u2l1 = TRB(channel_size_3, channel_size_2, kernel_up, True, stride_2, padding_up)
        self.u2l2 = GRB(channel_size_2, channel_size_2, kernel_up, False, stride, padding_up)
        self.u2l3 = GRB(channel_size_2, channel_size_2, kernel_up, False, stride, padding_up)

        self.u3l1 = TRB(channel_size_2, channel_size_1, kernel_up, True, stride_2, padding_up)
        self.u3l2 = GRB(channel_size_1, channel_size_1, kernel_up, False, stride, padding_up)
        self.u3l3 = GRB(channel_size_1, channel_size_1, kernel_up, False, stride, padding_up)

        self.out1 = ResidualBlock(channel_size_1, out_ch, (1, 1), False, (1, 1), (0, 0))
        self.out2 = Conv(out_ch, out_ch, (1, 1), (1, 1), (0, 0), active_function="")

    def forward(self, xin):
        x1 = self.init1(xin)
        x1 = self.init2(x1)
        x1 = self.init3(x1)

        # Downsample 1
        x2 = self.d1l1(x1)
        x2 = self.d1l2(x2)
        x2 = self.d1l3(x2)
        x2 = self.d1l4(x2)

        # Downsample 2
        x3 = self.d2l1(x2)
        x3 = self.d2l2(x3)
        x3 = self.d2l3(x3)
        x3 = self.d2l4(x3)

        # Downsample 3
        x4 = self.d3l1(x3)
        x4 = self.d3l2(x4)
        x4 = self.d3l3(x4)
        x4 = self.d3l4(x4)

        # Upsample 1
        x3 = self.u1l1(x4)
        x3 = self.u1l2(x3)
        x3 = self.u1l3(x3)

        # Upsample 2
        x2 = self.u2l1(x3)
        x2 = self.u2l2(x2)
        x2 = self.u2l3(x2)

        # # Upsample 3
        x1 = self.u3l1(x2)
        x1 = self.u3l2(x1)
        x1 = self.u3l3(x1)

        xout = self.out1(x1)  # 512, 512
        xout = self.out2(xout)

        return xout
