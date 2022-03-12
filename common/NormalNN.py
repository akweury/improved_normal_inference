import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.conv import _ConvNd
import numpy as np
from scipy.stats import poisson
from scipy import signal
import math

"""
Normal Neuron Network
input: vertex (3,512,512)
output: normal  (3,512,512)

"""


class NormalNN(nn.Module):
    def __init__(self, in_ch, out_ch, num_channels=3):
        super().__init__()
        self.__name__ = 'NormalNN'
        kernel_5 = (5, 5)
        kernel_3 = (3, 3)
        padding_2 = (2, 2)
        padding_1 = (1, 1)

        self.dconv1 = nn.Conv2d(in_ch, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv2 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv3 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv4 = nn.Conv2d(in_ch * num_channels, 2 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv5 = nn.Conv2d(2 * in_ch * num_channels, 4 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv6 = nn.Conv2d(4 * in_ch * num_channels, 8 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.dconv7 = nn.Conv2d(8 * in_ch * num_channels, 8 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv1 = nn.Conv2d(16 * in_ch * num_channels, 4 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv2 = nn.Conv2d(8 * in_ch * num_channels, 1 * in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv3 = nn.Conv2d(2 * in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv4 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.uconv5 = nn.Conv2d(in_ch * num_channels, in_ch * num_channels, kernel_3, (1, 1), padding_1)

        self.conv1 = nn.Conv2d(in_ch * num_channels, out_ch, (1, 1), (1, 1), (0, 0))

    def forward(self, x0):
        x1 = self.dconv1(x0)  # (1,18,512,512)
        x1 = self.dconv2(x1)  # (1,18,512,512)
        x1 = self.dconv3(x1)  # (1,18,512,512)

        # Downsample 1
        ds = 2
        x1_ds, idx = F.max_pool2d(x1, ds, ds, return_indices=True)  # (1,18,256,256)
        x1_ds /= 4

        x2_ds = self.dconv4(x1_ds)  # (1,18,256,256)
        x2_ds = self.dconv5(x2_ds)  # (1,18,256,256)

        # Downsample 2
        ds = 2
        x2_dss, idx = F.max_pool2d(x2_ds, ds, ds, return_indices=True)  # (1,18,131,131)
        x2_dss /= 4

        x3_ds = self.dconv6(x2_dss)  # (1,18,128,128)

        # Downsample 3
        ds = 2
        x3_dss, idx = F.max_pool2d(x3_ds, ds, ds, return_indices=True)  # (1,18,64,64)
        x3_dss /= 4

        x4_ds = self.dconv7(x3_dss)  # (1,18,64,64)

        # Upsample 1
        x4 = F.interpolate(x4_ds, x3_ds.size()[2:], mode='nearest')  # (1,18,128,128)

        x34_ds = self.uconv1(torch.cat((x3_ds, x4), 1))  # (1, 9, 128, 128)

        # Upsample 2
        x34 = F.interpolate(x34_ds, x2_ds.size()[2:], mode='nearest')
        x23_ds = self.uconv2(torch.cat((x2_ds, x34), 1))  # (1, 9, 256, 256)

        # # Upsample 3
        x23 = F.interpolate(x23_ds, x0.size()[2:], mode='nearest')  # (1, 9, 512, 512)
        xout = self.uconv3(torch.cat((x23, x1), 1))  # (1, 9, 512, 512)

        xout = self.conv1(xout)  # (1, 3, 512, 512)
        return xout


# Normalized Convolution Layer
class NConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, pos_fn='softplus', init_method='n', stride=(1, 1),
                 padding=(0, 0), dilation=(1, 1), groups=1, bias=True):

        # Call _ConvNd constructor
        super(NConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, False, (0, 0),
                                      groups, bias, padding_mode='zeros')

        self.eps = 1e-20
        self.pos_fn = pos_fn
        self.init_method = init_method

        # Initialize weights and bias
        self.init_parameters()

        if self.pos_fn is not None:
            EnforcePos.apply(self, 'weight', pos_fn)

    def forward(self, data, conf):
        # Normalized Convolution
        denom = F.conv2d(conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nomin = F.conv2d(data * conf, self.weight, None, self.stride,
                         self.padding, self.dilation, self.groups)
        nconv = nomin / (denom + self.eps)

        # Add bias
        b = self.bias
        sz = b.size(0)
        b = b.view(1, sz, 1, 1)
        b = b.expand_as(nconv)
        nconv += b

        # Propagate confidence
        cout = denom
        # cout = denom.sum(dim=1, keepdim=True)
        sz = cout.size()
        cout = cout.view(sz[0], sz[1], -1)

        k = self.weight
        k_sz = k.size()
        k = k.view(k_sz[0], -1)

        s = torch.sum(k, dim=-1, keepdim=True)
        # s = torch.sum(k)

        cout = cout / s
        cout = cout.view(sz)

        return nconv, cout

    def enforce_pos(self):
        p = self.weight
        if self.pos_fn.lower() == 'softmax':
            p_sz = p.size()
            p = p.view(p_sz[0], p_sz[1], -1)
            p = F.softmax(p, -1).data
            self.weight.data = p.view(p_sz)
        elif self.pos_fn.lower() == 'exp':
            self.weight.data = torch.exp(p).data
        elif self.pos_fn.lower() == 'softplus':
            self.weight.data = F.softplus(p, beta=10).data
        elif self.pos_fn.lower() == 'sigmoid':
            self.weight.data = F.sigmoid(p).data
        else:
            print('Undefined positive function!')
            return

    def init_parameters(self):
        # Init weights
        if self.init_method == 'x':  # Xavier
            torch.nn.init.xavier_uniform_(self.weight)
        elif self.init_method == 'k':  # Kaiming
            torch.nn.init.kaiming_uniform_(self.weight)
        elif self.init_method == 'n':  # Normal dist
            n = self.kernel_size[0] * self.kernel_size[1] * self.out_channels
            self.weight.data.normal_(2, math.sqrt(2. / n))
        elif self.init_method == 'p':  # Poisson
            mu = self.kernel_size[0] / 2
            dist = poisson(mu)
            x = np.arange(0, self.kernel_size[0])
            y = np.expand_dims(dist.pmf(x), 1)
            w = signal.convolve2d(y, y.transpose(), 'full')
            w = torch.Tensor(w).type_as(self.weight)
            w = torch.unsqueeze(w, 0)
            w = torch.unsqueeze(w, 1)
            w = w.repeat(self.out_channels, 1, 1, 1)
            w = w.repeat(1, self.in_channels, 1, 1)
            self.weight.data = w + torch.rand(w.shape)

        # Init bias
        self.bias = torch.nn.Parameter(torch.zeros(self.out_channels) + 0.01)


class EnforcePos(object):
    def __init__(self, name, pos_fn):
        self.name = name
        self.pos_fn = pos_fn

    def compute_weight(self, module):
        return _pos(getattr(module, self.name + '_p'), self.pos_fn)

    @staticmethod
    def apply(module, name, pos_fn):
        fn = EnforcePos(name, pos_fn)

        weight = getattr(module, name)

        # remove w from parameter list
        del module._parameters[name]

        #
        module.register_parameter(name + '_p', Parameter(_pos(weight, pos_fn).data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module):
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + '_p']
        module.register_parameter(self.name, Parameter(weight.data))

    def __call__(self, module, inputs):
        setattr(module, self.name, self.compute_weight(module))


def _pos(p, pos_fn):
    pos_fn = pos_fn.lower()
    if pos_fn == 'softmax':
        p_sz = p.size()
        p = p.view(p_sz[0], p_sz[1], -1)
        p = F.softmax(p, -1)
        return p.view(p_sz)
    elif pos_fn == 'exp':
        return torch.exp(p)
    elif pos_fn == 'softplus':
        return F.softplus(p, beta=10)
    elif pos_fn == 'sigmoid':
        return F.sigmoid(p)
    else:
        print('Undefined positive function!')
        return


def remove_weight_pos(module, name='weight'):
    r"""Removes the weight normalization reparameterization from a module.

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, EnforcePos) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError("weight_norm of '{}' not found in {}"
                     .format(name, module))
