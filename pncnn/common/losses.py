########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"

########################################

import torch
import torch.nn as nn
import torch.nn.functional as F

xout_channel = 3
cout_in_channel = 3
cout_out_channel = 6
cin_channel = 6


def get_loss_list():
    return loss_list.keys()


def get_loss_fn(args):
    loss = []
    for each in args.loss.split(","):
        loss.append(loss_list[each])
    return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        return F.l1_loss(outputs, target)


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        return F.mse_loss(outputs, target)


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        return F.smooth_l1_loss(outputs, target)


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        val_pixels = torch.ne(target, 0).float().detach()
        return F.l1_loss(outputs * val_pixels, target * val_pixels)


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        val_pixels = torch.ne(target, 0).float().detach()
        return F.mse_loss(outputs * val_pixels, target * val_pixels)


class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :xout_channel, :, :]
        val_pixels = torch.ne(target, 0).float().detach()
        loss = F.smooth_l1_loss(outputs * val_pixels, target * val_pixels, reduction='none')
        return torch.mean(loss)


# The proposed probabilistic loss for pNCNN
class MaskedProbLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        # means = out[:, :1, :, :]
        means = out[:, :xout_channel, :, :]

        # means = torch.permute(means, (0, 2, 3, 1)).unsqueeze(1)

        cout = out[:, cout_in_channel:cout_out_channel, :, :]
        # cout = torch.permute(cout, (0, 2, 3, 1)).unsqueeze(1)

        res = cout
        regl = torch.log(cout + 1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        # valid_mask = (targets.sum(dim=-1) > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        # loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss


class MaskedProbExpLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, out, targets):
        means = out[:, :xout_channel, :, :]
        cout = out[:, cout_in_channel:cout_out_channel, :, :]

        res = torch.exp(cout)  # Residual term
        regl = torch.log(cout + 1e-16)  # Regularization term

        # Pick only valid pixels
        valid_mask = (targets > 0).detach()
        targets = targets[valid_mask]
        means = means[valid_mask]
        res = res[valid_mask]
        regl = regl[valid_mask]

        loss = torch.mean(res * torch.pow(targets - means, 2) - regl)
        return loss


loss_list = {
    'l1': L1Loss(),
    'l2': L2Loss(),
    'masked_l1': MaskedL1Loss(),
    'masked_l2': MaskedL2Loss(),
    'masked_prob_loss': MaskedProbLoss(),
    'masked_prob_exp_loss': MaskedProbExpLoss(),
}
