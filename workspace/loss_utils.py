import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def LambertError(normal, albedo, lighting, image):
    """mean Lambert Reconstruction Error"""

    mask = ~torch.prod(lighting == 0, dim=1).bool()
    recon = albedo * torch.sum(normal * lighting, dim=1, keepdim=True)
    error = torch.norm(image - recon, p=2, dim=1)

    return error[mask].mean()


def l1_loss(outputs, target, ):
    return F.l1_loss(outputs, target)


def l2_loss(outputs, target):
    return F.mse_loss(outputs, target)


def masked_l1_loss(outputs, target):
    outputs = outputs[:, :3, :, :]
    val_pixels = torch.ne(target, 0).float().detach()
    return F.l1_loss(outputs * val_pixels, target * val_pixels)


def masked_l2_loss(outputs, target):
    outputs = outputs[:, :3, :, :]
    # val_pixels = torch.ne(target, 0).float().detach()
    val_pixels = (~torch.prod(target == 0, 1).bool()).float().unsqueeze(1)
    return F.mse_loss(outputs * val_pixels, target * val_pixels)


def weighted_log_l1_loss(outputs, target, penalty, scaleMin, scaleMax):
    boarder_right = torch.gt(outputs, scaleMax).bool().detach()
    boarder_left = torch.lt(outputs, scaleMin).bool().detach()
    outputs[boarder_right] = outputs[boarder_right] * penalty
    outputs[boarder_left] = outputs[boarder_left] * penalty
    mask = ~torch.prod(target == 0, dim=1, keepdim=True).bool()
    outputs[~mask] = 0
    target[~mask] = 0
    return torch.sum(torch.sqrt(torch.abs(outputs[mask] - target[mask]) + 1e-20)) / torch.sum(mask)


def weighted_l2_loss(outputs, target, penalty, scaleMin, scaleMax):
    outputs = outputs.permute(0, 2, 3, 1)
    target = target.permute(0, 2, 3, 1)
    boarder_right = torch.gt(outputs, scaleMax).bool().detach()
    boarder_left = torch.lt(outputs, scaleMin).bool().detach()
    outputs[boarder_right] = outputs[boarder_right] * penalty
    outputs[boarder_left] = outputs[boarder_left] * penalty
    mask = ~torch.prod(target == 0, dim=-1).bool()
    outputs[~mask] = 0
    target[~mask] = 0
    return F.mse_loss(outputs, target)


def weighted_unit_vector_loss(outputs, target, penalty, epoch, loss_type, scale_threshold=1):
    # give penalty to outliers
    outputs = outputs[:, :3, :, :]
    target = target[:, :3, :, :]
    mask_too_high = torch.gt(outputs, scale_threshold).bool().detach()
    mask_too_low = torch.lt(outputs, -scale_threshold).bool().detach()
    outputs[mask_too_high] = outputs[mask_too_high] * penalty
    outputs[mask_too_low] = outputs[mask_too_low] * penalty

    # calc the loss
    mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0
    axis = epoch % 3
    if loss_type == "l2":
        loss = F.mse_loss(outputs[:, axis, :, :][mask], target[:, axis, :, :][mask]).float()
    elif loss_type == "l1":
        loss = F.l1_loss(outputs[:, axis, :, :][mask], target[:, axis, :, :][mask]).float()
    else:
        raise ValueError

    return loss.float()


class AngleHistoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        outputs = outputs[:, :3, :, :]
        target = target[:, :3, :, :]
        mask_too_high = torch.gt(outputs, 1).bool().detach()
        mask_too_low = torch.lt(outputs, -1).bool().detach()
        outputs[mask_too_high] = outputs[mask_too_high] * args.penalty
        outputs[mask_too_low] = outputs[mask_too_low] * args.penalty

        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0

        axis = args.epoch % 3
        axis_diff = (outputs - target)[:, axis, :, :][mask]
        loss = torch.sum(axis_diff ** 2) / (axis_diff.size(0))
        # https://discuss.pytorch.org/t/differentiable-torch-histc/25865/3
        min = -1.05
        max = 1.05
        bins = 100
        delta = float(max - min) / float(bins)
        centers = min + delta * (torch.arange(bins).float() + 0.5)
        centers = centers.to(outputs.device)

        sigma = 0.6
        output_histo = outputs[:, axis, :, :][mask].unsqueeze(0) - centers.unsqueeze(1)
        output_histo = torch.exp(-0.5 * (output_histo / sigma) ** 2) / (sigma * np.sqrt(np.pi * 2)) * delta
        output_histo = output_histo.sum(dim=-1)
        output_histo = output_histo / output_histo.sum(dim=-1)  # normalization

        target_histo = target[:, axis, :, :][mask].unsqueeze(0) - centers.unsqueeze(1)
        target_histo = torch.exp(-0.5 * (target_histo / sigma) ** 2) / (sigma * np.sqrt(np.pi * 2)) * delta
        target_histo = target_histo.sum(dim=-1)
        target_histo = target_histo / target_histo.sum(dim=-1)  # normalization

        histo_loss = output_histo - target_histo
        # for i in range(outputs.size(0)):
        #     histo_mask = mask[i, :, :].to(outputs.device)
        #     histo_output = torch.histc(outputs[i, axis, :, :][histo_mask], bins=256, min=-1.05, max=1.05)
        #     histo_target = torch.histc(target[i, axis, :, :][histo_mask], bins=256, min=-1.05, max=1.05)
        #     histo_loss += torch.sum(torch.abs(histo_output - histo_target)) * (1e-6)

        return loss + histo_loss


def weighted_unit_vector_huber_loss(outputs, target, penalty, epoch, loss_type, scale_threshold=1):
    # give penalty to outliers
    outputs = outputs[:, :3, :, :].permute(0, 2, 3, 1)
    target = target[:, :3, :, :].permute(0, 2, 3, 1)
    mask = torch.sum(torch.abs(target[:, :, :, :3]), dim=-1) > 0
    mask = mask.unsqueeze(-1)
    scale_threshold = 0.2 * torch.max(torch.abs(outputs - target))
    mask_high = (torch.gt(torch.abs(outputs), scale_threshold).bool() * mask)
    mask_low = (torch.lt(torch.abs(outputs), scale_threshold).bool() * mask)

    loss = torch.abs(outputs[mask_low] - target[mask_low]).mean()

    loss += ((torch.pow((outputs[mask_high] - target[mask_high]), 2) + scale_threshold ** 2) / (
            2 * scale_threshold)).mean()

    return loss.float()
