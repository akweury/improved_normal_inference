#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:49:51 2022
@Author: J. Sha
"""
import datetime
import glob
import json
import os
import shutil
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import config
from help_funs import mu

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")
mse_criterion = torch.nn.MSELoss(reduction='mean')


# https://github.com/tyui592/Perceptual_loss_for_real_time_style_transfer/blob/master/train.py#L76
def extract_features(model, x, layers):
    features = list()
    for index, layer in enumerate(model):
        x = layer(x.float())
        if index in layers:
            features.append(x)
    return features


def gram(x):
    b, c, h, w = x.size()
    g = torch.bmm(x.view(b, c, h * w), x.view(b, c, h * w).transpose(1, 2))
    return g.div(h * w)


def calc_Content_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1 / len(features)] * len(features)

    content_loss = 0
    for f, t, w in zip(features, targets, weights):
        content_loss += mse_criterion(f, t) * w

    return content_loss


def calc_Gram_Loss(features, targets, weights=None):
    if weights is None:
        weights = [1 / len(features)] * len(features)

    gram_loss = 0
    for f, t, w in zip(features, targets, weights):
        gram_loss += mse_criterion(gram(f), gram(t)) * w
    return gram_loss


def calc_TV_Loss(x):
    tv_loss = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    tv_loss += torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return tv_loss


# -------------------------------------------------- loss function ---------------------------------------------------
class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :3, :, :]
        return F.mse_loss(outputs, target)


class WeightedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        outputs = outputs[:, :3, :, :]
        boarder_right = torch.gt(outputs, 255).bool().detach()
        boarder_left = torch.lt(outputs, 0).bool().detach()
        outputs[boarder_right] = outputs[boarder_right] * args.penalty
        outputs[boarder_left] = outputs[boarder_left] * args.penalty
        return F.mse_loss(outputs, target)


class NormalL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        outputs = outputs[:, :3, :, :]
        target = target[:, :3, :, :]
        mask_too_high = torch.gt(outputs, 1).bool().detach()
        mask_too_low = torch.lt(outputs, -1).bool().detach()
        outputs[mask_too_high] = outputs[mask_too_high] * args.penalty
        outputs[mask_too_low] = outputs[mask_too_low] * args.penalty

        # val_pixels = (~torch.prod(target == 0, 1).bool()).unsqueeze(1)
        # angle_loss = mu.angle_between_2d_tensor(outputs, target, mask=val_pixels).sum() / val_pixels.sum()
        # mask of non-zero positions
        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0
        # mask = mask.unsqueeze(1).repeat(1, 3, 1, 1).float()

        axis = args.epoch % 3
        axis_diff = (outputs - target)[:, axis, :, :][mask]
        loss = torch.sum(axis_diff ** 2) / (axis_diff.size(0))
        # print(f"\t axis: {axis}\t axis_loss: {loss:.5f}")

        return loss  # +  F.mse_loss(outputs, target)  # + angle_loss.mul(args.angle_loss_weight)


class GL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        mask = torch.abs(target) > 0
        G_diff = (outputs - target)[mask]
        loss = torch.sum(G_diff ** 2) / G_diff.size(0)

        return loss


class ImageL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, mask):
        mask = mask.bool()
        Image_diff = (outputs - target)[mask]
        loss = torch.sum(Image_diff ** 2) / Image_diff.size(0)

        return loss


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


class AngleLightLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.normal_loss = NormalL2Loss()
        self.g_loss = GL2Loss()
        self.img_loss = ImageL2Loss()

    def forward(self, outputs, target, args):
        normal_loss = self.normal_loss(outputs[:, :3, :, :], target[:, :3, :, :], args)

        input_mask = outputs[:, 4, :, :]
        G = torch.sum(outputs[:, :3, :, :] * target[:, 5:8, :, :], dim=1)  # N * L
        out_img = outputs[:, 3, :, :] * G
        img_loss = self.img_loss(out_img, target[:, 4, :, :], input_mask)

        loss = normal_loss + img_loss * args.albedo_penalty

        return loss


class AngleAlbedoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        # normal l2 loss
        mask_too_high = torch.gt(outputs, 1).bool().detach()
        mask_too_low = torch.lt(outputs, -1).bool().detach()
        outputs[mask_too_high] = outputs[mask_too_high] * args.penalty
        outputs[mask_too_low] = outputs[mask_too_low] * args.penalty
        axis = args.epoch % 3
        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0

        axis_target = torch.cat((target[:, :3, :, :], target[:, 5:8, :, :]), 1)
        axis_output = outputs[:, :6, :, :]
        axis_diff_normal = (axis_output - axis_target)[:, axis, :, :][mask]
        axis_diff_light = (axis_output - axis_target)[:, axis + 3, :, :][mask]
        ScaleProd_diff = (outputs[:, 6, :, :] - target[:, 3, :, :])[mask]

        loss = torch.sum(axis_diff_normal ** 2) / axis_diff_normal.size(0)
        loss += torch.sum(axis_diff_light ** 2) / axis_diff_light.size(0)
        loss += torch.sum(ScaleProd_diff ** 2) / ScaleProd_diff.size(0)

        # light loss
        # light_target = target[:, 5:8, :, :]
        # light_output = outputs[:, 3:6, :, :]
        # l_diff = (light_output - light_target).permute(0, 2, 3, 1)[mask]
        # loss += torch.sum(l_diff ** 2) / (l_diff.size(0))

        # N*L loss
        # G_target = target[:, 3, :, :]
        # G_output = outputs[:, 3, :, :]

        # add angle loss
        # light_rad_loss = mu.eval_angle_tensor(outputs[:, 3:6, :, :], target[:, 5:8, :, :]) * args.angle_loss_weight
        # normal_rad_loss = mu.eval_angle_tensor(outputs[:, :3, :, :], target[:, :3, :, :]) * args.angle_loss_weight
        # loss += light_rad_loss
        # loss += normal_rad_loss

        return loss


class AngleDetailLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        # add penalty to the extreme values, i.e. out of range [-1,1]
        mask_too_high = torch.gt(outputs, 1).bool().detach()
        mask_too_low = torch.lt(outputs, -1).bool().detach()
        outputs[mask_too_high] = outputs[mask_too_high] * args.penalty
        outputs[mask_too_low] = outputs[mask_too_low] * args.penalty

        # smooth loss and sharp loss
        axis = args.epoch % 3

        # mask of normals
        mask_smooth = (torch.abs(outputs[:, axis, :, :]) > 0)
        mask_sharp = (torch.abs(outputs[:, axis + 3, :, :]) > 0)

        target = target.permute(0, 2, 3, 1)
        target_smooth = target[:, :, :, axis][mask_smooth]
        output_smooth = outputs[:, axis, :, :]
        output_smooth = output_smooth[mask_smooth]

        target_sharp = target[:, :, :, axis][mask_sharp]
        output_sharp = outputs[:, axis + 3, :, :]
        output_sharp = output_sharp[mask_sharp]

        axis_smooth_diff = output_smooth - target_smooth
        axis_sharp_diff = output_sharp - target_sharp
        loss = torch.sum(axis_smooth_diff ** 2) / (axis_smooth_diff.size(0)) + \
               torch.sum(axis_sharp_diff ** 2) * args.sharp_penalty / ((axis_sharp_diff.size(0)) + 1e-20)

        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target):
        outputs = outputs[:, :3, :, :]
        return F.l1_loss(outputs, target)


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :3, :, :]
        val_pixels = torch.ne(target, 0).float().detach()
        return F.l1_loss(outputs * val_pixels, target * val_pixels)


class MaskedL2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        outputs = outputs[:, :3, :, :]
        # val_pixels = torch.ne(target, 0).float().detach()
        val_pixels = (~torch.prod(target == 0, 1).bool()).float().unsqueeze(1)
        return F.mse_loss(outputs * val_pixels, target * val_pixels)


loss_dict = {
    'l1': L1Loss(),
    'l2': L2Loss(),
    'masked_l1': MaskedL1Loss(),
    'masked_l2': MaskedL2Loss(),
    'weighted_l2': WeightedL2Loss(),
    'angle': NormalL2Loss(),
    'angle_detail': AngleDetailLoss(),
    'angleAlbedo': AngleAlbedoLoss(),
    'angleLight': AngleLightLoss(),
    'angle_histo': AngleHistoLoss(),
}


# ----------------------------------------- Dataset Loader -------------------------------------------------------------
class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k=0, output_type="normal_noise", setname='train'):
        if setname in ['train', 'val', 'test']:
            self.training_case = np.array(
                sorted(
                    glob.glob(str(data_path / setname / "tensor" / f"*_{k}_{output_type}.pth.tar"), recursive=True)))
        else:
            self.training_case = np.array(
                sorted(
                    glob.glob(str(data_path / "tensor" / f"*_{k}_{output_type}.pth.tar"), recursive=True)))

    def __len__(self):
        return len(self.training_case)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        training_case = torch.load(self.training_case[item])
        input_tensor = training_case['input_tensor']
        gt_tensor = training_case['gt_tensor']
        scale_factors = training_case['scale_factors']

        return input_tensor, gt_tensor, item


class NoiseDataset(Dataset):

    def __init__(self, data_path, k, output_type, setname='train'):
        if setname in ['train']:
            self.input = np.array(
                sorted(
                    glob.glob(str(data_path / "tensor" / f"*_input_{k}_{output_type}.pt"), recursive=True)))
            self.gt = np.array(
                sorted(glob.glob(str(data_path / "tensor" / f"*_gt_{k}_{output_type}.pt"), recursive=True)))

        assert (len(self.gt) == len(self.input))

    def __len__(self):
        return len(self.input)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        input_tensor = torch.load(self.input[item])
        gt_tensor = torch.load(self.gt[item])

        return input_tensor, gt_tensor, item


# ----------------------------------------- Training model -------------------------------------------------------------
class TrainingModel():
    def __init__(self, args, exp_dir, network, dataset_path, start_epoch=0):
        self.missing_keys = None
        self.args = args
        self.start_epoch = start_epoch
        self.device = torch.device(f"cuda:0")
        self.exp_name = self.args.exp
        self.exp_dir = Path(exp_dir)
        self.output_folder = self.init_output_folder()
        self.optimizer = None
        self.parameters = None
        self.losses = np.zeros((3, args.epochs))
        self.angle_losses = np.zeros((1, args.epochs))
        self.angle_losses_light = np.zeros((1, args.epochs))
        self.angle_sharp_losses = np.zeros((1, args.epochs))
        self.model = self.init_network(network)
        self.train_loader, self.test_loader = self.create_dataloader(dataset_path)
        self.val_loader = None
        self.loss = loss_dict[args.loss].to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)
        self.save_model()
        self.pretrained_weight = None
        self.best_loss = 1e+20

    def create_dataloader(self, dataset_path):
        train_on = self.args.train_on

        train_dataset = SyntheticDepthDataset(dataset_path, setname='train')
        test_dataset = SyntheticDepthDataset(dataset_path, setname='val')
        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
            train_dataset.training_case = train_dataset.training_case[training_idxs]

        test_dataset.training_case = test_dataset.training_case[:3]
        train_data_loader = DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=self.args.batch_size,
                                       num_workers=4)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=self.args.batch_size,
                                      num_workers=4)
        print('\n- Found {} images in "{}" folder.'.format(train_data_loader.dataset.__len__(), 'train'))
        print('\n- Found {} images in "{}" folder.'.format(test_data_loader.dataset.__len__(), 'val'))

        return train_data_loader, test_data_loader

    def init_output_folder(self):
        folder_path = self.exp_dir / f"output_{date_now}_{time_now}"
        if not os.path.exists(str(folder_path)):
            os.mkdir(str(folder_path))
        return folder_path

    def init_network(self, network):
        if self.args.resume:
            print(f'------------------ Resume a training work ----------------------- ')
            assert os.path.isfile(self.args.resume), f"No checkpoint found at:{self.args.resume}"
            checkpoint = torch.load(self.args.resume)
            self.start_epoch = checkpoint['epoch'] + 1  # resume epoch
            model = checkpoint['model'].to(self.device)  # resume model
            self.optimizer = checkpoint['optimizer']  # resume optimizer
            # self.optimizer = SGD(self.parameters, lr=self.args.lr, momentum=self.args.momentum, weight_decay=0)
            # self.args = checkpoint['args']
            self.parameters = filter(lambda p: p.requires_grad, model.parameters())

            self.losses[:, :checkpoint['epoch']] = checkpoint['losses']
            self.angle_losses[:, :checkpoint['epoch']] = checkpoint['angle_losses']

            print(f"- checkout {checkpoint['epoch']} was loaded successfully!")
            return model
        else:
            print(f"------------ start a new training work -----------------")

            # init model
            model = network.to(self.device)
            self.parameters = filter(lambda p: p.requires_grad, model.parameters())

            # init optimizer
            if self.args.optimizer.lower() == 'sgd':
                self.optimizer = SGD(self.parameters, lr=self.args.lr, momentum=self.args.momentum, weight_decay=0)
            elif self.args.optimizer.lower() == 'adam':
                self.optimizer = Adam(self.parameters, lr=self.args.lr, weight_decay=0, amsgrad=True)
            else:
                raise ValueError
            if self.args.init_net != None:
                model.init_net(self.args.init_net)
            # if self.exp_name == "degares":
            #     # load weight from pretrained resng model
            #     print(f'load model {config.resng_model}')
            #     pretrained_model = torch.load(config.resng_model)
            #     resng = pretrained_model['model']
            #     self.missing_keys = model.load_state_dict(resng.state_dict(), strict=False)
            #
            #     # load optimizer
            #     # self.optimizer = pretrained_model['optimizer']
            #
            #     # Print model's state_dict
            #     print("ResNg's state_dict:")
            #     for param_tensor in resng.state_dict():
            #         print(param_tensor, "\t", resng.state_dict()[param_tensor].size())
            #     print("DeGaRes's state_dict:")
            #     for param_tensor in model.state_dict():
            #         print(param_tensor, "\t", model.state_dict()[param_tensor].size())

            return model

    def init_lr_decayer(self):
        milestones = [int(x) for x in self.args.lr_scheduler.split(",")]
        self.lr_decayer = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                   gamma=self.args.lr_decay_factor)

    def print_info(self, args):
        print(f'\n------------------- Starting experiment {self.exp_name} ------------------ \n')
        print(f'- Model "{self.model.__name__}" was loaded successfully!')

    def save_checkpoint(self, is_best, epoch):
        checkpoint_filename = os.path.join(self.output_folder, 'checkpoint-' + str(epoch) + '.pth.tar')

        state = {'args': self.args,
                 'epoch': epoch,
                 'model': self.model,
                 'optimizer': self.optimizer,
                 'angle_losses': self.angle_losses[:, :epoch],
                 'losses': self.losses[:, :epoch],
                 }

        torch.save(state, checkpoint_filename)

        if is_best:
            best_filename = os.path.join(self.output_folder, 'model_best.pth.tar')
            shutil.copyfile(checkpoint_filename, best_filename)

        if epoch > 0:
            prev_checkpoint_filename = os.path.join(self.output_folder, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
            if os.path.exists(prev_checkpoint_filename):
                os.remove(prev_checkpoint_filename)

    def save_model(self):
        with open(str(Path(self.output_folder) / "param.json"), 'w') as f:
            json.dump(vars(self.args), f)
        with open(str(Path(self.output_folder) / "model.txt"), 'w') as f:
            f.write(str(self.model))


# ---------------------------------------------- Epoch ------------------------------------------------------------------
def train_epoch(nn_model, epoch):
    nn_model.args.epoch = epoch
    print(
        f"-{datetime.datetime.now().strftime('%H:%M:%S')} Epoch [{epoch}] lr={nn_model.optimizer.param_groups[0]['lr']:.1e}")
    # ------------ switch to train mode -------------------
    nn_model.model.train()
    loss_total = torch.tensor([0.0])
    angle_loss_avg = torch.tensor([0.0])
    angle_loss_avg_light = torch.tensor([0.0])
    angle_loss_sharp_total = torch.tensor([0.0])
    for i, (input, target, train_idx) in enumerate(nn_model.train_loader):
        # put input and target to device
        input, target = input.to(nn_model.device), target.to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        # Compute the loss
        loss = nn_model.loss(out, target, nn_model.args)

        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # gpu_time = time.time() - start
        loss_total += loss.detach().to('cpu')

        if not nn_model.args.fast_train:
            if nn_model.args.angle_loss:
                angle_loss_avg = mu.eval_angle_tensor(out[:, :3, :, :], target[:, :3, :, :])
                # angle_loss_avg_light = mu.eval_albedo_tensor(out[:, 3, :, :], target[:, 3, :, :])
                # angle_loss_total += mu.output_radians_loss(out[:, :3, :, :], target[:, :3, :, :]).to('cpu').detach().numpy()

        # visualisation
        if i == 0:
            # print statistics
            np.set_printoptions(precision=5)
            torch.set_printoptions(sci_mode=True, precision=3)
            input, out, target, train_idx = input.to("cpu"), out.to("cpu"), target.to("cpu"), train_idx.to('cpu')
            loss_0th = loss / int(nn_model.args.batch_size)
            print(f"\t loss: {loss_0th:.2e}\t axis: {epoch % 3}")

            # evaluation
            if epoch % nn_model.args.print_freq == nn_model.args.print_freq - 1:
                for j, (input, target, test_idx) in enumerate(nn_model.test_loader):
                    with torch.no_grad():
                        # put input and target to device
                        input, target = input.to(nn_model.device), target.to(nn_model.device)

                        # Wait for all kernels to finish
                        torch.cuda.synchronize()

                        # Forward pass
                        out = nn_model.model(input)
                        input, out, target, test_idx = input.to("cpu"), out.to("cpu"), target.to("cpu"), test_idx.to(
                            'cpu')
                        draw_output(nn_model.args.exp, input, out,
                                    target=target,
                                    exp_path=nn_model.output_folder,
                                    epoch=epoch,
                                    i=j,
                                    train_idx=test_idx,
                                    prefix=f"eval_epoch_{epoch}_{test_idx}_")

    # save loss
    loss_avg = loss_total / len(nn_model.train_loader.dataset)
    nn_model.losses[epoch % 3, epoch] = loss_avg
    if not nn_model.args.fast_train:
        if nn_model.args.angle_loss:
            angle_loss_avg = angle_loss_avg.to("cpu")
            angle_loss_avg_light = angle_loss_avg_light.to("cpu")
            nn_model.angle_losses[0, epoch] = angle_loss_avg
            nn_model.angle_losses_light[0, epoch] = angle_loss_avg_light

            if nn_model.args.exp == "degares":
                angle_loss_sharp_avg = angle_loss_sharp_total / len(nn_model.train_loader.dataset)
                nn_model.angle_sharp_losses[0, epoch] = angle_loss_sharp_avg

    # indicate for best model saving
    if nn_model.best_loss > loss_avg:
        nn_model.best_loss = loss_avg
        print(f'best loss updated to {float(loss_avg):.8e}')
        is_best = True
    else:
        is_best = False

    # draw line chart
    if epoch % 10 == 9:
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[1]]), nn_model.output_folder,
                        log_y=True, label=1, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[2]]), nn_model.output_folder,
                        log_y=True, label=2, epoch=epoch, start_epoch=nn_model.start_epoch, cla_leg=True, title="Loss")
        if nn_model.args.angle_loss:
            draw_line_chart(np.array([nn_model.angle_losses[0]]), nn_model.output_folder, log_y=True, label="normal",
                            epoch=epoch, start_epoch=nn_model.start_epoch, loss_type="angle")
            draw_line_chart(np.array([nn_model.angle_losses_light[0]]), nn_model.output_folder, log_y=True,
                            label="albedo",
                            epoch=epoch, start_epoch=nn_model.start_epoch, loss_type="angle", cla_leg=True)
    return is_best


def train_fugrc(nn_model, epoch, loss_network):
    nn_model.args.epoch = epoch
    print(
        f"-{datetime.datetime.now().strftime('%H:%M:%S')} Epoch [{epoch}] lr={nn_model.optimizer.param_groups[0]['lr']:.1e}")
    # ------------ switch to train mode -------------------
    nn_model.model.train()
    loss_logs = {'content_loss': [], 'style_loss': [], 'tv_loss': [], 'total_loss': []}
    # loss network
    # loss_network = None
    for i, (input, target, train_idx) in enumerate(nn_model.train_loader):
        # put input and target to device
        input, target = input[:, :3, :, :].to(nn_model.device), target[:, :3, :, :].to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        if nn_model.args.loss == "perceptual_loss":
            target_content_features = extract_features(loss_network, input, nn_model.args.content_layers)
            target_style_features = extract_features(loss_network, target, nn_model.args.style_layers)

            output_content_features = extract_features(loss_network, out, nn_model.args.content_layers)
            output_style_features = extract_features(loss_network, out, nn_model.args.style_layers)

            # Compute the loss
            content_loss = calc_Content_Loss(output_content_features, target_content_features)
            style_loss = calc_Gram_Loss(output_style_features, target_style_features)
            tv_loss = calc_TV_Loss(out)

            loss = content_loss * nn_model.args.content_weight + style_loss * nn_model.args.style_weight + tv_loss * nn_model.args.tv_weight

            loss_logs['content_loss'].append(float(content_loss))
            loss_logs['style_loss'].append(float(style_loss))
            loss_logs['tv_loss'].append(tv_loss.item())
            loss_logs['total_loss'].append(loss.item())
        else:
            loss = nn_model.loss(out, target, nn_model.args)
            loss_logs['total_loss'].append(float(loss))
            # print(f"loss: {loss}")
        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # print statistics
        np.set_printoptions(precision=5)
        torch.set_printoptions(sci_mode=True, precision=3)

        # evaluation
        if epoch % nn_model.args.print_freq == nn_model.args.print_freq - 1:
            for j, (input, target, test_idx) in enumerate(nn_model.test_loader):
                with torch.no_grad():
                    # put input and target to device
                    input, target = input.to(nn_model.device), target.to(nn_model.device)
                    # Wait for all kernels to finish
                    torch.cuda.synchronize()
                    # Forward pass
                    out = nn_model.model(input)
                    input, out, target, test_idx = input.to("cpu"), out.to("cpu"), target.to("cpu"), test_idx.to(
                        'cpu')
                    draw_output(nn_model.args.exp, input, out,
                                target=target,
                                exp_path=nn_model.output_folder,
                                epoch=epoch,
                                i=j,
                                train_idx=test_idx,
                                prefix=f"eval_epoch_{epoch}_{test_idx}_")

    # save loss
    nn_model.losses[0, epoch] = (np.sum(loss_logs['total_loss']) / len(nn_model.train_loader.dataset))
    nn_model.losses[1, epoch] = (np.sum(loss_logs['content_loss']) / len(nn_model.train_loader.dataset))
    nn_model.losses[2, epoch] = (np.sum(loss_logs['style_loss']) / len(nn_model.train_loader.dataset))

    # indicate for best model saving
    if nn_model.best_loss > nn_model.losses[0, epoch]:
        nn_model.best_loss = nn_model.losses[0, epoch]
        print(f'best loss updated to {nn_model.losses[0, epoch]}.')
        is_best = True
    else:
        is_best = False

    # draw line chart
    if epoch % 10 == 9:
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=nn_model.start_epoch, cla_leg=True, title="Loss")
    return is_best


# ---------------------------------------------- visualisation -------------------------------------------

def draw_line_chart(data_1, path, title=None, x_label=None, y_label=None, show=False, log_y=False,
                    label=None, epoch=None, cla_leg=False, start_epoch=0, loss_type="mse"):
    if data_1.shape[1] <= 1:
        return

    x = np.arange(epoch - start_epoch)
    y = data_1[0, start_epoch:epoch]
    x = x[y.nonzero()]
    y = y[y.nonzero()]
    plt.plot(x, y, label=label)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)
    if not os.path.exists(str(path)):
        os.mkdir(path)

    if loss_type == "mse":
        plt.savefig(str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}.png"))
    elif loss_type == "angle":
        plt.savefig(str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}_angle.png"))
    else:
        raise ValueError("loss type is not supported.")

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


def draw_output(exp_name, x0, xout, target, exp_path, epoch, i, train_idx, prefix):
    target_normal = target[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
    # xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()
    # xout_scaleProd = xout[0, :].permute(1, 2, 0)[:, :, 6].detach().numpy()
    # xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()
    target_img = target[0, :].permute(1, 2, 0)[:, :, 4].detach().numpy()
    target_light = target[0, :].permute(1, 2, 0)[:, :, 5:8].detach().numpy()
    xout_normal = xout[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
    # if exp_name == "ag":
    #     xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()
    #     xout_scaleProd = xout[0, :].permute(1, 2, 0)[:, :, 6].detach().numpy()

    # if xout.size() != (512, 512, 3):
    # if cout is not None:
    #     cout = xout[0, :].permute(1, 2, 0)[:, :, 3:6]
    #     x1 = xout[0, :].permute(1, 2, 0)[:, :, 6:9]
    # else:
    #     x1 = None

    output_list = []

    # input normal
    input = x0[:1, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).detach().numpy()
    x0_normalized_8bit = mu.normalize2_32bit(input)
    mu.addText(x0_normalized_8bit, "Input(Vertex)")
    mu.addText(x0_normalized_8bit, str(train_idx), pos='lower_left', font_size=0.3)
    output_list.append(x0_normalized_8bit)

    # gt normal
    normal_img = mu.normal2RGB(target_normal)
    mask = target_normal.sum(axis=2) == 0
    target_ranges = mu.addHist(normal_img)
    normal_gt_8bit = cv.normalize(normal_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    mu.addText(normal_gt_8bit, "GT")
    mu.addText(normal_gt_8bit, str(target_ranges), pos="upper_right", font_size=0.5)
    output_list.append(normal_gt_8bit)

    if exp_name == "degares":
        # pred base normal
        normal_cnn_base_8bit = mu.visual_output(xout[:, :, :3], mask)

        # pred sharp normal
        normal_cnn_sharp_8bit = mu.visual_output(xout[:, :, 3:6], mask)

        # pred combined normal
        pred_normal = xout[:, :, :3] + xout[:, :, 3:6]
        normal_cnn_8bit = mu.visual_output(pred_normal, mask)

        mu.addText(normal_cnn_base_8bit, "output_base")
        xout_base_ranges = mu.addHist(normal_cnn_base_8bit)
        mu.addText(normal_cnn_base_8bit, str(xout_base_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_base_8bit)

        # sharp edge detection input

        x1_normalized_8bit = mu.normalize2_32bit(xout[:, :, 6:9])
        x1_normalized_8bit = mu.image_resize(x1_normalized_8bit, width=512, height=512)
        mu.addText(x1_normalized_8bit, "Input(Sharp)")
        output_list.append(x1_normalized_8bit)

        mu.addText(normal_cnn_sharp_8bit, "output_sharp")
        xout_sharp_ranges = mu.addHist(normal_cnn_sharp_8bit)
        mu.addText(normal_cnn_sharp_8bit, str(xout_sharp_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_sharp_8bit)

        mu.addText(normal_cnn_8bit, "Output")
        xout_ranges = mu.addHist(normal_cnn_8bit)
        mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_8bit)

    else:
        # pred normal
        pred_normal = xout_normal[:, :, :3]
        pred_normal = mu.filter_noise(pred_normal, threshold=[-1, 1])
        pred_img = mu.normal2RGB(pred_normal)
        pred_img[mask] = 0
        normal_cnn_8bit = cv.normalize(pred_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        mu.addText(normal_cnn_8bit, "output")
        xout_ranges = mu.addHist(normal_cnn_8bit)
        mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_8bit)

    # err visualisation
    diff_img, diff_angle = mu.eval_img_angle(xout_normal, target_normal)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)
    mu.addText(diff_img, "Error")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)
    output_list.append(diff_img)

    output = cv.cvtColor(cv.hconcat(output_list), cv.COLOR_RGB2BGR)

    output_name = str(exp_path / f"{prefix}_epoch_{epoch}_{i}.png")
    cv.imwrite(output_name, output)


def main(args, exp_dir, network, train_dataset):
    nn_model = TrainingModel(args, exp_dir, network, train_dataset)

    print(f'- Training GPU: {nn_model.device}')
    print(f"- Training Date: {datetime.datetime.today().date()}\n")
    # if nn_model.args.exp == "fugrc":
    #     loss_network = torchvision.models.__dict__[nn_model.args.vgg_flag](pretrained=True).features.to(nn_model.device)
    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        # if nn_model.args.exp == "fugrc":
        #     is_best = train_fugrc(nn_model, epoch, loss_network)
        # else:
        is_best = train_epoch(nn_model, epoch)
        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        nn_model.save_checkpoint(is_best, epoch)


if __name__ == '__main__':
    """
    Call this method using 
     > main(args, exp_dir)
    """
    pass
