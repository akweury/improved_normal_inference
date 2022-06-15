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


class AngleLoss(nn.Module):
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


class AngleLightLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0

        N_diff = (outputs[:, :3, :, :] - target[:, :3, :, :]).permute(0, 2, 3, 1)[mask]
        l_diff = (outputs[:, 3:6, :, :] - target[:, 5:8, :, :]).permute(0, 2, 3, 1)[mask]
        loss = torch.sum(N_diff ** 2) / N_diff.size(0)
        loss += torch.sum(l_diff ** 2) / l_diff.size(0)

        return loss


class AngleAlbedoLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        # normal l2 loss
        # normal_out = outputs[:, :3, :, :]
        # normals_target = target[:, :3, :, :]
        # mask_too_high = torch.gt(normal_out, 1).bool().detach()
        # mask_too_low = torch.lt(normal_out, -1).bool().detach()
        # normal_out[mask_too_high] = normal_out[mask_too_high] * args.penalty
        # normal_out[mask_too_low] = normal_out[mask_too_low] * args.penalty
        # axis = args.epoch % 3

        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0

        N_diff = (outputs[:, :3, :, :] - target[:, :3, :, :]).permute(0, 2, 3, 1)[mask]
        L_diff = (outputs[:, 3:6, :, :] - target[:, 5:8, :, :]).permute(0, 2, 3, 1)[mask]
        ScaleProd_diff = (outputs[:, 6, :, :] - target[:, 3, :, :])[mask]

        loss = torch.sum(N_diff ** 2) / N_diff.size(0)
        loss += torch.sum(L_diff ** 2) / L_diff.size(0)
        loss += torch.sum(ScaleProd_diff ** 2) / ScaleProd_diff.size(0)

        # light loss
        # light_target = target[:, 5:8, :, :]
        # light_output = outputs[:, 3:6, :, :]
        # l_diff = (light_output - light_target).permute(0, 2, 3, 1)[mask]
        # loss += torch.sum(l_diff ** 2) / (l_diff.size(0))

        # N*L loss
        # G_target = target[:, 3, :, :]
        # G_output = outputs[:, 3, :, :]

        # val_pixels = (~torch.prod(target == 0, 1).bool()).unsqueeze(1)
        # angle_loss = mu.angle_between_2d_tensor(outputs, target, mask=val_pixels).sum() / val_pixels.sum()
        # mask of non-zero positions
        # mask = mask.unsqueeze(1).repeat(1, 3, 1, 1).float()

        return loss  # +  F.mse_loss(outputs, target)  # + angle_loss.mul(args.angle_loss_weight)


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
    'angle': AngleLoss(),
    'angle_detail': AngleDetailLoss(),
    'angleAlbedo': AngleAlbedoLoss(),
    'angleLight': AngleLightLoss(),
}


# ----------------------------------------- Dataset Loader -------------------------------------------------------------
class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k, output_type, setname='train'):
        if setname in ['train', 'test']:
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

        train_dataset = SyntheticDepthDataset(dataset_path, self.args.neighbor, self.args.output_type, setname='train')
        test_dataset = SyntheticDepthDataset(dataset_path, self.args.neighbor, self.args.output_type, setname='test')
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
        print('\n- Found {} images in "{}" folder.'.format(test_data_loader.dataset.__len__(), 'test'))

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

            if self.exp_name == "degares":
                # load weight from pretrained resng model
                print(f'load model {config.resng_model}')
                pretrained_model = torch.load(config.resng_model)
                resng = pretrained_model['model']
                self.missing_keys = model.load_state_dict(resng.state_dict(), strict=False)

                # load optimizer
                # self.optimizer = pretrained_model['optimizer']

                # Print model's state_dict
                print("ResNg's state_dict:")
                for param_tensor in resng.state_dict():
                    print(param_tensor, "\t", resng.state_dict()[param_tensor].size())
                print("DeGaRes's state_dict:")
                for param_tensor in model.state_dict():
                    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

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

        if nn_model.args.exp == "ncnn":
            target[:, 5:8, :, :] = (target[:, 5:8, :, :] + 1) * 0.5
            target[:, :3, :, :] = (target[:, :3, :, :] + 1) * 0.5

        # Compute the loss
        loss = nn_model.loss(out, target, nn_model.args)

        if nn_model.args.exp == "ncnn":
            target[:, 5:8, :, :] = (target[:, 5:8, :, :] * 2) - 1
            target[:, :3, :, :] = (target[:, :3, :, :] * 2) - 1

        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # gpu_time = time.time() - start
        loss_total += loss.detach().to('cpu')

        if not nn_model.args.fast_train:
            if nn_model.args.angle_loss:
                angle_loss_avg = mu.eval_angle_tensor(out[:, :3, :, :], target[:, :3, :, :])
                angle_loss_avg_light = mu.eval_albedo_tensor(out[:, 3, :, :], target[:, 3, :, :])
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
                                    output_type=nn_model.args.output_type,
                                    prefix=f"eval_{test_idx}_")

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
        print(f'best loss updated to {loss_avg}.')
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


def draw_output(exp_name, x0, xout, target, exp_path, epoch, i, train_idx, output_type, prefix):
    target_normal = target[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
    target_img = target[0, :].permute(1, 2, 0)[:, :, 4].detach().numpy()
    target_light = target[0, :].permute(1, 2, 0)[:, :, 5:8].detach().numpy()
    xout_normal = xout[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
    if exp_name == "ag":
        xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()
        xout_scaleProd = xout[0, :].permute(1, 2, 0)[:, :, 6].detach().numpy()
    # xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()

    # if xout.size() != (512, 512, 3):
    # if cout is not None:
    #     cout = xout[0, :].permute(1, 2, 0)[:, :, 3:6]
    #     x1 = xout[0, :].permute(1, 2, 0)[:, :, 6:9]
    # else:
    #     x1 = None

    output_list = []

    # input normal
    if output_type == "noise":
        input = mu.tenor2numpy(x0[:1, :1, :, :])
    else:
        input = mu.tenor2numpy(x0[:1, :3, :, :])
    x0_normalized_8bit = mu.normalize2_32bit(input)
    x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
    mu.addText(x0_normalized_8bit, "Input(Vertex)")
    mu.addText(x0_normalized_8bit, str(train_idx[0]), pos='lower_left', font_size=0.3)
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

    elif exp_name in ["ag"]:
        # pred base normal
        xout_normal[mask] = 0
        xout_scaleProd[mask] = 0
        xout_light[mask] = 0
        if exp_name == "ncnn":
            xout_normal = xout_normal * 2 - 1
            xout_scaleProd = xout_scaleProd * 2 - 1
        normal_cnn_8bit = mu.visual_output(xout_normal, mask)

        mu.addText(normal_cnn_8bit, "Output")
        xout_base_ranges = mu.addHist(normal_cnn_8bit)
        mu.addText(normal_cnn_8bit, str(xout_base_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_8bit)

        # light output visualization
        xout_light_img = mu.normalize2_32bit(xout_light)
        xout_light_img = mu.image_resize(xout_light_img, width=512, height=512)
        mu.addText(xout_light_img, "Light_Output")
        mu.addText(xout_light_img, str(train_idx[0]), pos='lower_left', font_size=0.3)
        output_list.append(xout_light_img)

        # light ground truth visualization
        target_light_img = mu.normalize2_32bit(target_light)
        target_light_img = mu.image_resize(target_light_img, width=512, height=512)
        mu.addText(target_light_img, "Light_GT")
        mu.addText(target_light_img, str(train_idx[0]), pos='lower_left', font_size=0.3)
        output_list.append(target_light_img)

        # light error visualization
        diff_light_img, diff_light_angle = mu.eval_img_angle(xout_light, target_light)
        diff_light = np.sum(np.abs(diff_light_angle)) / np.count_nonzero(diff_light_angle)
        mu.addText(diff_light_img, "Light_Error")
        mu.addText(diff_light_img, f"angle error: {int(diff_light)}", pos="upper_right", font_size=0.65)
        output_list.append(diff_light_img)

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

    if output_type != "noise":
        output = cv.cvtColor(cv.hconcat(output_list), cv.COLOR_RGB2BGR)
    else:
        output = cv.hconcat(output_list)
    output_name = str(exp_path / f"{prefix}_epoch_{epoch}_{i}.png")
    cv.imwrite(output_name, output)


def main(args, exp_dir, network, train_dataset):
    nn_model = TrainingModel(args, exp_dir, network, train_dataset)

    print(f'- Training GPU: {nn_model.device}')
    print(f"- Training Date: {datetime.datetime.today().date()}\n")
    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        is_best = train_epoch(nn_model, epoch)

        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        if nn_model.args.fast_train:
            save_modulo = 100
        else:
            save_modulo = 1

        if epoch % save_modulo == 0:
            nn_model.save_checkpoint(is_best, epoch)


if __name__ == '__main__':
    """
    Call this method using 
     > main(args, exp_dir)
    """
    pass
