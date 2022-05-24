#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 18:49:51 2022
@Author: J. Sha
"""
import os
import time
import shutil
import glob
import json
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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


class AngleDetailLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, args):
        # add penalty to the extreme values, i.e. out of range [-1,1]
        mask_too_high = torch.gt(outputs, 1).bool().detach()
        mask_too_low = torch.lt(outputs, -1).bool().detach()
        outputs[mask_too_high] = outputs[mask_too_high] * args.penalty
        outputs[mask_too_low] = outputs[mask_too_low] * args.penalty

        # mask of normals
        mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0
        mask = mask.unsqueeze(1).repeat(1, 3, 1, 1)
        outputs_smooth = outputs[:, :3, :, :]
        outputs_sharp = outputs[:, 3:6, :, :]
        mask_sharp = torch.sum(torch.abs(outputs_sharp), dim=1) > 0
        mask_sharp = mask_sharp.unsqueeze(1).repeat(1, 3, 1, 1)
        mask_smooth = (~mask_sharp) * mask

        axis = args.epoch % 3
        outputs_smooth[mask_sharp] = 0
        outputs_sharp[mask_smooth] = 0

        outputs_merged = outputs_sharp + outputs_smooth
        # smooth loss and sharp loss
        axis_diff = (outputs_merged - target)[:, axis, :, :][mask[:, 0, :, :]]
        loss = torch.sum(axis_diff ** 2) / (axis_diff.size(0))

        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
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
}


# ----------------------------------------- Dataset Loader -------------------------------------------------------------
class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k, output_type, setname='train'):
        if setname in ['train', 'test']:
            self.training_case = np.array(
                sorted(
                    glob.glob(str(data_path / setname / "tensor" / f"*_{k}_{output_type}.pth.tar"), recursive=True)))

    def __len__(self):
        return len(self.training_case)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        training_case = torch.load(self.training_case[item])
        input_tensor = training_case['input_tensor']
        gt_tensor = training_case['gt_tensor']
        scale_factors = training_case['scale_factors']

        return input_tensor, gt_tensor, scale_factors


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

        return input_tensor, gt_tensor


# ----------------------------------------- Training model -------------------------------------------------------------
class TrainingModel():
    def __init__(self, args, exp_dir, network, dataset_path, start_epoch=0):
        self.args = args
        self.start_epoch = start_epoch
        self.device = torch.device("cpu" if self.args.cpu else f"cuda:{self.args.gpu}")
        self.exp_name = self.args.exp
        self.exp_dir = Path(exp_dir)
        self.output_folder = self.init_output_folder()
        self.model = self.init_network(network)
        self.train_loader = self.create_dataloader(dataset_path)
        self.val_loader = None
        self.parameters = self.init_parameters()
        self.optimizer = self.init_optimizer()
        self.loss = loss_dict[args.loss].to(self.device)
        self.losses = np.zeros((3, args.epochs))
        self.angle_losses = np.zeros((1, args.epochs))
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)
        self.save_model()

    def create_dataloader(self, dataset_path):
        train_on = self.args.train_on
        if self.args.exp == "noise_net":
            dataset = NoiseDataset(dataset_path, self.args.neighbor, self.args.output_type, setname='train')
        else:
            dataset = SyntheticDepthDataset(dataset_path, self.args.neighbor, self.args.output_type, setname='train')
        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(dataset)), int(train_on)))
            dataset.training_case = dataset.training_case[training_idxs]

        data_loader = DataLoader(dataset,
                                 shuffle=True,
                                 batch_size=self.args.batch_size,
                                 num_workers=self.args.workers)

        print('\n- Found {} images in "{}" folder.'.format(data_loader.dataset.__len__(), 'train'))
        print('- Dataset "{}" was loaded successfully!'.format(self.args.dataset))

        return data_loader

    def init_output_folder(self):
        folder_path = self.exp_dir / f"output_{date_now}_{time_now}"
        if not os.path.exists(str(folder_path)):
            os.mkdir(str(folder_path))
        return folder_path

    def init_network(self, network):
        if self.args.resume:
            print(f'------------------ Resume a training work ----------------------- ')
            chkpt_path = self.args.resume
            assert os.path.isfile(chkpt_path), f"No checkpoint found at:{chkpt_path}"

            checkpoint = torch.load(chkpt_path)
            # self.args = checkpoint['args']
            self.start_epoch = checkpoint['epoch'] + 1
            print(f"- checkout {checkpoint['epoch']} was loaded successfully!")

            model = checkpoint['model'].to(self.device)
            return model
        else:
            print(f"------------ start a new training work -----------------")
            model = network.to(self.device)
            return model

    def init_parameters(self, ):
        return filter(lambda p: p.requires_grad, self.model.parameters())

    def init_lr_decayer(self):
        milestones = [int(x) for x in self.args.lr_scheduler.split(",")]
        self.lr_decayer = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                   gamma=self.args.lr_decay_factor)

    def init_optimizer(self):
        if self.args.optimizer.lower() == 'sgd':
            optimizer = SGD(self.parameters,
                            lr=self.args.lr,
                            momentum=self.args.momentum,
                            weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adam':
            optimizer = Adam(self.parameters,
                             lr=self.args.lr,
                             weight_decay=self.args.weight_decay,
                             amsgrad=True)
        else:
            raise ValueError
        return optimizer

    def print_info(self, args):
        print(f'\n------------------- Starting experiment {self.exp_name} ------------------ \n')
        print(f'- Model "{self.model.__name__}" was loaded successfully!')

    def save_checkpoint(self, is_best, epoch):
        checkpoint_filename = os.path.join(self.output_folder, 'checkpoint-' + str(epoch) + '.pth.tar')

        state = {'args': self.args,
                 'epoch': epoch,
                 'model': self.model,
                 'optimizer': self.optimizer}

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
    angle_loss_total = torch.tensor([0.0])
    for i, (input, target, scale_factor) in enumerate(nn_model.train_loader):
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
        mask = (~torch.prod(target == 0, 1).bool()).unsqueeze(1)
        angle_loss = mu.angle_between_2d_tensor(out[:, :3, :, :], target, mask=mask).sum() / mask.sum()
        angle_loss_total += angle_loss.to('cpu').detach().numpy()

        if i == 0:
            # print statistics
            np.set_printoptions(precision=5)
            torch.set_printoptions(sci_mode=True, precision=3)
            input, out, target, = input.to("cpu"), out.to("cpu"), target.to("cpu")
            if epoch % nn_model.args.print_freq == nn_model.args.print_freq - 1:
                draw_output(nn_model.args.exp, input, out, nn_model.args.cout, target=target,
                            exp_path=nn_model.output_folder,
                            loss=loss, epoch=epoch, i=i, output_type=nn_model.args.output_type, prefix="train")

            print(f"\t loss: {loss:.2e}\t axis: {epoch % 3}")
    loss_avg = loss_total / len(nn_model.train_loader.dataset)
    angle_loss_avg = angle_loss_total / len(nn_model.train_loader.dataset)

    nn_model.losses[epoch % 3, epoch] = loss_avg
    nn_model.angle_losses[0, epoch] = angle_loss_avg
    if epoch % 10 == 9:
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[1]]), nn_model.output_folder,
                        log_y=True, label=1, epoch=epoch, start_epoch=nn_model.start_epoch)
        draw_line_chart(np.array([nn_model.losses[2]]), nn_model.output_folder,
                        log_y=True, label=2, epoch=epoch, start_epoch=nn_model.start_epoch, cla_leg=True)
        draw_line_chart(np.array([nn_model.angle_losses[0]]), nn_model.output_folder, log_y=True,
                        epoch=epoch, start_epoch=nn_model.start_epoch, loss_type="angle", cla_leg=True)

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


def draw_output(exp_name, x0, xout, cout, target, exp_path, loss, epoch, i, output_type, prefix):
    if target.size() != (512, 512, 3):
        target = target[0, :].permute(1, 2, 0)[:, :, :3]
    # if xout.size() != (512, 512, 3):
    # if cout is not None:
    #     cout = xout[0, :].permute(1, 2, 0)[:, :, 3:6]
    #     x1 = xout[0, :].permute(1, 2, 0)[:, :, 6:9]
    # else:
    #     x1 = None
    xout = xout[0, :].permute(1, 2, 0).detach().numpy()

    output_list = []

    # input normal
    if output_type == "noise":
        input = mu.tenor2numpy(x0[:1, :1, :, :])
    else:
        input = mu.tenor2numpy(x0[:1, :3, :, :])
    x0_normalized_8bit = mu.normalize2_8bit(input)
    x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
    mu.addText(x0_normalized_8bit, "Input(Vertex)")
    output_list.append(x0_normalized_8bit)

    # gt normal
    target = target.numpy()
    target_img = mu.normal2RGB(target)
    mask = target.sum(axis=2) == 0
    target_ranges = mu.addHist(target_img)
    normal_gt_8bit = cv.normalize(target_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    mu.addText(normal_gt_8bit, "gt")
    mu.addText(normal_gt_8bit, str(target_ranges), pos="upper_right", font_size=0.5)
    output_list.append(normal_gt_8bit)

    if exp_name == "degares":
        # pred base normal
        xout_base = xout[:, :, :3]
        xout_base = mu.filter_noise(xout_base, threshold=[-1, 1])
        xout_base[(np.sum(xout[:, :, 3:], axis=2) != 0)] = 0
        pred_base_img = mu.normal2RGB(xout_base)
        pred_base_img[mask] = 0
        normal_cnn_base_8bit = cv.normalize(pred_base_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # pred sharp normal
        xout_sharp = xout[:, :, 3:]
        xout_sharp = mu.filter_noise(xout_sharp, threshold=[-1, 1])
        pred__sharp_img = mu.normal2RGB(xout_sharp)
        pred__sharp_img[mask] = 0
        normal_cnn_sharp_8bit = cv.normalize(pred__sharp_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # pred combined normal
        xout = xout_base + xout_sharp
        normal_cnn_8bit = normal_cnn_base_8bit + normal_cnn_sharp_8bit

        mu.addText(normal_cnn_base_8bit, "output_base")
        xout_base_ranges = mu.addHist(normal_cnn_base_8bit)
        mu.addText(normal_cnn_base_8bit, str(xout_base_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_base_8bit)

        mu.addText(normal_cnn_sharp_8bit, "output_sharp")
        xout_sharp_ranges = mu.addHist(normal_cnn_sharp_8bit)
        mu.addText(normal_cnn_sharp_8bit, str(xout_sharp_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_sharp_8bit)

        mu.addText(normal_cnn_8bit, "output")
        xout_ranges = mu.addHist(normal_cnn_8bit)
        mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_8bit)

    else:
        # pred normal
        xout = xout[:, :, :3]
        xout = mu.filter_noise(xout, threshold=[-1, 1])
        pred_img = mu.normal2RGB(xout)
        pred_img[mask] = 0
        normal_cnn_8bit = cv.normalize(pred_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        mu.addText(normal_cnn_8bit, "output")
        xout_ranges = mu.addHist(normal_cnn_8bit)
        mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)
        output_list.append(normal_cnn_8bit)

    # err visualisation
    diff_img, diff_angle = mu.eval_img_angle(xout, target)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)
    mu.addText(diff_img, "Error")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)
    output_list.append(diff_img)

    # cout
    # if cout is not None:
    #     cout = cout.detach().numpy()
    #     if output_type == 'normal' or 'normal_noise':
    #         cout = mu.filter_noise(cout, threshold=[0, 1])
    #         cout = mu.normal2RGB(cout)
    #     else:
    #         cout = mu.filter_noise(cout, threshold=[0, 255])
    #
    #     normal_cout_8bit = cv.normalize(cout, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    #     normal_cout_8bit = cv.applyColorMap(normal_cout_8bit, cv.COLORMAP_BONE)
    #     cout_ranges = mu.addHist(cout)
    #     mu.addText(normal_cout_8bit, "c_out")
    #     mu.addText(normal_cout_8bit, str(cout_ranges), pos="upper_right", font_size=0.5)
    #     output_list.append(normal_cout_8bit)

    if output_type != "noise":
        output = cv.cvtColor(cv.hconcat(output_list), cv.COLOR_RGB2BGR)
    else:
        output = cv.hconcat(output_list)
    output_name = str(exp_path / f"{prefix}_epoch_{epoch}_{i}_loss_{loss:.8f}.png")
    cv.imwrite(output_name, output)


def main(args, exp_dir, network, train_dataset):
    nn_model = TrainingModel(args, exp_dir, network, train_dataset)

    print(f'- Training GPU: {nn_model.device}')
    print(f"- Training Date: {datetime.datetime.today().date()}\n")
    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        train_epoch(nn_model, epoch)

        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        nn_model.save_checkpoint(False, epoch)


if __name__ == '__main__':
    """
    Call this method using 
     > main(args, exp_dir)
    """
    pass
