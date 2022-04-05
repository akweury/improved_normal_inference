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

    def forward(self, outputs, target, penalty):
        outputs = outputs[:, :3, :, :]
        boarder_right = torch.gt(outputs, 255).bool().detach()
        boarder_left = torch.lt(outputs, 0).bool().detach()
        outputs[boarder_right] = outputs[boarder_right] * penalty
        outputs[boarder_left] = outputs[boarder_left] * penalty
        return F.mse_loss(outputs, target)


class AngleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, penalty_weight):
        outputs = outputs[:, :3, :, :]
        diff_angle = mu.angle_between_2d_tensor(outputs, target)
        # norm_penalty = torch.abs((torch.norm(outputs, dim=1) - 1))*penalty_weight
        return (torch.sum(diff_angle)) / (outputs.shape[0] * outputs.shape[2] * outputs.shape[3])


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
        val_pixels = torch.ne(target, 0).float().detach()
        return F.mse_loss(outputs * val_pixels, target * val_pixels)


loss_dict = {
    'l1': L1Loss(),
    'l2': L2Loss(),
    'masked_l1': MaskedL1Loss(),
    'masked_l2': MaskedL2Loss(),
    'weighted_l2': WeightedL2Loss(),
    'angle': AngleLoss()
}


# ----------------------------------------- Dataset Loader -------------------------------------------------------------
class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k, setname='train'):
        if setname in ['train', 'selval']:
            self.input = np.array(
                sorted(glob.glob(str(data_path / "train" / "tensor" / f"*_input_{k}.pt"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(data_path / "train" / "tensor" / f"*_gt_{k}.pt"), recursive=True)))

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
        self.losses = np.array([])
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)
        self.save_model()

    def create_dataloader(self, dataset_path):
        train_on = self.args.train_on
        dataset = SyntheticDepthDataset(dataset_path, self.args.neighbor, setname='train')
        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(dataset)), int(train_on)))
            dataset.input = dataset.input[training_idxs]
            dataset.gt = dataset.gt[training_idxs]

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
    print(
        f"-{datetime.datetime.today().date()} {datetime.datetime.now().strftime('%H-%M-%S')} Epoch [{epoch}] lr={nn_model.optimizer.param_groups[0]['lr']:.1e}",
        end="\t")
    # ------------ switch to train mode -------------------
    nn_model.model.train()
    loss_total = torch.tensor([0.0])
    start = time.time()
    for i, (input, target) in enumerate(nn_model.train_loader):
        # put input and target to device
        input, target = input.to(nn_model.device), target.to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # record data load time
        data_time = time.time() - start

        # start count the model time
        start = time.time()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        # Compute the loss
        loss = nn_model.loss(out, target, nn_model.args.penalty)

        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # record model time
        gpu_time = time.time() - start
        loss_total += loss.detach().to('cpu')
        if i == 0:
            # print statistics
            np.set_printoptions(precision=5)
            torch.set_printoptions(sci_mode=True, precision=3)
            input, out, target, = input.to("cpu"), out.to("cpu"), target.to("cpu")
            if epoch % 100 == 0:
                draw_output(input, out, target=target, exp_path=nn_model.output_folder,
                            loss=loss, epoch=epoch, i=i, prefix="train")

            print(f" loss: {loss:.2e}", end="    ")
            print(
                f'output range:[{out.min():.1f} - {out.max():.1f}], target range:[{target.min():.1f} - {target.max():.1f}]')
        start = time.time()
    loss_avg = loss_total / len(nn_model.train_loader.dataset)
    nn_model.losses = np.append(nn_model.losses, loss_avg)
    if epoch % 100 == 0:
        draw_line_chart(np.array([nn_model.losses]), nn_model.output_folder, log_y=True)

    # ---------------------------------------------- visualisation -------------------------------------------


def draw_line_chart(data, path, title=None, x_scale=None, y_scale=None, x_label=None, y_label=None,
                    show=False, log_y=False):
    if data.shape[1] <= 1:
        return

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for row in data:
        x = np.arange(row.shape[0]) * x_scale[1] + x_scale[0]
        y = row
        plt.plot(x, y)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}.png"))

    if show:
        plt.show()


def draw_output(x0, xout, target, exp_path, loss, epoch, i, prefix):
    if target.size() != (512, 512, 3):
        target = target[0, :].permute(1, 2, 0)[:, :, :3]
    if xout.size() != (512, 512, 3):
        xout = xout[0, :].permute(1, 2, 0)[:, :, :3]

    # input normal
    input = mu.tenor2numpy(x0[:1, :3, :, :])
    x0_normalized_8bit = mu.normalize2_8bit(input)
    x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
    mu.addText(x0_normalized_8bit, "Input(Normals)")

    # gt normal
    target = target.numpy()
    target = mu.normal2RGB(target)
    target_ranges = mu.addHist(target)
    target = cv.normalize(target, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    normal_gt_8bit = target
    mu.addText(normal_gt_8bit, "gt")
    mu.addText(normal_gt_8bit, str(target_ranges), pos="upper_right", font_size=0.5)

    xout = xout.detach().numpy()
    xout = mu.filter_noise(xout, threshold=[-1, 1])
    xout = mu.normal2RGB(xout)
    xout_ranges = mu.addHist(xout)
    normal_cnn_8bit = cv.normalize(xout, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    mu.addText(normal_cnn_8bit, "output")
    mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)

    output = cv.hconcat([normal_gt_8bit, normal_cnn_8bit])
    output_name = str(exp_path / f"{prefix}_epoch_{epoch}_{i}_loss_{loss:.18f}.png")
    cv.imwrite(output_name, output)


def main(args, exp_dir, network, train_dataset):
    nn_model = TrainingModel(args, exp_dir, network, train_dataset)

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
