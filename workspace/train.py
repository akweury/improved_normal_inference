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
from torch import nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from help_funs import mu
from workspace import loss_utils

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


# ----------------------------------------- Dataset Loader -------------------------------------------------------------
class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k=0, output_type="normal_noise", setname='train'):
        if setname in ['train', 'selval', 'test']:
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
        self.losses = np.zeros((7, args.epochs))
        self.angle_losses = np.zeros((1, args.epochs))
        self.angle_losses_light = np.zeros((1, args.epochs))
        self.angle_sharp_losses = np.zeros((1, args.epochs))
        self.model = self.init_network(network)
        self.train_loader, self.test_loader = self.create_dataloader(dataset_path)
        self.val_loader = None
        self.albedo_loss = None
        self.normal_loss = None
        self.light_loss = None
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)
        self.save_model()
        self.pretrained_weight = None
        self.best_loss = 1e+20
        self.best_albedo_loss = 1e+20

    def create_dataloader(self, dataset_path):
        train_on = self.args.train_on

        train_dataset = SyntheticDepthDataset(dataset_path, setname='train')
        test_dataset = SyntheticDepthDataset(dataset_path, setname='selval')
        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
            train_dataset.training_case = train_dataset.training_case[training_idxs]
        print("test case number: " + str(test_dataset.training_case.shape))
        # test_dataset.training_case = test_dataset.training_case[:3]
        train_data_loader = DataLoader(train_dataset,
                                       shuffle=True,
                                       batch_size=self.args.batch_size,
                                       num_workers=4)
        test_data_loader = DataLoader(test_dataset,
                                      batch_size=self.args.batch_size,
                                      num_workers=4)
        print('\n- Found {} images in "{}" folder.'.format(train_data_loader.dataset.__len__(), 'train'))
        print('\n- Found {} images in "{}" folder.'.format(test_data_loader.dataset.__len__(), 'selval'))

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
            print(f"parameters that require grads: {self.parameters}")
            self.losses[:, :checkpoint['epoch']] = checkpoint['losses']
            self.angle_losses[:, :checkpoint['epoch']] = checkpoint['angle_losses']

            print(f"- checkout {checkpoint['epoch']} was loaded successfully!")
        else:
            print(f"------------ start a new training work -----------------")

            # init model
            model = network.to(self.device)
            self.parameters = filter(lambda p: p.requires_grad, model.parameters())
            print(f"parameters that require grads: {self.parameters}")
            # init optimizer
            if self.args.optimizer.lower() == 'sgd':
                self.optimizer = SGD(self.parameters, lr=self.args.lr, momentum=self.args.momentum, weight_decay=0)
            elif self.args.optimizer.lower() == 'adam':
                self.optimizer = Adam(self.parameters, lr=self.args.lr, weight_decay=0, amsgrad=True)
            else:
                raise ValueError

            # load pre-trained model
            if self.args.init_net is not None or self.args.init_light_net is not None:
                model.init_net()

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


def plot_loss_per_axis(loss_total, nn_model, epoch, title):
    loss_avg = loss_total / len(nn_model.train_loader.dataset)
    if title == "normal_loss":
        shift = 0
    elif title == "light_loss":
        shift = 3
    else:
        raise ValueError

    nn_model.losses[epoch % 3 + shift, epoch] = loss_avg

    # draw line chart
    if epoch % 10 == 9:
        draw_line_chart(np.array([nn_model.losses[0 + shift]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=0, title=title)
        draw_line_chart(np.array([nn_model.losses[1 + shift]]), nn_model.output_folder,
                        log_y=True, label=1, epoch=epoch, start_epoch=0, title=title)
        draw_line_chart(np.array([nn_model.losses[2 + shift]]), nn_model.output_folder,
                        log_y=True, label=2, epoch=epoch, start_epoch=0, cla_leg=True, title=title)


# ---------------------------------------------- Epoch ------------------------------------------------------------------
def train_epoch(nn_model, epoch):
    nn_model.args.epoch = epoch
    print(
        f"-{datetime.datetime.now().strftime('%H:%M:%S')} Epoch [{epoch}] lr={nn_model.optimizer.param_groups[0]['lr']:.1e}")
    # ------------ switch to train mode -------------------
    nn_model.model.train()
    normal_loss_total = torch.tensor([0.0])
    loss_total = torch.tensor([0.0])
    albedo_loss_total = torch.tensor([0.0])
    light_loss_total = torch.tensor([0.0])
    for i, (input, target, train_idx) in enumerate(nn_model.train_loader):
        # put input and target to device
        input, target, loss = input.float().to(nn_model.device), target.float().to(nn_model.device), torch.tensor(
            [0.0]).to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        # Compute the loss
        if nn_model.args.normal_loss:
            nn_model.normal_loss = loss_utils.weighted_unit_vector_loss(out[:, :3, :, :],
                                                                        target[:, :3, :, :],
                                                                        nn_model.args.penalty,
                                                                        epoch,
                                                                        nn_model.args.loss_type)

            loss += nn_model.normal_loss
            # for plot purpose
            normal_loss_total += nn_model.normal_loss.detach().to('cpu')
            loss_total += normal_loss_total

        if nn_model.args.albedo_loss:
            albedo_target = target[:, 4:5, :, :] / target[:, 3:4, :, :]
            print("albedo maxmin: " + str(albedo_target.max()) + str(albedo_target.min()))
            nn_model.albedo_loss = loss_utils.masked_l2_loss(out[:, 3:4, :, :], albedo_target)

            loss += nn_model.albedo_loss

            # for plot purpose
            albedo_loss_total += nn_model.albedo_loss.detach().to('cpu')
            loss_total += albedo_loss_total

        if nn_model.args.light_loss:
            nn_model.light_loss = loss_utils.weighted_unit_vector_loss(out[:, :3, :, :],
                                                                       target[:, 5:8, :, :],
                                                                       nn_model.args.penalty,
                                                                       epoch,
                                                                       nn_model.args.loss_type)
            loss += nn_model.light_loss

            # for plot purpose
            light_loss_total += nn_model.light_loss.detach().to('cpu')
            loss_total += light_loss_total

            # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        if i == 0:
            # print statistics
            np.set_printoptions(precision=5)
            torch.set_printoptions(sci_mode=True, precision=3)

            if nn_model.normal_loss is not None:
                normal_loss_0th_avg = nn_model.normal_loss / int(nn_model.args.batch_size)
                print(f"\t normal loss: {normal_loss_0th_avg:.2e}\t axis: {epoch % 3}", end="")
            if nn_model.albedo_loss is not None:
                albedo_loss_0th_avg = nn_model.albedo_loss / int(nn_model.args.batch_size)
                print(f"\t albedo loss: {albedo_loss_0th_avg:.2e}", end="")
            if nn_model.light_loss is not None:
                light_loss_0th_avg = nn_model.light_loss / int(nn_model.args.batch_size)
                print(f"\t light loss: {light_loss_0th_avg:.2e}", end="")
            print("\n")

    # save loss and plot
    if nn_model.args.normal_loss:
        plot_loss_per_axis(normal_loss_total, nn_model, epoch, title="normal_loss")
    if nn_model.args.light_loss:
        plot_loss_per_axis(light_loss_total, nn_model, epoch, title="light_loss")
    if nn_model.args.albedo_loss:
        nn_model.losses[6, epoch] = albedo_loss_total / len(nn_model.train_loader.dataset)
        draw_line_chart(np.array([nn_model.losses[6]]), nn_model.output_folder,
                        log_y=True, label="albedo", epoch=epoch, start_epoch=0, title="albedo_loss", cla_leg=True)

    # indicate for best model saving
    if nn_model.best_loss > loss_total:
        nn_model.best_loss = loss_total
        print(f'best loss updated to {float(loss_total / len(nn_model.train_loader.dataset)):.8e}')
        is_best = True
    else:
        is_best = False

    return is_best


def test_epoch(nn_model, epoch):
    for j, (input, target, test_idx) in enumerate(nn_model.test_loader):
        with torch.no_grad():
            # put input and target to device
            input, target = input.to(nn_model.device), target.to(nn_model.device)
            input = input[-1:, :]
            target = target[-1:, :]
            print(test_idx)
            test_idx = test_idx[-1]
            # Wait for all kernels to finish
            torch.cuda.synchronize()

            # Forward pass
            out = nn_model.model(input)
            input, out, target, test_idx = input.to("cpu"), out.to("cpu"), target.to("cpu"), test_idx.to(
                'cpu')
            if nn_model.exp_name == "light":
                input = input[:1, 4:7, :, :].permute(2, 3, 1, 0).squeeze(-1).detach().numpy()
                out = out[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
                target = target[0, :].permute(1, 2, 0)[:, :, 5:8].detach().numpy()
            if nn_model.exp_name == "nnnn":
                input = input[:1, :].permute(2, 3, 1, 0).squeeze(-1).detach().numpy()
                out = out[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()
                target = target[0, :].permute(1, 2, 0)[:, :, :3].detach().numpy()

            draw_output(nn_model.args.exp, input, out,
                        target=target,
                        exp_path=nn_model.output_folder,
                        epoch=epoch,
                        i=j,
                        train_idx=test_idx,
                        prefix=f"eval_epoch_{epoch}_{test_idx}_")


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


def draw_output(exp_name, input, xout, target, exp_path, epoch, i, train_idx, prefix):
    output_list = []

    # input
    vertex_0 = input[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    gt = target[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    x_out_normal = xout[0, :3, :, :].permute(1, 2, 0).to('cpu').numpy()
    print("x_out_normal shape: " + str(x_out_normal.shape))
    x0_normalized_8bit = mu.normalize2_32bit(vertex_0)
    mu.addText(x0_normalized_8bit, "Input(Vertex)")
    mu.addText(x0_normalized_8bit, str(train_idx), pos='lower_left', font_size=0.3)
    output_list.append(x0_normalized_8bit)

    # target
    target_img = mu.normal2RGB(gt)
    mask = gt.sum(axis=2) == 0
    target_gt_8bit = cv.normalize(target_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    mu.addText(target_gt_8bit, "GT")
    target_ranges = mu.addHist(target_img)
    mu.addText(target_gt_8bit, str(target_ranges), pos="upper_right", font_size=0.5)
    output_list.append(target_gt_8bit)

    # pred

    xout = mu.filter_noise(x_out_normal, threshold=[-1, 1])
    pred_img = mu.normal2RGB(xout)
    pred_img[mask] = 0
    normal_cnn_8bit = cv.normalize(pred_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    mu.addText(normal_cnn_8bit, "output")
    xout_ranges = mu.addHist(normal_cnn_8bit)
    mu.addText(normal_cnn_8bit, str(xout_ranges), pos="upper_right", font_size=0.5)
    output_list.append(normal_cnn_8bit)

    # err visualisation
    diff_img, diff_angle = mu.eval_img_angle(xout, gt)
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

        # evaluation
        if epoch % nn_model.args.print_freq == nn_model.args.print_freq - 1:
            test_epoch(nn_model, epoch)


if __name__ == '__main__':
    """
    Call this method using 
     > main(args, exp_dir)
    """
    pass
