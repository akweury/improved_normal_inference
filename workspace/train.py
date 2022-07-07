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
        self.losses = np.zeros((4, args.epochs))
        self.angle_losses = np.zeros((1, args.epochs))
        self.angle_losses_light = np.zeros((1, args.epochs))
        self.angle_sharp_losses = np.zeros((1, args.epochs))
        self.model = self.init_network(network)
        self.train_loader, self.test_loader = self.create_dataloader(dataset_path)
        self.val_loader = None
        self.img_loss = None
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)
        self.save_model()
        self.pretrained_weight = None
        self.best_loss = 1e+20

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


def plot_loss_per_axis(loss_total, nn_model, epoch):
    loss_avg = loss_total / len(nn_model.train_loader.dataset)

    nn_model.losses[epoch % 3, epoch] = loss_avg

    # draw line chart
    if epoch % 10 == 9:
        draw_line_chart(np.array([nn_model.losses[0]]), nn_model.output_folder,
                        log_y=True, label=0, epoch=epoch, start_epoch=0)
        draw_line_chart(np.array([nn_model.losses[1]]), nn_model.output_folder,
                        log_y=True, label=1, epoch=epoch, start_epoch=0)
        draw_line_chart(np.array([nn_model.losses[2]]), nn_model.output_folder,
                        log_y=True, label=2, epoch=epoch, start_epoch=0, cla_leg=True, title="Loss")


# ---------------------------------------------- Epoch ------------------------------------------------------------------
def train_epoch(nn_model, epoch):
    nn_model.args.epoch = epoch
    print(
        f"-{datetime.datetime.now().strftime('%H:%M:%S')} Epoch [{epoch}] lr={nn_model.optimizer.param_groups[0]['lr']:.1e}")
    # ------------ switch to train mode -------------------
    nn_model.model.train()
    loss_total = torch.tensor([0.0])
    angle_loss_avg = torch.tensor([0.0])
    img_loss_total = torch.tensor([0.0])

    for i, (input, target, train_idx) in enumerate(nn_model.train_loader):
        # put input and target to device
        input, target = input.float().to(nn_model.device), target.float().to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        # Compute the loss
        loss = loss_utils.weighted_normal_loss(out[:, :3, :, :],
                                               target[:, :3, :, :],
                                               nn_model.args.penalty,
                                               epoch,
                                               nn_model.args.loss_type)

        if nn_model.args.img_loss:
            nn_model.img_loss = loss_utils.LambertError(normal=target[:, :3, :, :],
                                                        albedo=out[:, 3:4, :, :],
                                                        lighting=target[:, 5:8, :, :],
                                                        image=target[:, 4:5, :, :], ) * nn_model.args.img_penalty
            loss += nn_model.img_loss
            # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # gpu_time = time.time() - start
        loss_total += loss.detach().to('cpu')
        img_loss_total += nn_model.img_loss.detach().to('cpu')

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
            loss_0th_avg = loss / int(nn_model.args.batch_size)
            print(f"\t normal loss: {loss_0th_avg:.2e}\t axis: {epoch % 3}", end="")
            if nn_model.img_loss is not None:
                img_loss_0th_avg = nn_model.img_loss / int(nn_model.args.batch_size)
                print(f"\t img loss: {img_loss_0th_avg:.2e}", "")
            print("\n")

            # evaluation
            if epoch % nn_model.args.print_freq == nn_model.args.print_freq - 1:
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
                        draw_output(nn_model.args.exp, input, out,
                                    target=target,
                                    exp_path=nn_model.output_folder,
                                    epoch=epoch,
                                    i=j,
                                    train_idx=test_idx,
                                    prefix=f"eval_epoch_{epoch}_{test_idx}_")

    # save loss and plot
    plot_loss_per_axis(loss_total, nn_model, epoch)

    nn_model.losses[3, epoch] = img_loss_total / len(nn_model.train_loader.dataset)
    draw_line_chart(np.array([nn_model.losses[3]]), nn_model.output_folder,
                    log_y=True, label="image", epoch=epoch, start_epoch=0)

    # indicate for best model saving
    if nn_model.best_loss > loss_total:
        nn_model.best_loss = loss_total
        print(f'best loss updated to {float(loss_total / len(nn_model.train_loader.dataset)):.8e}')
        is_best = True
    else:
        is_best = False

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

    # xout_light = xout[0, :].permute(1, 2, 0)[:, :, 3:6].detach().numpy()
    target_img = target[0, :].permute(1, 2, 0)[:, :, 4].detach().numpy()
    target_light = target[0, :].permute(1, 2, 0)[:, :, 5:8].detach().numpy()
    target_scaleProd = target[0, :].permute(1, 2, 0)[:, :, 3].detach().numpy()
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
    elif exp_name == "ag":
        xout_scaleProd = xout[0, :].permute(1, 2, 0)[:, :, 5].detach().numpy()
        xout_albedo = xout[0, :].permute(1, 2, 0)[:, :, 3].detach().numpy()

        # img_out = xout_albedo * xout_scaleProd
        # img_out[mask] = 0
        # img_out = np.uint8(img_out)
        # img_out = mu.visual_img(img_out, "img_out")
        # output_list.append(img_out)

        # xout_albedo = np.uint8(xout_albedo)
        # output_list.append(mu.visual_img(xout_albedo, "albedo_out"))

        xout_albedo2 = np.uint8(target_img / (xout_scaleProd + 1e-20))
        output_list.append(mu.visual_img(xout_albedo2, "albedo_out_2"))

        albedo_gt = np.uint8(target_img / (target_scaleProd + 1e-20))
        output_list.append(mu.visual_img(albedo_gt, "albedo_gt"))

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
