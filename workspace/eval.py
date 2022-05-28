#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 24-03-2022 at 13:19
@author: J. Sha
"""

import sys
from os.path import dirname

import config

sys.path.append(dirname(__file__))

import time
import torch
import numpy as np
from help_funs import mu, data_preprocess
from pncnn.utils import args_parser
from workspace.svd import eval as svd
from torch.utils.data import Dataset, DataLoader
import glob

from workspace.train import SyntheticDepthDataset


def eval(dataset_path, name, model_path, gpu=0):
    print(f"---------- Start {name} Evaluation --------")
    print(f"load checkpoint... ", end="")
    # SVD model
    if model_path is None:
        # load dataset
        dataset = SyntheticDepthDataset(dataset_path, 1, "normal_noise", setname='test')
        data_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)
        loss_list, time_list = svd.eval(data_loader, 2)
        return loss_list, time_list

    # load model
    checkpoint = torch.load(model_path)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    model = checkpoint['model']

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()  # Swith to evaluate mode
    print("ok.")
    print("load dataset...", end="")
    # load dataset
    if dataset_path == config.real_data:
        dataset = SyntheticDepthDataset(dataset_path, args.neighbor, "normal_noise", setname=None)
    else:
        dataset = SyntheticDepthDataset(dataset_path, args.neighbor, "normal_noise", setname="test")
    data_loader = DataLoader(dataset,
                             shuffle=True,
                             batch_size=1,
                             num_workers=1)

    print(f'ok, {data_loader.dataset.__len__()} items.')

    loss_list = np.zeros(data_loader.dataset.__len__())
    time_list = np.zeros(data_loader.dataset.__len__())

    for i, (input, target, scale_factor) in enumerate(data_loader):
        with torch.no_grad():
            # put input and target to device
            input, target = input.to(device), target.to(device)

            # Wait for all kernels to finish
            torch.cuda.synchronize()

            # start count the model time
            start = time.time()

            # Forward pass
            out = model(input)

            # record data load time
            gpu_time = time.time() - start

            # calculate loss
            out = mu.filter_noise(out, threshold=[-1, 1])

            diff = mu.output_radians_loss(out[:, :3, :, :], target).to("cpu").detach().numpy()
            if args.exp == "degares":
                diff += mu.output_radians_loss(out[:, 3:6, :, :], target).to("cpu").detach().numpy()

            loss_list[i] = diff
            time_list[i] = gpu_time * 1000
            print(
                f"[{name}] Test Case: {i + 1}/{loss_list.shape[0]}, Angle Loss: {diff:.2e}, Time: {(gpu_time * 1000):.2e} ms")

    return loss_list, time_list


def eval_post_processing(normal, normal_img, normal_gt, name):
    out_ranges = mu.addHist(normal_img)
    mu.addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)
    mu.addText(normal_img, name, font_size=0.8)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img, diff


if __name__ == '__main__':
    pass
