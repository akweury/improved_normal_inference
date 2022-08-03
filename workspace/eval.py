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
from help_funs import mu
from workspace.svd import eval as svd
from torch.utils.data import DataLoader

from workspace.train import SyntheticDepthDataset


def eval(dataset_path, name, model_path, gpu=0, data_type="normal_noise", setname="individual"):
    print(f"---------- Start {name} Evaluation --------")
    print(f"load checkpoint... ", end="")
    # SVD model
    if model_path is None:
        # load dataset
        dataset = SyntheticDepthDataset(dataset_path, 0, data_type, setname=setname)
        data_loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1)
        loss_list, time_list, size_list, median_loss_list, d5_list, d11_list, d22_list, d30_list = svd.eval(data_loader,
                                                                                                            2)
        return loss_list, time_list, size_list, median_loss_list, d5_list, d11_list, d22_list, d30_list

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

    dataset = SyntheticDepthDataset(dataset_path, 0, data_type, setname=setname)
    data_loader = DataLoader(dataset,
                             shuffle=True,
                             batch_size=1,
                             num_workers=1)

    print(f'ok, {data_loader.dataset.__len__()} items.')

    loss_list = np.zeros(data_loader.dataset.__len__())
    median_loss_list = np.zeros(data_loader.dataset.__len__())
    time_list = np.zeros(data_loader.dataset.__len__())
    size_list = np.zeros(data_loader.dataset.__len__())

    d5_list = np.zeros(data_loader.dataset.__len__())
    d11_list = np.zeros(data_loader.dataset.__len__())
    d22_list = np.zeros(data_loader.dataset.__len__())
    d30_list = np.zeros(data_loader.dataset.__len__())

    for i, (input, target, scale_factor) in enumerate(data_loader):
        with torch.no_grad():
            mask = torch.sum(torch.abs(target[:, :3, :, :]), dim=1) > 0
            mask = mask.permute(1, 2, 0).squeeze(-1)
            size_list[i] = mask.sum()
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

            if "light" in name:
                light_output = out[:, :3, :, :].permute(0, 2, 3, 1).squeeze(0)
                light_gt = target[:, 13:16, :, :].permute(0, 2, 3, 1).squeeze(0)
                diff = mu.avg_angle_between_tensor(light_output[mask], light_gt[mask]).to("cpu").detach().numpy()
            else:

                normal = out[:, :3, :, :].permute(0, 2, 3, 1).squeeze(0)
                normal_target = target[:, :3, :, :].permute(0, 2, 3, 1).squeeze(0)
                diff, median_err, deg_diff_5, deg_diff_11d25, deg_diff_22d5, deg_diff_30 = mu.avg_angle_between_tensor(
                    normal[mask], normal_target[mask])

            loss_list[i] = diff
            time_list[i] = gpu_time * 1000

            median_loss_list[i] = median_err
            d5_list[i] = deg_diff_5
            d11_list[i] = deg_diff_11d25
            d22_list[i] = deg_diff_22d5
            d30_list[i] = deg_diff_30
            # print(
            #     f"[{name}] Test Case: {i + 1}/{loss_list.shape[0]}, Angle Loss: {diff:.2e}, Time: {(gpu_time * 1000):.2e} ms")

    return loss_list, time_list, size_list, median_loss_list, d5_list, d11_list, d22_list, d30_list


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
