#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 24-03-2022 at 13:19
@author: J. Sha
"""

import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import time
import torch
import numpy as np
from help_funs import mu, data_preprocess
from pncnn.utils import args_parser
from workspace.svd import eval as svd
from torch.utils.data import Dataset, DataLoader
import glob


class SyntheticDepthDataset(Dataset):

    def __init__(self, data_path, k, output_type):
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


def eval(dataset_path, name, model_path, gpu=0):
    print(f"---------- Start {name} Evaluation --------")
    print(f"load checkpoint... ", end="")
    # SVD model
    if model_path is None:
        # load dataset
        dataset = SyntheticDepthDataset(dataset_path, 1, "normal_noise")
        data_loader = DataLoader(dataset, shuffle=True, batch_size=1, num_workers=1)
        loss_list, time_list = svd.eval(data_loader, 2)
        return loss_list, time_list

    # load model
    checkpoint = torch.load(model_path)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    model = checkpoint['model']

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(gpu))
    model = model.to(device)
    model.eval()  # Swith to evaluate mode
    print("ok.")
    print("load dataset...", end="")
    # load dataset
    dataset = SyntheticDepthDataset(dataset_path, args.neighbor, "normal_noise")
    data_loader = DataLoader(dataset,
                             shuffle=True,
                             batch_size=1,
                             num_workers=1)

    print(f'ok, {data_loader.dataset.__len__()} items.')

    loss_list = np.zeros(data_loader.dataset.__len__())
    time_list = np.zeros(data_loader.dataset.__len__())

    for i, (input, target) in enumerate(data_loader):
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
            mask = (~torch.prod(target == 0, 1).bool()).unsqueeze(1)

            output = out[0, :].permute(1, 2, 0)[:, :, :3]
            target = target[0, :].permute(1, 2, 0)[:, :, :3]

            output = output.to('cpu').numpy()
            target = target.to('cpu').numpy()

            diff_img, diff_angle = mu.eval_img_angle(output, target)

            diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

            loss_list[i] = diff
            time_list[i] = gpu_time
            print(
                f"[{name}] Test Case: {i}/{loss_list.shape[0]}, Angle Loss: {diff:.2e}, Time: {(gpu_time * 1000):.2e} ms")

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
