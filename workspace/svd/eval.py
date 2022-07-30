#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 11-03-2022 at 16:25
@author: J. Sha
"""

import sys
from os.path import dirname

import numpy as np
import torch

sys.path.append(dirname(__file__))

from help_funs import mu


def eval(dataloader, farthest_neighbour):
    print("---------- Start SVD Evaluation --------")

    loss_list = np.zeros(len(dataloader.dataset))
    time_list = np.zeros(len(dataloader.dataset))
    size_list = np.zeros(len(dataloader.dataset))

    for i, (input, target, idx) in enumerate(dataloader):
        input, target = input.to("cpu"), target.to("cpu")

        vertex = input.squeeze(0).permute(1, 2, 0)[:, :, :3].numpy()
        target = target.squeeze(0).permute(1, 2, 0)[:, :, :3].numpy()
        mask = target.sum(axis=2) == 0
        size_list[i] = np.sum(~mask)

        # start count the model time
        normal, gpu_time = eval_single(vertex, ~mask, np.array([0, 0.8, 7.5]), farthest_neighbour=2)

        # evaluation
        # angle_loss = mu.angle_between(normal[~mask], target[~mask]).sum() / mask.sum()
        normal_target = torch.from_numpy(target)[~mask]
        normal = torch.from_numpy(normal)[~mask]
        diff = mu.avg_angle_between_tensor(normal, normal_target).to("cpu").detach().numpy()

        # record data load time

        # record time and loss
        loss_list[i] = diff
        time_list[i] = gpu_time * 1000
        # print(f"[SVD] Test Case: {i}/{loss_list.shape[0]}, Angle Loss: {diff}, Time: {(gpu_time * 1000):.2e} ms")

    return loss_list, time_list, size_list


def eval_single(v, mask, cam_pos, farthest_neighbour):
    normal, normal_img, gpu_time = mu.vertex2normal(v, mask, cam_pos, farthest_neighbour)
    return normal, gpu_time


if __name__ == '__main__':
    pass
