#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 11-03-2022 at 16:25
@author: J. Sha
"""

import sys
from os.path import dirname
import time

import numpy as np
import torch

sys.path.append(dirname(__file__))

from help_funs import mu


def eval(dataloader, farthest_neighbour):
    print("---------- Start SVD Evaluation --------")

    loss_list = np.zeros(len(dataloader.dataset))
    time_list = np.zeros(len(dataloader.dataset))

    for i, (input, target) in enumerate(dataloader):
        input, target = input.to("cpu"), target.to("cpu")

        vertex = input.squeeze(0).permute(1, 2, 0).numpy()
        target = target.squeeze(0).permute(1, 2, 0).numpy()
        normal, normal_img = mu.vertex2normal(vertex, farthest_neighbour)

        mask = np.sum(np.abs(target), axis=2) == 0

        # start count the model time
        start = time.time()

        # evaluation
        angle_loss = mu.angle_between(normal[~mask], target[~mask]).sum() / mask.sum()

        # record data load time
        gpu_time = time.time() - start

        # record time and loss
        loss_list[i] = angle_loss
        time_list[i] = gpu_time
        print(f"[SVD] Test Case: {i}/{loss_list.shape[0]}, Angle Loss: {angle_loss}, Time: {gpu_time}")

    return loss_list, time_list


def eval_single(v, mask, cam_pos, farthest_neighbour):
    normal, normal_img = mu.vertex2normal(v, mask, cam_pos, farthest_neighbour)
    return normal


if __name__ == '__main__':
    pass
