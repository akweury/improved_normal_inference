#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created in 11-03-2022 at 16:25
@author: J. Sha
"""

import sys
from os.path import dirname
import json
import numpy as np
import torch

sys.path.append(dirname(__file__))

from help_funs import mu, file_io, data_preprocess
import config


def eval(dataloader, farthest_neighbour):
    print("---------- Start SVD Evaluation --------")

    loss_list = np.zeros(len(dataloader.dataset))
    time_list = np.zeros(len(dataloader.dataset))
    size_list = np.zeros(len(dataloader.dataset))
    median_loss_list = np.zeros(dataloader.dataset.__len__())

    d5_list = np.zeros(dataloader.dataset.__len__())
    d11_list = np.zeros(dataloader.dataset.__len__())
    d22_list = np.zeros(dataloader.dataset.__len__())
    d30_list = np.zeros(dataloader.dataset.__len__())

    for i, (input, target, idx) in enumerate(dataloader):
        input, target = input.to("cpu"), target.to("cpu")

        vertex = input.squeeze(0).permute(1, 2, 0)[:, :, :3].numpy()
        target = target.squeeze(0).permute(1, 2, 0)[:, :, :3].numpy()
        mask = target.sum(axis=2) == 0
        size_list[i] = np.sum(~mask)

        # start count the model time
        normal, gpu_time = eval_single(vertex, ~mask, np.array([0, 0, -7]), farthest_neighbour=2, data_idx=idx)

        # evaluation
        # angle_loss = mu.angle_between(normal[~mask], target[~mask]).sum() / mask.sum()
        # normal_target = torch.from_numpy(target)
        # normal = torch.from_numpy(normal)
        diff, median_err, deg_diff_5, deg_diff_11d25, deg_diff_22d5, deg_diff_30 = mu.avg_angle_between_np(
            normal, target)

        # record data load time

        # record time and loss
        loss_list[i] = diff
        time_list[i] = gpu_time * 1000

        median_loss_list[i] = median_err
        d5_list[i] = deg_diff_5
        d11_list[i] = deg_diff_11d25
        d22_list[i] = deg_diff_22d5
        d30_list[i] = deg_diff_30

        # print(f"[SVD] Test Case: {i}/{loss_list.shape[0]}, Angle Loss: {diff}, Time: {(gpu_time * 1000):.2e} ms")

    return loss_list, time_list, size_list, median_loss_list, d5_list, d11_list, d22_list, d30_list


def eval_single(v, mask, cam_pos, farthest_neighbour, data_idx):
    # data_path = config.synthetic_data / 'synthetic128' / "selval"
    # image_file, ply_file, json_file, depth_file, depth_noise_file, normal_file = file_io.get_file_name(data_idx, data_path)
    #
    # # input vertex
    # f = open(json_file)
    # data = json.load(f)
    # f.close()
    #
    #
    # depth = file_io.load_scaled16bitImage(depth_file,
    #                                       data['minDepth'],
    #                                       data['maxDepth'])
    #
    #
    # mask = depth.sum(axis=2) == 0
    #
    # data['R'], data['t'] = np.identity(3), np.zeros(3)
    # vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
    #                          torch.tensor(data['K']),
    #                          torch.tensor(data['R']).float(),
    #                          torch.tensor(data['t']).float())
    # vertex[mask] = 0
    #
    # vertex_norm, scale_factors, shift_vector = data_preprocess.vectex_normalization(vertex, mask)

    normal, normal_img, gpu_time = mu.vertex2normal(v, mask, cam_pos, farthest_neighbour)
    return normal, gpu_time


if __name__ == '__main__':
    pass
