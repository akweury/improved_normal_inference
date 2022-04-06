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
import cv2 as cv
import numpy as np
from help_funs import mu
from pncnn.utils import args_parser
from common import data_preprocess


def eval(vertex, model_path, k, output_type='rgb'):
    # load model
    checkpoint = torch.load(model_path)

    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print('- Checkpoint was loaded successfully.')

    # Compare the checkpoint args with the json file in case I wanted to change some args
    # args_parser.compare_args_w_json(args, exp_dir, start_epoch + 1)
    args.evaluate = model_path

    # load model
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    model = checkpoint['model'].to(device)

    args_parser.print_args(args)

    mask = vertex.sum(axis=2) == 0
    # move all the vertex as close to original point as possible,
    vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / vertex[:, :, :1][
        ~mask].max()
    vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / vertex[:, :, 1:2][
        ~mask].max()
    vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / vertex[:, :, 2:3][
        ~mask].max()

    # calculate delta x, y, z of between each point and its neighbors
    vectors = data_preprocess.neighbor_vectors_k(vertex, k)
    vectors[mask] = 0

    input_tensor = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
    input_tensor = input_tensor.permute(2, 0, 1)

    normal, normal_img, eval_point_counter, total_time = evaluate_epoch(model, input_tensor, start_epoch, device,
                                                                        output_type)

    return normal, normal_img, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, input_tensor, epoch, device, output_type='normal'):
    model.eval()  # Swith to evaluate mode

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)
        torch.cuda.synchronize()

        # Forward Pass
        start = time.time()
        output = model(input_tensor)
        gpu_time = time.time() - start

        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)[:, :, :3]
        output = output.to('cpu').numpy()
    mask = input_tensor.sum(axis=1) == 0
    mask = mask.to('cpu').numpy().reshape(512, 512)
    eval_point_counter = np.sum(mask)
    if output_type == 'normal':
        # normal = output / np.linalg.norm(output, axis=2, ord=2, keepdims=True)
        normal = mu.filter_noise(output, threshold=[-1, 1])
        output_img = mu.normal2RGB(normal)
        normal_8bit = np.ascontiguousarray(output_img, dtype=np.uint8)
        # normal_8bit = cv.normalize(output_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        normal_8bit[mask] = 0

    else:
        output = output.astype(np.uint8)
        output = mu.filter_noise(output, threshold=[0, 255])
        output[mask] = 0
        normal_8bit = np.ascontiguousarray(output, dtype=np.uint8)
        # normal_8bit = cv.normalize(output, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        normal = mu.rgb2normal(normal_8bit)
    # normal_cnn_8bit = mu.normal2RGB(xout_normal)
    # mu.addText(normal_8bit, "output")

    return normal, normal_8bit, eval_point_counter, gpu_time


if __name__ == '__main__':
    pass
