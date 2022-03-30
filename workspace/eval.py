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


def eval(vertex, model_path, k):
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

    normal_img, eval_point_counter, total_time = evaluate_epoch(model, input_tensor, start_epoch, device)

    normal = mu.rgb2normal(normal_img)

    return normal, normal_img, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, input_tensor, epoch, device):
    mask = input_tensor.sum(axis=2) == 0

    model.eval()  # Swith to evaluate mode
    eval_point_counter = torch.sum(mask)

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)
        torch.cuda.synchronize()

        # Forward Pass
        start = time.time()
        normal_img = model(input_tensor)
        gpu_time = time.time() - start

        # store the predicted normal
        normal_img = normal_img[0, :].permute(1, 2, 0)[:, :, :3]
        normal_img = normal_img.to('cpu').numpy().astype(np.uint8)

    normal_img = mu.filter_noise(normal_img, threshold=[0, 255])
    out_ranges = mu.addHist(normal_img)
    normal_8bit = cv.normalize(normal_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # normal_cnn_8bit = mu.normal2RGB(xout_normal)
    # mu.addText(normal_8bit, "output")
    mu.addText(normal_8bit, str(out_ranges), pos="upper_right", font_size=0.5)

    return normal_8bit, eval_point_counter, gpu_time


if __name__ == '__main__':
    pass
