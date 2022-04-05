# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import torch
import numpy as np

import config
from pncnn.utils import args_parser
from workspace.nnnx import candidate_normals
from help_funs import mu


def eval(vertex):
    print('\n==> Evaluation mode!')
    check_point_id = 999

    # Define paths
    exp_path = config.ws_path / "nnnx"

    # load args
    json_file = str(exp_path / "args.json")
    if os.path.isfile(json_file):
        with open(json_file, 'r') as fp:
            args = json.load(fp)

    # load model
    chkpt_path = exp_path / f"checkpoint-{check_point_id}.pth.tar"
    assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)
    print('- Loading checkpoint:', chkpt_path)
    if args['cpu']:
        checkpoint = torch.load(chkpt_path, map_location="cpu")
    else:
        checkpoint = torch.load(chkpt_path)

    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print('- Checkpoint was loaded successfully.')

    # Compare the checkpoint args with the json file in case I wanted to change some args
    # args_parser.compare_args_w_json(args, exp_dir, start_epoch + 1)
    args.evaluate = chkpt_path

    # load model
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    model = checkpoint['model'].to(device)

    args_parser.print_args(args)

    normal_img, eval_point_counter, total_time = evaluate_epoch(model, vertex, start_epoch, device)

    return normal_img, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, vertex, epoch, device):
    print('\n==> Evaluating Epoch [{}]'.format(epoch))

    h, w, c = vertex.shape
    mask = vertex.sum(axis=2) == 0
    normal_img = np.zeros(shape=vertex.shape)

    model.eval()  # Swith to evaluate mode
    eval_point_counter = 0
    total_time = 0.0
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                eval_point_counter += 1
                with torch.no_grad():
                    # start counting time
                    start = time.time()
                    # candidate normals are the input of the model
                    input = candidate_normals.generate_candidates(vertex, i, j).astype(np.float32)
                    input_tensor = torch.from_numpy(input)
                    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                    input_tensor = input_tensor.to(device)

                    torch.cuda.synchronize()
                    data_time = time.time() - start

                    # Forward Pass
                    start = time.time()
                    normal_svdn = model(input_tensor)
                    gpu_time = time.time() - start

                    # store the predicted normal
                    normal_svdn = normal_svdn.to('cpu').numpy().reshape(3)
                    normal_img[i, j] = mu.normal2RGB_single(normal_svdn).reshape(3)
                total_time += gpu_time

    avg_time = total_time / eval_point_counter
    normal_img = normal_img.astype(np.uint8)
    return normal_img, eval_point_counter, total_time


if __name__ == '__main__':
    pass
