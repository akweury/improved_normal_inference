#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
"""
import os
import sys
import importlib
import time
import datetime

import torch
from torch.optim import SGD, Adam
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from improved_normal_inference.pncnn.utils import args_parser
from improved_normal_inference.pncnn.dataloaders.my_creator import create_dataloader
# from dataloaders.dataloader_creator import create_dataloader

from improved_normal_inference.pncnn.utils.error_metrics import AverageMeter, create_error_metric
from improved_normal_inference.pncnn.utils.save_output_images import create_out_image_saver
from improved_normal_inference.pncnn.common.losses import get_loss_fn

from improved_normal_inference import config


def main(model_param):
    # Make some variable global
    # global args, train_csv, test_csv, exp_dir, best_result, device, tb_writer, tb_freq
    args = model_param['args']

    start_epoch = 0
    ############ EVALUATE MODE ############

    print('\n==> Evaluation mode!')

    # Define paths
    # synthetic paths
    chkpt_path = config.ws_path / args.exp / "checkpoint-154.pth.tar"
    # kitti paths
    # chkpt_path = config.exper_kitti / args.evaluate / "checkpoint-154.pth.tar"

    # Check that the checkpoint file exist
    assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

    # Experiment director
    exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
    sys.path.append(exp_dir)

    # Load checkpoint
    print('- Loading checkpoint:', chkpt_path)

    # Load the checkpoint
    if args.cpu:
        checkpoint = torch.load(chkpt_path, map_location="cpu")
    else:
        checkpoint = torch.load(chkpt_path)
    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    best_result = checkpoint['best_result']
    print('- Checkpoint was loaded successfully.')

    # Compare the checkpoint args with the json file in case I wanted to change some args
    # args_parser.compare_args_w_json(args, exp_dir, start_epoch + 1)
    args.evaluate = chkpt_path

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    model = checkpoint['model'].to(device)

    args_parser.print_args(args)

    _, val_loader = create_dataloader(args, eval_mode=True)

    loss = get_loss_fn(args).to(device)

    model_param['loss'] = loss

    model_param['model'] = model

    model_param['val_loader'] = val_loader

    evaluate_epoch(model_param, start_epoch)

    return  # End program


############ EVALUATION FUNCTION ############
def evaluate_epoch(model_param, epoch):
    """
    Evluation function

    Args:
        dataloader: The dataloader object for the dataset
        model: The model to be trained
        epoch: What epoch to start from

    Returns:
        AverageMeter() object.

    Raises:
        KeyError: Raises an exception.
    """
    print('\n==> Evaluating Epoch [{}]'.format(epoch))

    err = create_error_metric(model_param['args'])
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model_param['model'].eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = create_out_image_saver(model_param['exp_dir'], model_param['args'], epoch)
    out_image = None

    start = time.time()
    with torch.no_grad():  # Disable gradients computations
        for i, (input, target) in enumerate(model_param['val_loader']):
            input, target = input.to(model_param['device']), target.to(model_param['device'])

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out = model_param['model'](input)

            # Check if there is cout There is Cout
            loss = model_param['loss'](out, target)  # Compute the loss

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(model_param['args'])
            err.evaluate(out[:, :1, :, :].data, target.data)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

            # Save output images
            if model_param['args'].save_val_imgs:
                out_image = out_img_saver.update(i, out_image, input, out, target)

            if model_param['args'].evaluate is None:
                if model_param['tb_writer'] is not None and i == 1:  # Retrun batch 1 for tensorboard logging
                    out_image = out

            if (i + 1) % model_param['args'].print_freq == 0 or i == len(model_param['val_loader']) - 1:
                print('[Eval] Epoch: ({0}) [{1}/{2}]: '.format(
                    epoch, i + 1, len(model_param['val_loader'])), end='')
                print(err_avg)

            start = time.time()

    return err_avg, out_image


if __name__ == '__main__':
    main()
