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

from utils.args_parser import args_parser, save_args, print_args, initialize_args, compare_args_w_json
from dataloaders.my_creator import create_dataloader
# from dataloaders.dataloader_creator import create_dataloader

from utils.error_metrics import AverageMeter, create_error_metric, LogFile
from utils.save_output_images import create_out_image_saver, colored_depthmap_tensor
from utils.checkpoints import save_checkpoint
from common.losses import get_loss_fn
from utils.eval_uncertainty import eval_ause

from improved_normal_inference import config


def main():
    # Make some variable global
    global args, train_csv, test_csv, exp_dir, best_result, device, tb_writer, tb_freq

    # Args parser
    args = args_parser()

    start_epoch = 0
    ############ EVALUATE MODE ############
    if args.evaluate:  # Evaluate mode
        print('\n==> Evaluation mode!')

        # Define paths
        # synthetic paths
        # chkpt_path = config.exper_synthetic / args.evaluate / "checkpoint-9.pth.tar"
        # kitti paths
        chkpt_path = config.exper_kitti / args.evaluate / "checkpoint-9.pth.tar"

        # Check that the checkpoint file exist
        assert os.path.isfile(chkpt_path), "- No checkpoint found at: {}".format(chkpt_path)

        # Experiment director
        exp_dir = os.path.dirname(os.path.abspath(chkpt_path))
        sys.path.append(exp_dir)

        # Load checkpoint
        print('- Loading checkpoint:', chkpt_path)

        # Load the checkpoint
        checkpoint = torch.load(chkpt_path, map_location="cpu")

        # Assign some local variables
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
        print('- Checkpoint was loaded successfully.')

        # Compare the checkpoint args with the json file in case I wanted to change some args
        compare_args_w_json(args, exp_dir, start_epoch + 1)
        args.evaluate = chkpt_path

        # device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        model = checkpoint['model'].to(device)

        print_args(args)

        _, val_loader = create_dataloader(args, eval_mode=True)

        loss = get_loss_fn(args).to(device)

        evaluate_epoch(val_loader, model, loss, start_epoch)

        return  # End program


############ EVALUATION FUNCTION ############
def evaluate_epoch(dataloader, model, objective, epoch):
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

    err = create_error_metric(args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model.eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = create_out_image_saver(exp_dir, args, epoch)
    out_image = None

    start = time.time()
    with torch.no_grad():  # Disable gradients computations
        for i, (input, target) in enumerate(dataloader):
            input, target = input.to(device), target.to(device)

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out = model(input)

            # Check if there is cout There is Cout
            loss = objective(out, target)  # Compute the loss

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(args)
            err.evaluate(out[:, :1, :, :].data, target.data)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

            # Save output images
            if args.save_val_imgs:
                out_image = out_img_saver.update(i, out_image, input, out, target)

            if args.evaluate is None:
                if tb_writer is not None and i == 1:  # Retrun batch 1 for tensorboard logging
                    out_image = out

            if (i + 1) % args.print_freq == 0 or i == len(dataloader) - 1:
                print('[Eval] Epoch: ({0}) [{1}/{2}]: '.format(
                    epoch, i + 1, len(dataloader)), end='')
                print(err_avg)

            start = time.time()

    return err_avg, out_image


if __name__ == '__main__':
    main()
