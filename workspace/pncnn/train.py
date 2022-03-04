#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
@modified: J. Sha

"""
import os, sys
from os.path import dirname

sys.path.append(dirname(__file__))

import time
import torch

from workspace.model import NeuralNetworkModel
from workspace.pncnn import network
from pncnn.utils.error_metrics import create_error_metric, AverageMeter
from pncnn.utils import save_output_images


xout_channel = 3
cout_in_channel = 3
cout_out_channel = 6
cin_channel = 6


############ TRAINING FUNCTION ############
def train_epoch(nn_model, epoch):
    """
    Training function

    Args:
        nn_model: icl. The dataloader object for the dataset,optimizer:
        The optimizer to be used; objective: The objective function, model: The model to be trained
        epoch: What epoch to start from

    Returns:
        AverageMeter() object.

    Raises:
        KeyError: Raises an exception.
    """

    print('\n==> Training Epoch [{}] (lr={})'.format(epoch, nn_model.optimizer.param_groups[0]['lr']))

    err = create_error_metric(nn_model.args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    nn_model.model.train()  # switch to train mode

    start = time.time()
    for i, (input, target) in enumerate(nn_model.train_loader):

        # put input and target to device
        input, target = input.to(nn_model.device), target.to(nn_model.device)

        # Wait for all kernels to finish
        torch.cuda.synchronize()

        # record data load time
        data_time = time.time() - start

        # start count the model time
        start = time.time()

        # Clear the gradients
        nn_model.optimizer.zero_grad()

        # Forward pass
        out = nn_model.model(input)

        # Compute the loss
        loss = nn_model.loss(out, target)

        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # record model time
        gpu_time = time.time() - start

        # Calculate Error metrics
        err = create_error_metric(nn_model.args)
        err.evaluate(out[:, :xout_channel, :, :].data, target.data)
        err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))
        if ((i + 1) % nn_model.args.print_freq == 0) or (i == len(nn_model.train_loader) - 1):
            print(f"train_loader: {len(nn_model.train_loader)}")
            print(f"[Train] Epoch ({epoch}) [{i + 1}/{len(nn_model.train_loader)}]: ", end='')
            print(err_avg)

        # Log to Tensorboard if enabled
        if nn_model.tb_writer is not None:
            if (i + 1) % nn_model.tb_freq == 0:
                avg_meter = err_avg.get_avg()
                nn_model.tb_writer.add_scalar('Loss/train', avg_meter.loss,
                                              epoch * len(nn_model.train_loader) + i)
                nn_model.tb_writer.add_scalar('MAE/train', avg_meter.metrics['mae'],
                                              epoch * len(nn_model.train_loader) + i)
                nn_model.tb_writer.add_scalar('RMSE/train', avg_meter.metrics['rmse'],
                                              epoch * len(nn_model.train_loader) + i)

        # Start counting again for the next iteration
        start = time.time()

    # update log
    nn_model.train_csv.update_log(err_avg, epoch)

    return err_avg


# -------------- EVALUATION FUNCTION ----------------------
def evaluate_epoch(nn_model, epoch):
    """
    Evluation function

    Args:
        model_param: all the parameter of the model
        epoch: What epoch to start from

    Returns:
        AverageMeter() object.

    Raises:
        KeyError: Raises an exception.
    """
    print('\n==> Evaluating Epoch [{}]'.format(epoch))

    err = create_error_metric(nn_model.args)
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    nn_model.model.eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = save_output_images.create_out_image_saver(nn_model.exp_dir, nn_model.args, epoch)
    out_image = None

    start = time.time()
    with torch.no_grad():  # Disable gradients computations
        for i, (input, target) in enumerate(nn_model.val_loader):
            input, target = input.to(nn_model.device), target.to(nn_model.device)

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out = nn_model.model(input)

            # Check if there is cout There is Cout
            loss = nn_model.loss(out, target)  # Compute the loss

            gpu_time = time.time() - start

            # Calculate Error metrics
            err = create_error_metric(nn_model.args)
            err.evaluate(out[:, :xout_channel, :, :].data, target.data)
            # target.data = target.data.permute(0, 4, 2, 3, 1).sum(dim=-1)
            err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

            # Save output images
            if nn_model.args.save_val_imgs:
                out_image = out_img_saver.update(i, out_image, input, out, target)

            if nn_model.args.evaluate is None:
                if nn_model.tb_writer is not None and i == 1:  # Retrun batch 1 for tensorboard logging
                    out_image = out

            if (i + 1) % nn_model.args.print_freq == 0 or i == len(nn_model.val_loader) - 1:
                print('[Eval] Epoch: ({0}) [{1}/{2}]: '.format(
                    epoch, i + 1, len(nn_model.val_loader)), end='')
                print(err_avg)

            start = time.time()

    # Evaluate Uncerainty
    ause, ause_fig = nn_model.evaluate_uncertainty(epoch)

    # Update Log files
    nn_model.test_csv.update_log(err_avg, epoch, ause)

    return err_avg, out_image, ause, ause_fig


def main(args, network):
    nn_model = NeuralNetworkModel(args, network)

    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        train_epoch(nn_model, epoch)

        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        nn_model.save_checkpoint(False, epoch)

        # Evaluate the trained epoch
        test_err_avg, out_image, ause, ause_fig = evaluate_epoch(nn_model, epoch)  # evaluate on validation set

        # Log to tensorboard if enabled
        nn_model.log_to_tensorboard(test_err_avg, ause, ause_fig, out_image, epoch)
        # save the best model
        nn_model.save_best_model(test_err_avg, epoch)


if __name__ == '__main__':
    pass
