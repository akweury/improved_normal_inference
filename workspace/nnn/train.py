#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
@modified: J. Sha

"""
import sys
from os.path import dirname

sys.path.append(dirname(__file__))

import cv2
import time
import torch
import torch.nn as nn
import numpy as np

from help_funs import mu, chart
from common.model import NeuralNetworkModel
from pncnn.utils.error_metrics import create_error_metric, AverageMeter

xout_channel = 3
cin_channel = 6


def save_output(out, target, output_1, nn_model, epoch, i, prefix, loss):
    # ------------------ visualize outputs ----------------------------------------------

    # normalize output normal
    normal_cnn_8bit = mu.tenor2numpy(out[:1, :, :, :]).astype(np.uint8)
    mu.addText(normal_cnn_8bit, "output")

    normal_gt_8bit = mu.tenor2numpy(target[:1, :, :, :]).astype(np.uint8)
    mu.addText(normal_gt_8bit, "normal(gt)")

    output_2 = cv2.hconcat([normal_cnn_8bit, normal_gt_8bit])
    output = cv2.vconcat([output_1, output_2])
    output = cv2.resize(output, (512, 512))
    cv2.imwrite(str(nn_model.exp_dir / "output" / f"{prefix}_NNN_epoch_{epoch}_{i}_loss_{loss}.png"), output)

    np_array1, np_array2 = mu.tenor2numpy(out[:1, :, :, :]), mu.tenor2numpy(target[:1, :, :, :])
    b1, g1, r1 = cv2.split(np_array1)
    b2, g2, r2 = cv2.split(np_array2)
    output_3 = mu.concat_vh([[b1, g1, r1], [b2, g2, r2]])

    # cv2.imwrite(str(nn_model.exp_dir / "output" / f"{prefix}_NNN_epoch_{epoch}_{i}_loss_{loss}_tensor.png"), output_3)

    # mu.show_images(output_3, f"tensor_different")
    # mu.show_images(output, f"{prefix}_NNN_epoch_{epoch}_{i}.png")


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
    #
    # err = create_error_metric(nn_model.args)
    # err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    nn_model.model.train()  # switch to train mode
    start = time.time()
    loss_total = 0.0
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
        out, output_1 = nn_model.model(input)

        # Compute the loss
        if epoch < nn_model.args.loss_milestone:
            loss_fn = nn_model.loss[0]
        else:
            loss_fn = nn_model.loss[1]

        loss_fn.to(nn_model.device)
        loss = loss_fn(out, target)

        # loss = nn_model.criterion(out, target)

        # Backward pass
        loss.backward()

        # Update the parameters
        nn_model.optimizer.step()

        # record model time
        gpu_time = time.time() - start

        # print statistics
        print(f'[epoch: {epoch}, idx:{i:5d}] loss: {loss:.3f}')

        # save output
        loss_total += loss
        if i == 4:
            save_output(out, target, output_1, nn_model, epoch, i, "train", f"{loss:.3f}")

        # Start counting again for the next iteration
        start = time.time()

    # # update log
    # nn_model.train_csv.update_log(err_avg, epoch)
    loss_avg = loss_total / nn_model.train_loader.__len__()
    nn_model.losses.append(loss_avg.item())

    chart.line_chart(np.array([nn_model.losses]), nn_model.exp_dir / "output")
    return loss_total

    # return err_avg


# -------------- EVALUATION FUNCTION ----------------------
def evaluate_epoch(nn_model, epoch):
    """
    Evluation function

    Args:
        nn_model: all the parameter of the model
        epoch: What epoch to start from

    Returns:
        AverageMeter() object.

    Raises:
        KeyError: Raises an exception.
    """
    print('\n==> Evaluating Epoch [{}]'.format(epoch))

    # err = create_error_metric(nn_model.args)
    # err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    nn_model.model.eval()  # Swith to evaluate mode

    start = time.time()
    with torch.no_grad():  # Disable gradients computations
        for i, (input, target) in enumerate(nn_model.val_loader):
            input, target = input.to(nn_model.device), target.to(nn_model.device)

            torch.cuda.synchronize()

            data_time = time.time() - start

            # Forward Pass
            start = time.time()

            out, output_1 = nn_model.model(input)

            # Check if there is cout There is Cout
            # loss = nn_model.criterion(out, target)  # Compute the loss
            loss = nn_model.loss(out, target)

            gpu_time = time.time() - start
            print(f'eval: [{epoch + 1}, {i + 1:5d}] loss: {loss:.3f}')
            # ------------------ visualize outputs ----------------------------------------------
            if i == 4:
                save_output(out, target, output_1, nn_model, epoch, i, "eval", f"{loss:.3f}")
            # ------------------------------------------------------------------------------------

            start = time.time()

    # Evaluate Uncerainty
    # ause, ause_fig = nn_model.evaluate_uncertainty(epoch)

    # Update Log files
    # nn_model.test_csv.update_log(err_avg, epoch, ause)
    #
    # return err_avg


def main(args, network):
    nn_model = NeuralNetworkModel(args, network)
    losses = []
    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        train_epoch(nn_model, epoch)

        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        nn_model.save_checkpoint(False, epoch)

        # Evaluate the trained epoch
        # evaluate_epoch(nn_model, epoch)  # evaluate on validation set

        # Log to tensorboard if enabled
        # nn_model.log_to_tensorboard(test_err_avg, out_image, epoch)
        # save the best model
        # nn_model.save_best_model(test_err_avg, epoch)


if __name__ == '__main__':
    pass
