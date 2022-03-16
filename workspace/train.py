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

import cv2
import time
import torch
import numpy as np
from pprint import pprint
from workspace.svdn import utils
from help_funs import mu, chart

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

    folder_path = nn_model.exp_dir / f"output_{chart.date_now}_{chart.time_now}"
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    print('\n==> Training Epoch [{}] (lr={})'.format(epoch, nn_model.optimizer.param_groups[0]['lr']))
    nn_model.model.train()  # switch to train mode
    loss_total = 0.0
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
        loss_total += loss
        if i == 1:
            # print statistics
            np.set_printoptions(precision=3)

            input, out, target, = input.to("cpu"), out.to("cpu"), target.to("cpu")
            pprint(f'[epoch: {epoch}] loss: {nn_model.losses}')
            chart.draw_output_svd(input, out, target=target, exp_path=folder_path,
                                  loss=loss, epoch=epoch, i=i, prefix="train")

        start = time.time()

    loss_avg = loss_total / (nn_model.train_loader.__len__() * nn_model.args.batch_size)
    nn_model.losses.append(loss_avg.item())

    chart.line_chart(np.array([nn_model.losses]), folder_path, log_y=True)


def main(args):
    nn_model = utils.NeuralNetworkModel(args)

    ############ TRAINING LOOP ############
    for epoch in range(nn_model.start_epoch, nn_model.args.epochs):
        # Train one epoch
        train_epoch(nn_model, epoch)

        # Learning rate scheduler
        nn_model.lr_decayer.step()

        # Save checkpoint in case evaluation crashed
        nn_model.save_checkpoint(False, epoch)


if __name__ == '__main__':
    pass
