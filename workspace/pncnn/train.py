#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:16:29 2019

@author: abdel62
@modified: J. Sha

"""
import time
import torch

from improved_normal_inference.workspace.pncnn import network
from improved_normal_inference.pncnn.utils.error_metrics import create_error_metric, AverageMeter
from improved_normal_inference.workspace import model
from improved_normal_inference.pncnn.utils import save_output_images
from improved_normal_inference.pncnn.utils import checkpoints


############ TRAINING FUNCTION ############
def train_epoch(model_param, epoch):
    """
    Training function

    Args:
        model_param: icl. The dataloader object for the dataset,optimizer:
        The optimizer to be used; objective: The objective function, model: The model to be trained
        epoch: What epoch to start from

    Returns:
        AverageMeter() object.

    Raises:
        KeyError: Raises an exception.
    """

    print('\n==> Training Epoch [{}] (lr={})'.format(epoch, model_param['optimizer'].param_groups[0]['lr']))

    err = create_error_metric(model_param['args'])
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model_param['model'].train()  # switch to train mode

    start = time.time()
    for i, (input, target) in enumerate(model_param['train_loader']):
        input, target = input.to(model_param['device']), target.to(model_param['device'])

        # torch.cuda.synchronize()  # Wait for all kernels to finish

        data_time = time.time() - start

        start = time.time()

        model_param['optimizer'].zero_grad()  # Clear the gradients

        # Forward pass
        out = model_param['model'](input)

        loss = model_param['loss'](out, target)  # Compute the loss

        # Backward pass
        loss.backward()

        model_param['optimizer'].step()  # Update the parameters

        gpu_time = time.time() - start

        # Calculate Error metrics
        err = create_error_metric(model_param['args'])

        err.evaluate(out[:, :1, :, :].data, target.data)
        # target.data = target.data.permute(0, 4, 2, 3, 1).sum(dim=-1)
        # err.evaluate(out[:, :3, :, :].data, target.data)
        err_avg.update(err.get_results(), loss.item(), gpu_time, data_time, input.size(0))

        if ((i + 1) % model_param['args'].print_freq == 0) or (i == len(model_param['train_loader']) - 1):
            print(f"train_loader: {len(model_param['train_loader'])}")
            print(f"[Train] Epoch ({epoch}) [{i + 1}/{len(model_param['train_loader'])}]: ", end='')
            print(err_avg)

        # Log to Tensorboard if enabled
        if model_param['tb_writer'] is not None:
            if (i + 1) % model_param['tb_freq'] == 0:
                avg_meter = err_avg.get_avg()
                model_param['tb_writer'].add_scalar('Loss/train', avg_meter.loss,
                                                    epoch * len(model_param['train_loader']) + i)
                model_param['tb_writer'].add_scalar('MAE/train', avg_meter.metrics['mae'],
                                                    epoch * len(model_param['train_loader']) + i)
                model_param['tb_writer'].add_scalar('RMSE/train', avg_meter.metrics['rmse'],
                                                    epoch * len(model_param['train_loader']) + i)

        start = time.time()  # Start counting again for the next iteration

    # update log
    model_param['train_csv'].update_log(err_avg, epoch)

    return err_avg


# -------------- EVALUATION FUNCTION ----------------------
def evaluate_epoch(model_param, epoch):
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

    err = create_error_metric(model_param['args'])
    err_avg = AverageMeter(err.get_metrics())  # Accumulator for the error metrics

    model_param['model'].eval()  # Swith to evaluate mode

    # Save output images
    out_img_saver = save_output_images.create_out_image_saver(model_param['exp_dir'], model_param['args'], epoch)
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
            # target.data = target.data.permute(0, 4, 2, 3, 1).sum(dim=-1)
            # err.evaluate(out[:, :3, :, :].data, target.data)
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

    # Evaluate Uncerainty
    ause, ause_fig = model.evaluate_uncertainty(model_param['args'],
                                                model_param['model'], model_param['val_loader'], epoch)

    # Update Log files
    model_param['test_csv'].update_log(err_avg, epoch, ause)

    return err_avg, out_image, ause, ause_fig


def main():
    model_param = model.init_env(network)

    ############ TRAINING LOOP ############
    for epoch in range(model_param['start_epoch'], model_param['args'].epochs):
        # Train one epoch
        train_epoch(model_param, epoch)

        # Learning rate scheduler
        model_param['lr_decayer'].step()

        # Save checkpoint in case evaluation crashed
        checkpoints.save_checkpoint(model_param, False, epoch)

        # Evaluate the trained epoch
        test_err_avg, out_image, ause, ause_fig = evaluate_epoch(model_param, epoch)  # evaluate on validation set

        # Log to tensorboard if enabled
        model.log_to_tensorboard(model_param['tb_writer'], test_err_avg, ause, ause_fig, out_image, epoch)

        # save the best model
        model.save_best_model(model_param, test_err_avg, epoch)


if __name__ == '__main__':
    main()
