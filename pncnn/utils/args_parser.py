#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:25:52 2019

@author: abdel62
"""
import argparse
import os
import json
from pprint import pprint
from pncnn.common.losses import get_loss_list
import config

# Lists for args which have mandatory selections
datasets_list = ['nyudepthv2', 'kitti_depth', 'synthetic', 'kitti_odo', 'flat']
modality_list = ['rgb', 'rgbd', 'd']
losses_list = get_loss_list()
optimizers_list = ['sgd', 'adam']
lr_scheduler_list = ['step', 'lambda', 'plateau']


def args_parser():
    """
    Parese command line arguments 
    
    Args:
    opt_args: Optional args for testing the function. By default sys.argv is used
    
    Returns:
        args: Dictionary of args.
    
    Raises:
        ValueError: Raises an exception if some arg values are invalid.
    """
    # Construct the parser
    parser = argparse.ArgumentParser(description='NConv')

    # Mode selection

    parser.add_argument('--neighbor', type=int, default=2, help='the neighbors will be considered')
    parser.add_argument('--machine', type=str, default="local", help='choose the training machin, local or remote')
    parser.add_argument('--num-channels', type=int, help='choose the number of channels in the model')
    parser.add_argument('--output_type', type=str, default="normal",
                        help='choose the meaning of output tensor, rgb or normal')

    parser.add_argument('--args', '-a', type=str, default='', choices=['defaults', 'json'],
                        help='How to read args? (json file or dataset defaults)')

    parser.add_argument('--exp', '--e', help='Experiment name')

    parser.add_argument('--workspace', '--ws', default='', type=str, help='Workspace name')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    ########### General Dataset arguments ##########

    parser.add_argument('--dataset-path', type=str, default='', help='Dataset path.')
    parser.add_argument('--batch_size', '-b', default=8, type=int, help='Mini-batch size (default: 8)')

    parser.add_argument('--train-on', default='full', type=str, help='The number of images to train on from the data.')

    parser.add_argument('--sharp_penalty', default=1.5, type=float, help='penalty of sharp normal prediction')
    ########### KITTI-Depth arguments ###########
    parser.add_argument('--raw-kitti-path', type=str, default='', help='Dataset path')

    parser.add_argument('--selval-ds', default='selval', type=str, choices=['selval, selval, test'],
                        help='Which set to evaluate on? ' + ' | '.join(['selval, selval, test']) + ' (default: selval)')

    parser.add_argument('--norm-factor', default=256, type=float,
                        help='Normalization factor for the input data (default: 256)')

    parser.add_argument('--train-disp', default=False, type=bool,
                        help='Train on disparity (1/depth) (default: False)')

    ########### Training arguments ###########
    parser.add_argument('--epochs', default=20, type=int,
                        help='Total number of epochs to run (default: 30)')

    parser.add_argument('--optimizer', '-o', default='adam', choices=optimizers_list,
                        help='Optimizer method: ' + ' | '.join(optimizers_list) + ' (default: sgd)')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='Initial learning rate (default 0.001)')

    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum.')

    parser.add_argument('--lr-scheduler', default='step', choices=lr_scheduler_list,
                        help='LR scheduler method: ' + ' | '.join(lr_scheduler_list) + ' (default: step)')

    parser.add_argument('--lr-decay-step', default=5, type=int,
                        help='Learning rate decay step (default: 20)')

    parser.add_argument('--lr-decay-factor', default=0.1, type=float,
                        help='Learning rate decay factor(default: 0.1)')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        help='Weight decay (default: 0)')
    parser.add_argument('--loss', '-l', default='l1', choices=losses_list,
                        help='Loss function: ' + ' | '.join(losses_list) + ' (default: l1)')

    parser.add_argument('--penalty', '-pena', default=1.2, help='penalty of output value which out of range [0-255]')
    ########### Logging ###########
    parser.add_argument('--print-freq', default=10, type=int,
                        help='Printing evaluation criterion frequency (default: 10)')

    parser.add_argument('--tb_log', default=False, type=bool,
                        help='Log to Tensorboard (default: False)')

    parser.add_argument('--tb_freq', default=1000, type=int,
                        help='Logging Frequence to Tensorboard (default: 1000)')

    parser.add_argument('--save-selval-imgs', default=False, type=bool,
                        help='A flag for saving validation images (default: False)')

    parser.add_argument('--eval_uncert', default=False, type=bool,
                        help='Evaluate uncertainty or not')
    parser.add_argument('--angle_loss', default=False, type=bool,
                        help='Calculate angle loss and plot')
    # Parse the arguments
    args = parser.parse_args()

    args = initialize_args(args)

    return args


def initialize_args(args):
    # Path to the workspace directory
    ws_path = config.ws_path
    args_path = ws_path / args.exp / 'args.json'
    load_args_from_file(args_path, args)
    print_args(args)
    return args


def save_args(exp_dir, args, file_name='args.json'):
    with open(os.path.join(exp_dir, file_name), 'w') as outfile:
        dic = {}
        for arg in vars(args):
            dic[arg] = getattr(args, arg)
        json.dump(dic, outfile, separators=(',\n', ': '))


# Load default args for each dataset
def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            if key not in ['workspace', 'exp', 'evaluate', 'resume', 'gpu']:  # Do not overwrite these keys
                setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))


def print_args(args):
    pprint(f'==> Experiment Args:  {args} ')


# This function compares the args saved in the checkpoint with the json file
def compare_args_w_json(chkpt_args, exp_dir, epoch):
    path_to_json = os.path.join(exp_dir, 'args.json')

    if os.path.isfile(path_to_json):
        with open(path_to_json, 'r') as fp:
            json_args = json.load(fp)

    old_args_saved = False
    for key, json_value in json_args.items():
        chkpt_value = getattr(chkpt_args, key)
        if chkpt_value != json_value:
            print('! Argument "{}" was changed from "{}" in the checkpoint to "{}" in the JSON file!'.format(key,
                                                                                                             chkpt_value,
                                                                                                             json_value))

            # Save the old args to another file for history
            # if not old_args_saved:
            #    f_name = 'args_epoch_' + str(epoch - 1) + '.json'
            #    save_args(exp_dir, chkpt_args, file_name=f_name)  # Save the original args
            #    print('==> Original args were saved to "{}".'.format(f_name))
            #    old_args_saved = True

            setattr(chkpt_args, key, json_value)

    print('')


if __name__ == '__main__':
    args_parser()
