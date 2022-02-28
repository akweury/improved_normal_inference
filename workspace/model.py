import datetime
import importlib
import os, sys
from os.path import dirname

sys.path.append(dirname(__file__))

import torch
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import config
from pncnn.utils import args_parser
from pncnn.utils.save_output_images import colored_depthmap_tensor
from pncnn.dataloaders import my_creator
from pncnn.utils import checkpoints
from pncnn.utils import eval_uncertainty
from pncnn.utils import error_metrics
from pncnn.common import losses
from pncnn.utils import args_parser


def save_best_model(model_param, test_err_avg, epoch):
    # Save best model
    # TODO: How to decide the best based on dataset?
    is_best = test_err_avg.metrics['rmse'] < model_param['best_result'].metrics['rmse']
    if is_best:
        best_result = test_err_avg  # Save the new best locally
        test_err_avg.print_to_txt(model_param['best_txt'], epoch)  # Print to a text file
    else:
        best_result = model_param['best_result']

    # Save it again if it is best checkpoint
    checkpoints.save_checkpoint(model_param, is_best, epoch)


def log_to_tensorboard(tb_writer, test_err_avg, ause, ause_fig, out_image, epoch):
    if tb_writer is not None:
        avg_meter = test_err_avg.get_avg()
        tb_writer.add_scalar('Loss/selval', avg_meter.loss, epoch)
        tb_writer.add_scalar('MAE/selval', avg_meter.metrics['mae'], epoch)
        tb_writer.add_scalar('RMSE/selval', avg_meter.metrics['rmse'], epoch)
        if ause is not None:
            tb_writer.add_scalar('AUSE/selval', ause, epoch)
        # TODO: change colored depthmap tensor function to a correct one.
        tb_writer.add_images('Prediction', colored_depthmap_tensor(out_image[:, :1, :, :]), epoch)
        tb_writer.add_images('Input_Conf_Log_Scale', colored_depthmap_tensor(torch.log(out_image[:, 2:, :, :] + 1)),
                             epoch)
        tb_writer.add_images('Output_Conf_Log_Scale',
                             colored_depthmap_tensor(torch.log(out_image[:, 1:2, :, :] + 1)), epoch)
        tb_writer.add_figure('Sparsification_Plot', ause_fig, epoch)


def evaluate_uncertainty(args, model, val_loader, epoch):
    if args.eval_uncert:
        if args.loss == 'masked_prob_loss_var':
            ause, ause_fig = eval_uncertainty.eval_ause(model, val_loader, args, epoch, uncert_type='v')
        else:
            ause, ause_fig = eval_uncertainty.eval_ause(model, val_loader, args, epoch, uncert_type='c')
    else:
        raise ValueError

    return ause, ause_fig


def init_env(network):
    # Make some variable global
    model_param = {}

    # Args parser
    args = args_parser.args_parser()
    model_param['args'] = args

    if args.cpu:
        device = "cpu"
    else:
        device = args.gpu

    start_epoch = 0
    model_param['start_epoch'] = start_epoch

    print('\n==> Starting a new experiment "{}" \n'.format(args.exp))

    # Which device to use
    device = torch.device(device)
    model_param['device'] = device

    args_parser.print_args(args)
    exp_dir = config.ws_path / args.exp
    model_param['exp_dir'] = exp_dir

    # Create dataloader
    train_loader, val_loader = my_creator.create_dataloader(args, eval_mode=False)
    model_param['train_loader'] = train_loader
    model_param['val_loader'] = val_loader

    # import the model
    model = network.CNN().to(device)
    model_param['model'] = model
    print('\n==> Model "{}" was loaded successfully!'.format(model.__name__))

    # Optimize only parameters that requires_grad
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_param['parameters'] = parameters

    # Create Optimizer
    if args.optimizer.lower() == 'sgd':
        optimizer = SGD(parameters, lr=args.lr, momentum=args.momentum,
                        weight_decay=args.weight_decay)
    elif args.optimizer.lower() == 'adam':
        optimizer = Adam(parameters, lr=args.lr, weight_decay=args.weight_decay, amsgrad=True)
    else:
        raise ValueError
    model_param['optimizer'] = optimizer

    ############ IF RESUME/NEW EXP ############
    # Error metrics that are set to the worst
    best_result = error_metrics.create_error_metric(args)
    best_result.set_to_worst()
    model_param['best_result'] = best_result

    # Tensorboard
    tb = args.tb_log if hasattr(args, 'tb_log') else False
    tb_freq = args.tb_freq if hasattr(args, 'tb_freq') else 1000

    if tb:
        tb_writer = SummaryWriter(
            os.path.join(exp_dir, 'tb_log', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
    else:
        raise ValueError

    model_param['tb'] = tb
    model_param['tb_freq'] = tb_freq
    model_param['tb_writer'] = tb_writer

    # Create Loss
    loss = losses.get_loss_fn(args).to(device)
    model_param['loss'] = loss

    # Define Learning rate decay
    lr_decayer = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_factor,
                                     last_epoch=start_epoch - 1)
    model_param['lr_decayer'] = lr_decayer

    # Create or Open Logging files
    train_csv = error_metrics.LogFile(os.path.join(exp_dir, 'train.csv'), args)
    test_csv = error_metrics.LogFile(os.path.join(exp_dir, 'test.csv'), args)
    best_txt = os.path.join(exp_dir, 'best.txt')

    model_param['train_csv'] = train_csv
    model_param['test_csv'] = test_csv
    model_param['best_txt'] = best_txt

    args_parser.save_args(exp_dir, args)  # Save args to JSON file

    return model_param
