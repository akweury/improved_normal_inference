import datetime
import os, sys
from os.path import dirname

sys.path.append(dirname(__file__))

import shutil
import torch
import torch.nn as nn
from torch.optim import SGD, Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import config

from pncnn.common import losses
from pncnn.dataloaders import my_creator
from pncnn.utils.save_output_images import colored_normal_tensor
from pncnn.utils import eval_uncertainty
from pncnn.utils import error_metrics
from pncnn.utils import args_parser


class NeuralNetworkModel():
    def __init__(self, args, network, start_epoch=0):
        self.args = args
        self.start_epoch = start_epoch
        self.device = torch.device("cpu" if self.args.cpu else f"cuda:{self.args.gpu}")
        self.exp_dir = config.ws_path / self.args.exp
        self.train_loader, self.val_loader = my_creator.create_dataloader(self.args, eval_mode=False)
        self.model = network.CNN().to(self.device)
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.init_optimizer()
        self.best_result = error_metrics.create_error_metric(self.args)
        self.best_result.set_to_worst()
        self.tb = args.tb_log if hasattr(args, 'tb_log') else False
        self.tb_freq = args.tb_freq if hasattr(args, 'tb_freq') else 1000
        self.tb_writer = SummaryWriter(os.path.join(self.exp_dir, 'tb_log',
                                                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        self.loss = losses.get_loss_fn(args)
        self.losses = []
        self.criterion = nn.CrossEntropyLoss()

        # self.lr_decayer = lr_scheduler.StepLR(self.optimizer,
        #                                       step_size=args.lr_decay_step,
        #                                       gamma=args.lr_decay_factor,
        #                                       last_epoch=start_epoch - 1)
        self.init_lr_decayer()
        self.train_csv = error_metrics.LogFile(os.path.join(self.exp_dir, 'train.csv'), args)
        self.test_csv = error_metrics.LogFile(os.path.join(self.exp_dir, 'test.csv'), args)
        self.best_txt = os.path.join(self.exp_dir, 'best.txt')

        # args_parser.save_args(self.exp_dir, args)  # Save args to JSON file
        self.print_info(args)

    def init_lr_decayer(self):
        milestones = [int(x) for x in self.args.lr_scheduler.split(",")]
        self.lr_decayer = lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones,
                                                   gamma=self.args.lr_decay_factor)

    def init_optimizer(self):
        if self.args.optimizer.lower() == 'sgd':
            optimizer = SGD(self.parameters,
                            lr=self.args.lr,
                            momentum=self.args.momentum,
                            weight_decay=self.args.weight_decay)
        elif self.args.optimizer.lower() == 'adam':
            optimizer = Adam(self.parameters,
                             lr=self.args.lr,
                             weight_decay=self.args.weight_decay,
                             amsgrad=True)
        else:
            raise ValueError
        return optimizer

    def print_info(self, args):
        print(f'\n==> Starting a new experiment "{args.exp}" \n')
        print(f'\n==> Model "{self.model.__name__}" was loaded successfully!')
        args_parser.print_args(args)

    def evaluate_uncertainty(self, epoch):
        if self.args.eval_uncert:
            if self.args.loss == 'masked_prob_loss_var':
                ause, ause_fig = eval_uncertainty.eval_ause(self.model, self.val_loader, self.args, epoch,
                                                            uncert_type='v')
            else:
                ause, ause_fig = eval_uncertainty.eval_ause(self.model, self.val_loader, self.args, epoch,
                                                            uncert_type='c')
        else:
            raise ValueError

        return ause, ause_fig

    def log_to_tensorboard(self, test_err_avg, ause, ause_fig, out_image, epoch):
        if self.tb_writer is not None:
            avg_meter = test_err_avg.get_avg()
            self.tb_writer.add_scalar('Loss/selval', avg_meter.loss, epoch)
            self.tb_writer.add_scalar('MAE/selval', avg_meter.metrics['mae'], epoch)
            self.tb_writer.add_scalar('RMSE/selval', avg_meter.metrics['rmse'], epoch)
            if ause is not None:
                self.tb_writer.add_scalar('AUSE/selval', ause, epoch)
            # TODO: change colored depthmap tensor function to a correct one.
            self.tb_writer.add_images('Prediction', colored_normal_tensor(out_image[:, :config.xout_channel, :, :]),
                                      epoch)
            self.tb_writer.add_images('Input_Conf_Log_Scale',
                                      colored_normal_tensor(torch.log(out_image[:, config.cin_channel:, :, :] + 1)),
                                      epoch)
            self.tb_writer.add_images('Output_Conf_Log_Scale',
                                      colored_normal_tensor(
                                          torch.log(
                                              out_image[:, config.cout_in_channel:config.cout_out_channel, :, :] + 1)),
                                      epoch)
            self.tb_writer.add_figure('Sparsification_Plot', ause_fig, epoch)

    def save_best_model(self, test_err_avg, epoch):
        # Save best model
        is_best = test_err_avg.metrics['rmse'] < self.best_result.metrics['rmse']
        if is_best:
            self.best_result = test_err_avg  # Save the new best locally
            test_err_avg.print_to_txt(self.best_txt, epoch)  # Print to a text file
        else:
            self.best_result = self.best_result

        # Save it again if it is best checkpoint
        self.save_checkpoint(is_best, epoch)

    def save_checkpoint(self, is_best, epoch):
        checkpoint_filename = os.path.join(self.exp_dir, 'checkpoint-' + str(epoch) + '.pth.tar')

        state = {'args': self.args,
                 'epoch': epoch,
                 'model': self.model,
                 'best_result': self.best_result,
                 'optimizer': self.optimizer}

        torch.save(state, checkpoint_filename)

        if is_best:
            best_filename = os.path.join(self.exp_dir, 'model_best.pth.tar')
            shutil.copyfile(checkpoint_filename, best_filename)

        if epoch > 0:
            prev_checkpoint_filename = os.path.join(self.exp_dir, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
            if os.path.exists(prev_checkpoint_filename):
                os.remove(prev_checkpoint_filename)
