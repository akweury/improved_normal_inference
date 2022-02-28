import torch
import shutil
import os


def save_checkpoint(model_param, is_best, epoch):
    checkpoint_filename = os.path.join(model_param['exp_dir'], 'checkpoint-' + str(epoch) + '.pth.tar')

    state = {'args': model_param['args'],
             'epoch': epoch,
             'model': model_param['model'],
             'best_result': model_param['best_result'],
             'optimizer': model_param['optimizer']}

    torch.save(state, checkpoint_filename)

    if is_best:
        best_filename = os.path.join(model_param['exp_dir'], 'model_best.pth.tar')
        shutil.copyfile(checkpoint_filename, best_filename)

    if epoch > 0:
        prev_checkpoint_filename = os.path.join(model_param['exp_dir'], 'checkpoint-' + str(epoch-1) + '.pth.tar')
        if os.path.exists(prev_checkpoint_filename):
            os.remove(prev_checkpoint_filename)