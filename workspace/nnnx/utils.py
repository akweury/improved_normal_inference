import os, sys
from os.path import dirname

sys.path.append(dirname(__file__))

import json
import glob
import numpy as np
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD, Adam, lr_scheduler

import config
from help_funs import file_io, mu
from workspace.nnnx import network, candidate_normals


class L2Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, target, *args):
        return F.mse_loss(outputs, target)


class NeuralNetworkModel():
    def __init__(self, args, start_epoch=0):
        self.args = args
        self.start_epoch = start_epoch
        self.device = torch.device("cpu" if self.args.cpu else f"cuda:{self.args.gpu}")
        self.exp_dir = config.ws_path / self.args.exp
        self.train_loader, self.val_loader = create_dataloader(self.args, eval_mode=False)
        self.model = network.CNN().to(self.device)
        self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = self.init_optimizer()
        self.loss = self.init_loss()
        self.losses = []
        self.criterion = nn.CrossEntropyLoss()
        self.init_lr_decayer()
        self.print_info(args)

    def init_loss(self):
        self.loss = L2Loss()
        self.loss.to(self.device)
        return self.loss

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

    def save_checkpoint(self, is_best, epoch):
        checkpoint_filename = os.path.join(self.exp_dir, 'checkpoint-' + str(epoch) + '.pth.tar')

        state = {'args': self.args,
                 'epoch': epoch,
                 'model': self.model,
                 'optimizer': self.optimizer}

        torch.save(state, checkpoint_filename)

        if is_best:
            best_filename = os.path.join(self.exp_dir, 'model_best.pth.tar')
            shutil.copyfile(checkpoint_filename, best_filename)

        if epoch > 0:
            prev_checkpoint_filename = os.path.join(self.exp_dir, 'checkpoint-' + str(epoch - 1) + '.pth.tar')
            if os.path.exists(prev_checkpoint_filename):
                os.remove(prev_checkpoint_filename)


class SyntheticDataset(Dataset):

    def __init__(self, synthetic_depth_path, setname='train'):

        self.synthetic_depth_path = synthetic_depth_path
        self.setname = setname

        if setname in ['train', 'selval']:
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "train" / "*depth0.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "train" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "train" / "*data0.json"), recursive=True)))

        elif setname == 'selval':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "eval" / "*depth0.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "eval" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "eval" / "*data0.json"), recursive=True)))

        elif setname == 'test':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "test" / "*depth0.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "test" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "test" / "*data0.json"), recursive=True)))

        assert (len(self.gt) == len(self.depth))

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        f = open(self.data[item])
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(self.depth[item],
                                              data['minDepth'],
                                              data['maxDepth'])
        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())

        h, w = vertex.shape[:2]
        x, y = 0, 0
        while vertex[x, y, :].sum() == 0:
            x = np.random.randint(h)
            y = np.random.randint(w)

        # gt = file_io.load_24bitImage(self.gt[item]).astype(np.float32)
        gt = file_io.load_24bitNormal(self.gt[item]).astype(np.float32)
        gt = gt[x, y, :]
        normal_gt = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        # normal_gt = normal_gt / np.linalg.norm(normal_gt)

        normal_svd = candidate_normals.generate_candidates_random(vertex, x, y).astype(np.float32)
        # normal_svd = normal_svd_rgb / np.linalg.norm(normal_svd_rgb, ord=2, axis=2, keepdims=True)
        normal_svd = torch.from_numpy(normal_svd)  # (depth, dtype=torch.float)
        normal_svd = normal_svd.permute(2, 0, 1)

        # print(f"data: max, min:{vertex_input.max(), vertex_input.min()}")
        return normal_svd, normal_gt




def create_dataloader(args, eval_mode=False):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    ds_dir = args.dataset_path
    train_on = args.train_on
    val_set = args.val_ds

    train_loader = []
    val_loader = []

    ds_dir = eval(f"config.{ds_dir}")

    if eval_mode is not True:
        ###### Training Set ######
        train_dataset = SyntheticDataset(ds_dir, setname='train')
        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
            train_dataset.depth = train_dataset.depth[training_idxs]
            train_dataset.gt = train_dataset.gt[training_idxs]

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    # Validation set
    if val_set == 'selval':
        ###### Validation Set ######
        val_dataset = SyntheticDataset(ds_dir, setname='selval')

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'selval':
        ###### Selected Validation set ######
        val_dataset = SyntheticDataset(ds_dir, setname='selval')

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'test':
        ###### Test set ######
        val_dataset = SyntheticDataset(ds_dir, setname='test')

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader


def get_data_set_path(args):
    args.dataset_path = config.synthetic_data
