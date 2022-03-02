########################################
__author__ = "Jingyuan Sha"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Jingyuan Sha"
__email__ = "jingyuan@rhrk.uni-kl.de"

########################################

from torch.utils.data import Dataset
import torch
import numpy as np
import glob
import os
import json
from torchvision import transforms

import config
from help_funs import file_io


class SyntheticDepthDataset(Dataset):

    def __init__(self, synthetic_depth_path,
                 setname='train', transform=None, invert_depth=False, load_rgb=False,
                 rgb2gray=False, hflip=False):

        self.synthetic_depth_path = synthetic_depth_path
        self.setname = setname
        self.transform = transform
        self.invert_depth = invert_depth
        self.load_rgb = load_rgb
        self.rgb2gray = rgb2gray
        self.hflip = hflip

        if setname in ['train', 'selval']:
            depth_path = self.synthetic_depth_path
            self.depth = np.array(
                sorted(glob.glob(str(depth_path / "train" / "*depth0_noise.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "train" / "*depth0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "train" / "*data0.json"), recursive=True)))

        elif setname == 'selval':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "eval" / "*depth0_noise.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "eval" / "*depth0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "eval" / "*data0.json"), recursive=True)))

        elif setname == 'test':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "test" / "*depth0_noise.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "test" / "*depth0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "test" / "*data0.json"), recursive=True)))

        self.gt = self.gt[:10]
        self.depth = self.depth[:10]

        assert (len(self.gt) == len(self.depth))

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Read depth input and gt

        f = open(self.data[item])
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(self.depth[item],
                                              data['minDepth'],
                                              data['maxDepth'])
        gt = file_io.load_scaled16bitImage(self.gt[item],
                                           data['minDepth'],
                                           data['maxDepth'])

        # Normalize the depth
        depth = depth.reshape(512, 512)
        gt = gt.reshape(512, 512)

        # gt = gt.reshape(512, 512,3)
        depth_mask = ~(depth == 0)
        gt_mask = ~(gt == 0)

        depth[depth_mask] = (depth[depth_mask] - data['minDepth']) / (data['maxDepth'] - data['minDepth'])  # [0,1]
        gt[gt_mask] = (gt[gt_mask] - data['minDepth']) / (data['maxDepth'] - data['minDepth'])

        # Expand dims into Pytorch format
        depth = np.expand_dims(depth, 0)
        gt = np.expand_dims(gt, 0)

        # Convert to Pytorch Tensors
        depth = torch.from_numpy(depth)  # (depth, dtype=torch.float)
        gt = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)

        # Convert depth to disparity
        if self.invert_depth:
            depth[depth == 0] = -1
            depth = 1 / depth
            depth[depth == -1] = 0

            gt[gt == 0] = -1
            gt = 1 / gt
            gt[gt == -1] = 0

        input = depth

        return input, gt
