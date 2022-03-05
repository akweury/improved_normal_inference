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

import cv2 as cv
from torchvision import transforms
import config
from help_funs import file_io, mu


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
            self.gt = np.array(sorted(glob.glob(str(depth_path / "train" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "train" / "*data0.json"), recursive=True)))

        elif setname == 'selval':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "eval" / "*depth0_noise.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "eval" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "eval" / "*data0.json"), recursive=True)))

        elif setname == 'test':
            depth_path = self.synthetic_depth_path
            self.depth = np.array(sorted(glob.glob(str(depth_path / "test" / "*depth0_noise.png"), recursive=True)))
            self.gt = np.array(sorted(glob.glob(str(depth_path / "test" / "*normal0.png"), recursive=True)))
            self.data = np.array(sorted(glob.glob(str(depth_path / "test" / "*data0.json"), recursive=True)))
        #
        # self.gt = self.gt[:10]
        # self.depth = self.depth[:10]

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
        # depth_mask = ~(depth == 0)
        # depth[depth_mask] = (depth[depth_mask] - data['minDepth']) / (data['maxDepth'] - data['minDepth'])  # [0,1]
        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)
        vertex_input = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                       torch.tensor(data['K']),
                                       torch.tensor(data['R']).float(),
                                       torch.tensor(data['t']).float())
        vertex_input = torch.from_numpy(vertex_input)  # (depth, dtype=torch.float)
        vertex_input = vertex_input.permute( 2, 0, 1)



        # gt = file_io.load_scaled16bitImage(self.gt[item],
        #                                    data['minDepth'],
        #                                    data['maxDepth'])
        # gt_mask = ~(gt == 0)
        # gt[gt_mask] = (gt[gt_mask] - data['minDepth']) / (data['maxDepth'] - data['minDepth'])
        # vertex_gt = mu.depth2vertex(torch.tensor(gt).permute(2, 0, 1),
        #                             torch.tensor(data['K']),
        #                             torch.tensor(data['R']).float(),
        #                             torch.tensor(data['t']).float())
        # vertex_gt = torch.from_numpy(vertex_gt)  # tensor(gt, dtype=torch.float)
        # vertex_gt = vertex_gt.permute(2,0,1)

        gt = file_io.load_24bitNormal(self.gt[item])
        normal_gt = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        normal_gt = normal_gt.permute(2,0,1)


        return vertex_input, normal_gt
