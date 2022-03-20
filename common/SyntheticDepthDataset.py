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

from help_funs import file_io, mu


class SyntheticDepthDataset(Dataset):

    def __init__(self, synthetic_depth_path, setname='train'):

        self.synthetic_depth_path = synthetic_depth_path

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
        #

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
        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())

        input = torch.from_numpy(vertex)  # (depth, dtype=torch.float)
        input = input.permute(2, 0, 1)

        gt = file_io.load_24bitNormal(self.gt[item]).astype(np.float32)
        gt = mu.normal2RGB(gt)
        gt = (gt).astype(np.float32)
        normal_gt = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        normal_gt = normal_gt.permute(2, 0, 1)
        # print(f"data: max, min:{vertex_input.max(), vertex_input.min()}")
        return input, normal_gt
