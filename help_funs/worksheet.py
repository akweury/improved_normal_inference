import mu
import file_io
import json

import os
from pathlib import Path
import json
import numpy as np
import torch
import cv2 as cv
import glob
import shutil
import matplotlib.pyplot as plt
import config
from help_funs import file_io, mu, chart
from pncnn.utils import args_parser


def test_torch():
    # depth_file = str(config.ws_path / "a.png")
    data_file = str(config.synthetic_data / "test" / '00327.normal0.png')
    normal = file_io.load_24bitNormal(data_file).astype(np.float32)
    data_tensor = torch.from_numpy(normal).permute(2, 0, 1)
    img = mu.hpf_torch(data_tensor)
    cv.imwrite(str(config.ws_path / f"aa.png"), img.numpy())


def test_png():
    depth_file = str(
        config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / "train_epoch_199_0_loss_0.16585270.png")
    img = mu.hpf(depth_file)
    cv.imwrite(str(config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / f"aa.png"), img)


if __name__ == '__main__':
    # test_torch()
    test_png()

    pass
