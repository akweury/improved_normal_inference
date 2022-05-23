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

if __name__ == '__main__':
    # data_file = str(config.synthetic_data / "test" / '00327.data0.json')
    depth_file = str(
        config.ws_path / "nnnn" / "trained_model" / "output_2022-05-23_09_47_23" / "train_epoch_2999_0_loss_0.01001242.png")
    img = mu.hpf(depth_file)

    cv.imwrite(str(Path(config.ws_path) / f"detail.png"), img)

    pass
