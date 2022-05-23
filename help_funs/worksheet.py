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
        config.ws_path / "00327.depth0_noise.png")
    img = mu.hpf(depth_file)
    img[img.sum(axis=2) != 0] = 255
    cv.imwrite(str(Path(config.ws_path) / f"detail.png"), img)

    pass
