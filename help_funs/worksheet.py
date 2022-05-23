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
    depth_file = str(config.synthetic_data_noise / "test" / "00349.normal0.png")
    img = mu.hpf(depth_file)

    cv.imwrite(str(config.root / "paper" / "pic" / f"00349.hpf0.png"), img)

    pass
