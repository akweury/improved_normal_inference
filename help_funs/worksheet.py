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
    data_file = str(config.synthetic_data / "test" / '00327.normal0.png')
    # depth_file = str(config.ws_path / "a.png")
    normal = file_io.load_24bitNormal(data_file).astype(np.float32)

    data_tensor = torch.from_numpy(normal).permute(2, 0, 1)
    img = mu.hpf_torch(data_tensor)

    cv.imwrite(str(config.ws_path / f"aa.png"), img.numpy())

    pass
