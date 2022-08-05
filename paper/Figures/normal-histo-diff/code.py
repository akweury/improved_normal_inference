import shutil

import cv2 as cv
import numpy as np
import torch

import config
from help_funs import file_io, mu, chart

if __name__ == '__main__':
    # test_torch()
    # print_cuda_info()
    folder_path = config.paper_pic

    gcnn_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.npy"))
    gt_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_gt.npy"))
    shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.png"),
                    str(folder_path / f"rgb-diff" / "gcnn.png"))
    shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_groundtruth.png"),
                    str(folder_path / f"rgb-diff" / "gt.png"))

    histo_gcnn, hist_x = mu.normal_histo(gcnn_normal)
    histo_gt, _ = mu.normal_histo(gt_normal)
    mu.save_array(histo_gcnn, str(folder_path / f"rgb-diff" / "gcnn-histo"))

    chart.line_chart(histo_gcnn, str(folder_path / f"rgb-diff"), ["x", "y", "z"], x=hist_x[1:])
    chart.line_chart(histo_gt, str(folder_path / f"rgb-diff"), ["x_gt", "y_gt", "z_gt"], x=hist_x[1:])
