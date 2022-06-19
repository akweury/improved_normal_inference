import shutil

import cv2 as cv
import numpy as np
import torch

import config
from help_funs import file_io, mu, chart


def test_torch():
    # depth_file = str(config.ws_path / "a.png")
    data_file = str(config.synthetic_data / "train" / '00102.normal0.png')
    normal = file_io.load_24bitNormal(data_file).astype(np.float32)
    data_tensor = torch.from_numpy(normal).permute(2, 0, 1)
    img, img_extended = mu.hpf_torch(data_tensor)
    cv.imwrite(str(config.ws_path / f"aa.png"), img.numpy())


#
#
# def test_png():
#     depth_file = str(
#         config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / "train_epoch_199_0_loss_0.16585270.png")
#     img = mu.hpf(depth_file)
#     cv.imwrite(str(config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / f"aa.png"), img)


def print_cuda_info():
    print(f"Cuda is available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


if __name__ == '__main__':
    # test_torch()
    # print_cuda_info()
    folder_path = config.paper_pic

    gcnn_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.npy"))
    gt_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_gt.npy"))

    diff = mu.rgb_diff(gcnn_normal, gt_normal)
    mu.save_array(diff, str(folder_path / f"rgb-diff" / "diff"))
    diff_plot = chart.line_chart(diff, str(folder_path / f"rgb-diff"), ["red", "green", "blue"])
    shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.png"),
                    str(folder_path / f"rgb-diff" / "gcnn.png"))
    shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_groundtruth.png"),
                    str(folder_path / f"rgb-diff" / "gt.png"))
