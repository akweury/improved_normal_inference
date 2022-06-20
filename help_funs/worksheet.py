import shutil

import numpy as np
import torch
import glob
import config
from help_funs import mu, chart


def gau_histo(gt_normal, sigma):
    gt_normal = torch.from_numpy(gt_normal)
    mask = gt_normal != 0
    min = -2
    max = 2
    bins = 100
    delta = float(max - min) / float(bins)
    centers = min + delta * (torch.arange(bins).float() + 0.5)

    output_histo = gt_normal[mask].unsqueeze(0) - centers.unsqueeze(1)
    output_histo = torch.exp(-0.5 * (output_histo / sigma) ** 2) / (sigma * np.sqrt(np.pi * 2)) * delta
    output_histo = output_histo.sum(dim=-1)
    output_histo = (output_histo - output_histo.min()) / (output_histo.max() - output_histo.min())  # normalization
    return output_histo


if __name__ == '__main__':
    test_0_data = np.array(sorted(glob.glob(str(config.synthetic_data_noise_128_local / "train" / "tensor" /
                                                f"*_0_*"), recursive=True)))
    test_0 = torch.load(test_0_data[0])
    test_0_tensor = test_0['input_tensor'].unsqueeze(0)

    # test_torch()
    # print_cuda_info()
    # folder_path = config.paper_pic
    #
    # gcnn_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.npy"))
    # gt_normal = np.load(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_gt.npy"))
    # shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_normal_GCNN.png"),
    #                 str(folder_path / f"normal-histo-diff" / "gcnn.png"))
    # shutil.copyfile(str(config.paper_pic / "comparison" / "fancy_eval_3_groundtruth.png"),
    #                 str(folder_path / f"normal-histo-diff" / "gt.png"))
    #
    # histo_gcnn, hist_x = mu.normal_histo(gcnn_normal)
    # histo_gt, _ = mu.normal_histo(gt_normal)
    #
    # tensor_histo = histo_gt[0, :]
    # tensor_histo = (tensor_histo - tensor_histo.min()) / (tensor_histo.max() - tensor_histo.min())
    #
    # gaussian_histo = gau_histo(gt_normal[:, :, 0], 3 * 25).numpy()
    #
    # mu.save_array(histo_gcnn, str(folder_path / f"normal-histo-diff" / "gcnn-histo"))
    #
    # chart.line_chart(histo_gcnn, str(folder_path / f"normal-histo-diff"), ["x", "y", "z"], x=hist_x[1:])
    # chart.line_chart(histo_gt, str(folder_path / f"normal-histo-diff"), ["x_gt", "y_gt", "z_gt"], x=hist_x[1:])
