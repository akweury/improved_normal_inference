"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import os
import argparse

import glob
import cv2 as cv
import torch
import datetime
import numpy as np
from help_funs import file_io, mu
import config

from workspace import eval
from help_funs import chart
from help_funs.data_preprocess import noisy_a_folder


def eval_post_processing(normal, normal_img, normal_gt, name):
    out_ranges = mu.addHist(normal_img)
    mu.addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)
    mu.addText(normal_img, name, font_size=0.8)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img, diff


def model_eval(model_path, input, gt, name):
    noraml, img, _, _ = eval.eval(input, model_path, output_type='normal')
    normal_img, angle_err_img, err = eval_post_processing(noraml, img, gt, name)
    return normal_img, angle_err_img, err


def preprocessing():
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic', help="choose evaluate dataset")
    parser.add_argument('--noise', type=bool, default=False, help='add noise to the test folder')
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    if args.noise:
        original_folder = config.synthetic_data / "test"
        if args.machine == "remote":
            dataset_folder = config.synthetic_data_noise_dfki / "test"
        elif args.machine == 'local':
            dataset_folder = config.synthetic_data_noise / "test"
        else:
            raise ValueError
        noisy_a_folder(original_folder, dataset_folder)

    # load data file names
    if args.data == "synthetic":
        path = config.synthetic_data_noise / "test"
    elif args.data == "real":
        path = config.real_data  # key tests 103, 166, 189,9
    else:
        raise ValueError
    # test dataset indices
    all_names = [os.path.basename(path) for path in sorted(glob.glob(str(path / f"*.image?.png"), recursive=True))]
    all_names = np.array(all_names)
    eval_indices = [int(name.split(".")[0]) for name in all_names]
    eval_indices = eval_indices[:5]

    # load test model names
    models = {
        "SVD": None,
        "Neigh_9999": config.ws_path / "nnn24" / "trained_model" / "full_normal_2999" / "checkpoint-9999.pth.tar",
        "NNNN_10082": config.ws_path / "nnnn" / "trained_model" / "output_2022-05-02_08_09_16" / "checkpoint-10082.pth.tar",
        "NG_2994": config.ws_path / "ng" / "trained_model" / "output_2022-05-08_08_17_02" / "checkpoint-2994.pth.tar",
        "NG+_5516": config.ws_path / "ng" / "trained_model" / "output_2022-05-09_21_40_33" / "checkpoint-5516.pth.tar",
    }
    eval_res = np.zeros((len(models), len(eval_indices)))

    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    folder_path = config.ws_path / "eval_output" / f"{eval_date}_{eval_time}"
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    print(f"\n\n==================== Evaluation Start =============================\n"
          f"Eval Date: {eval_date}\n"
          f"Eval Time: {eval_time}\n"
          f"Eval Objects: {len(eval_indices)}\n"
          f"Eval Models: {models.keys()}\n")
    return models, eval_indices, path, folder_path, eval_res, eval_date, eval_time


def main():
    models, eval_idx, dataset_path, folder_path, eval_res, eval_date, eval_time = preprocessing()

    for i, data_idx in enumerate(eval_idx):
        # read data
        data, depth, depth_noise, normal_gt, _ = file_io.load_single_data(dataset_path, idx=data_idx)

        vertex_filted_gt = None
        if dataset_path == config.real_data:
            depth_filtered = mu.median_filter(depth)
            vertex_filted_gt = mu.depth2vertex(torch.tensor(depth_filtered).permute(2, 0, 1),
                                               torch.tensor(data['K']),
                                               torch.tensor(data['R']).float(),
                                               torch.tensor(data['t']).float())
        vertex = mu.depth2vertex(torch.tensor(depth_noise).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())

        img_list = []
        diff_list = []
        # ground truth normal
        normal_gt_img = mu.normal2RGB(normal_gt)
        # normal_gt_ = mu.rgb2normal(normal_gt_img)
        # gt_img, gt_diff = eval_post_processing(normal_gt_, normal_gt_img, normal_gt, "GT")
        img_list.append(normal_gt_img)
        # diff_list.append(gt_diff)

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):
            if dataset_path == config.real_data and name == "Neigh_9999":
                normal_img, angle_err_img, err = model_eval(model, vertex_filted_gt, normal_gt, name)
            else:
                normal_img, angle_err_img, err = model_eval(model, vertex, normal_gt, name)
            img_list.append(normal_img)
            diff_list.append(angle_err_img)
            eval_res[model_idx, i] = err
        # show the results
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        output_diff = cv.hconcat(diff_list)
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        date_now = datetime.datetime.today().date()
        cv.imwrite(str(folder_path / f"evaluation_{date_now}_{time_now}.png"), output)
        cv.imwrite(str(folder_path / f"diff{date_now}_{time_now}.png"), output_diff)
        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    main()
