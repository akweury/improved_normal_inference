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

import config
from help_funs import file_io, mu, chart
from pncnn.utils import args_parser


##################

def save_input():
    data_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.data0.json"
    depth_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.depth0_noise.png"

    f = open(data_file)
    data = json.load(f)
    f.close()

    depth = file_io.load_scaled16bitImage(depth_file,
                                          data['minDepth'],
                                          data['maxDepth'])

    output_name = "00440.vertex"
    mu.visual_input(depth, data, output_name)


def save_data_range():
    data_path = config.synthetic_data / "train"

    if not os.path.exists(str(data_path)):
        raise FileNotFoundError

    depth_files = np.array(sorted(glob.glob(str(data_path / "*depth0.png"), recursive=True)))[:100]
    data_files = np.array(sorted(glob.glob(str(data_path / "*data0.json"), recursive=True)))[:100]

    data_extreme = np.zeros((6, depth_files.shape[0]))
    data_range = np.zeros((3, depth_files.shape[0]))
    for item in range(len(data_files)):
        f = open(data_files[item])
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(depth_files[item],
                                              data['minDepth'],
                                              data['maxDepth'])

        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())
        mask = vertex.sum(axis=2) == 0
        data_extreme[0, item] = vertex[:, :, :1][~mask].min()
        data_extreme[1, item] = vertex[:, :, :1][~mask].max()
        data_extreme[2, item] = vertex[:, :, 1:2][~mask].min()
        data_extreme[3, item] = vertex[:, :, 1:2][~mask].max()
        data_extreme[4, item] = vertex[:, :, 2:3][~mask].min()
        data_extreme[5, item] = vertex[:, :, 2:3][~mask].max()

        data_range[0, item] = vertex[:, :, :1][~mask].max() - vertex[:, :, :1][~mask].min()
        data_range[1, item] = vertex[:, :, 1:2][~mask].max() - vertex[:, :, 1:2][~mask].min()
        data_range[2, item] = vertex[:, :, 2:3][~mask].max() - vertex[:, :, 2:3][~mask].min()

        print(f'Processing file {item} in {len(data_files)}.')
    extreme_labels = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max", ]
    range_labels = ["x_range", "y_range", "z_range"]

    chart.line_chart(data_extreme, config.dataset, title="Data_Extreme_Value", labels=extreme_labels, cla_leg=True)
    chart.line_chart(data_range, config.dataset, title="Data_Range", labels=range_labels)

    print(f"x_range: {data_range[0, :].sum() / len(data_files):.2e}\n"
          f"y_range: {data_range[1, :].sum() / len(data_files):.2e}\n"
          f"z_range: {data_range[2, :].sum() / len(data_files):.2e}\n"
          f"x_min: {data_extreme[0, :].sum() / len(data_files):.2e}\n"
          f"x_max: {data_extreme[1, :].sum() / len(data_files):.2e}\n"
          f"y_min: {data_extreme[2, :].sum() / len(data_files):.2e}\n"
          f"y_max: {data_extreme[3, :].sum() / len(data_files):.2e}\n"
          f"z_min: {data_extreme[4, :].sum() / len(data_files):.2e}\n"
          f"z_max: {data_extreme[5, :].sum() / len(data_files):.2e}\n"
          f"\n")


if __name__ == '__main__':
    save_data_range()
