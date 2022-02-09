import os
from os import path
import cv2 as cv
import numpy as np
import json
import torch
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree

import config
import chart
import statistic
import file_io

k_min = 3
k_max = 10
max_file_num = 4
data_type = "synthetic_basic"
# data_type = "synthetic_captured"
# data_type = "real"

print(f"knn start, k in range [{k_min}, {k_max}]. ")
# new k loop
for k_idx in range(k_min, k_max):
    print(f"\nk={k_idx} ")
    file_idx = 0

    file = file_io.get_file_name(file_idx, "pointcloud", data_type)

    # new scene loop
    while path.exists(file) and file_idx < max_file_num:
        print(f"Reading {file}...")

        # f = open(file)
        # data = json.load(f)
        # f.close()

        ply_data = PlyData.read(file)
        vertex = np.c_[ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]
        # remove camera(1944) and light(1330) points
        vertex = vertex[:-(1944 + 1330), :]

        dist, ind = KDTree(vertex).query(vertex, k_idx)
        vertexs_neighbors = vertex[ind]
        normals = np.zeros(shape=vertex.shape)
        for i in range(vertexs_neighbors.shape[0]):
            u, s, vh = np.linalg.svd(vertexs_neighbors[i])
            normal = vh.T[:, -1]
            normals[i] = normal * 0.5 + 0.5  # shift the range from [-1,1] to [0,1]
            normals[i][1] = 1 - normals[i][1]
            normals[i] = (normals[i] * 255).astype(np.int32)

        normal_image = cv.imread(file_io.get_output_file_name(file_idx, "normal", "gt", data_type), -1)
        valid_pixel_idx = file_io.get_valid_pixels_idx(normal_image)
        assert np.count_nonzero(valid_pixel_idx) == normals.shape[0]
        normal_idx = 0
        for i in range(normal_image.shape[0]):
            for j in range(normal_image.shape[1]):
                if valid_pixel_idx[i][j]:
                    normal_image[i][j] = normals[normal_idx]
                    normal_idx += 1

        cv.imwrite(file_io.get_output_file_name(file_idx, "normal", "knn", data_type, k_idx), normal_image)
        file_idx += 1
        file = file_io.get_file_name(file_idx, "pointcloud", data_type)

# --------------------------- visualisation -------------------------------- #
if k_max - k_min < 2:
    exit()

diffs = np.zeros((max_file_num, k_max))
file_idx = 0
file_name = file_io.get_output_file_name(file_idx, "normal", "gt", data_type)
while path.exists(file_name) and file_idx < max_file_num:
    gt = file_io.get_output_file_name(file_idx, "normal", "gt", data_type)
    if os.path.exists(gt):
        normal_gt = cv.imread(gt)
        valid_pixels = statistic.get_valid_pixels(normal_gt)
    else:
        continue

    for j in range(k_min, k_max):
        knn = file_io.get_output_file_name(file_idx, file_type="normal", method="knn", data_type=data_type, param=j)
        if os.path.exists(knn):
            normal_knn = cv.imread(knn)
            diffs[file_idx, j] = statistic.mse(normal_gt, normal_knn)
        else:
            diffs[file_idx, j] = 0
            continue

    print(f"data {file_idx}: mse \n {diffs}")
    file_idx += 1
    file_name = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
# remove 0 elements
diffs = diffs[:, k_min:k_max]
# visualisation
chart.line_chart(diffs,
                 title="normal_performance",
                 x_scale=[k_min, 1],
                 x_label="k_value",
                 y_label="RGB_difference")
