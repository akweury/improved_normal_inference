import os
from os import path
import cv2 as cv
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree

import config
import chart
import statistic

k_min = 3
k_max = 7
max_file_num = 10
data_type = "synthetic"
# data_type = "real"

print(f"knn start, k in range [{k_min}, {k_max}]. ")
# new k loop
for k_idx in range(k_min, k_max):
    print(f"\nk={k_idx} ")
    file_idx = 0
    file_name = config.get_file_name(file_idx, "pointcloud", data_type)
    # new scene loop
    while path.exists(file_name) and file_idx < max_file_num:
        print(f"Reading {file_name}...")
        ply_data = PlyData.read(file_name)
        vertex = np.c_[ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]
        dist, ind = KDTree(vertex).query(vertex, k_idx)
        vertexs_neighbors = vertex[ind]
        normals = np.zeros(shape=vertex.shape)
        for i in range(vertexs_neighbors.shape[0]):
            u, s, vh = np.linalg.svd(vertexs_neighbors[i])
            normal = vh.T[:, -1]
            normals[i] = normal * 0.5 + 0.5  # shift the range from [-1,1] to [0,1]
            normals[i][1] = 1 - normals[i][1]
            normals[i] = (normals[i] * 255).astype(np.int32)
        normal_image = cv.imread(config.get_output_file_name(file_idx, "normal", "gt"))

        normal_idx = 0
        for i in range(normal_image.shape[0]):
            for j in range(normal_image.shape[1]):
                if np.sum(normal_image[i][j]) != 0:
                    normal_image[i][j] = normals[normal_idx]
                    normal_idx += 1

        cv.imwrite(config.get_output_file_name(file_idx, "normal", "knn", k_idx), normal_image)
        file_idx += 1
        file_name = config.get_file_name(file_idx, "pointcloud", data_type)

##################### visualisation  #######################################
diffs = np.zeros((max_file_num, k_max))
file_idx = 0
file_name = config.get_output_file_name(file_idx, "normal", "gt")
while path.exists(file_name) and file_idx < max_file_num:
    gt = config.get_output_file_name(file_idx, "normal", "gt")
    if os.path.exists(gt):
        normal_gt = cv.imread(gt)
        valid_pixels = statistic.get_valid_pixels(normal_gt)
    else:
        continue

    for j in range(k_min, k_max):
        knn = config.get_output_file_name(file_idx, "normal", "knn", j)
        if os.path.exists(knn):
            normal_knn = cv.imread(knn)
            diffs[file_idx, j] = statistic.mse(normal_gt, normal_knn)
        else:
            diffs[file_idx, j] = 0
            continue

    print(f"data {file_idx}: mse \n {diffs}")
    file_idx += 1
    file_name = config.get_output_file_name(file_idx, "normal", "gt")
# remove 0 elements
diffs = diffs[:, k_min:k_max]
# visualisation
chart.line_chart(diffs,
                 title="normal performance",
                 x_scale=[k_min, 1],
                 x_label="k value",
                 y_label="RGB difference")
