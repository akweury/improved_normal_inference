from os import path
import json
import cv2 as cv
import numpy as np
import torch

import file_io
import math_utils
from improved_normal_inference.file_io import copy_make_border
from ply_generator import generate_ply

k_min, k_max = 2, 3
data_type = "synthetic_basic"


# data_type = "synthetic_captured"
# data_type = "real"


def compute_normal(vertex, mask, k):
    normals = np.zeros(shape=vertex.shape)
    for i in range(vertex.shape[0]):
        for j in range(vertex.shape[1]):
            if mask[i, j]:
                neighbors = vertex[i - k:i + k, j - k:j + k]  # get its k neighbors
                neighbors = neighbors.reshape(neighbors.shape[0] * neighbors.shape[1], 3)
                neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)
                plane_vectors = neighbors - vertex[i, j]

                u, s, vh = np.linalg.svd(plane_vectors)
                normal = vh.T[:, -1]
                normal = math_utils.normal_point_to_view_point(normal, vertex[i][j], np.array([0, 0, 0]))
                if np.linalg.norm(normal) != 1:
                    normal = normal / np.linalg.norm(normal)
                normals[i, j] = normal

    return normals


def normal2RGB(normals, mask):
    # convert normal to RGB color
    h, w, c = normals.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                normals[i, j] = normals[i, j] * 0.5 + 0.5
                normals[i, j, 2] = 1 - normals[i, j, 2]
                # normals[i, j,2] = 1 - normals[i, j,2]
                # normals[i, j,0] = 1 - normals[i, j,0]
                normals[i, j] = (normals[i, j] * 255).astype(np.int32)

    return normals


def knn_normal(file_idx, k_min=1, k_max=3):
    print(f"knn start, k in range [{k_min}, {k_max}]. \n")

    # k loop
    for k_idx in range(k_min, k_max):
        print(f"k={k_idx} ")
        file = file_io.get_file_name(file_idx, "pointcloud", data_type)

        if file_idx == 0 and not path.exists(file):
            print("No '.ply' file available.")
            exit()

        # read json file
        file_json = file_io.get_file_name(file_idx, "data", data_type=data_type)
        f = open(file_json)
        data = json.load(f)
        f.close()

        # image = load_8bitImage(get_file_name(file_idx, "image", data_type))
        # calculate vertex from depth map
        depth = file_io.load_scaled16bitImage(file_io.get_file_name(file_idx, "depth", data_type),
                                              data['minDepth'],
                                              data['maxDepth'])

        depth = -depth
        depth_padded = np.expand_dims(copy_make_border(depth, k_idx), axis=2)

        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)

        vertex = file_io.depth2vertex(torch.tensor(depth_padded).permute(2, 0, 1),
                                      torch.tensor(data['K']),
                                      torch.tensor(data['R']).float(),
                                      torch.tensor(data['t']).float())
        mask = np.sum(np.abs(depth_padded), axis=2) != 0

        # compute normals from neighbors
        normals = compute_normal(vertex, mask, k_idx)
        normal_gt = cv.imread(file_io.get_output_file_name(file_idx, "normal", "gt", data_type), -1)
        # normal_gt = file_io.load_24bitNormal(file_io.get_output_file_name(file_idx, "normal", "gt", data_type))
        angle_difference = np.zeros(normal_gt.shape[:2])
        generate_ply(file_idx, normals, method="knn", data_type=data_type, param=k_idx)

        normals = normal2RGB(normals, mask)

        for i in range(angle_difference.shape[0]):
            for j in range(angle_difference.shape[1]):
                angle_difference[i, j] = math_utils.angle_between(normals[i, j], normal_gt[i, j])

        cv.imwrite(file_io.get_output_file_name(file_idx, "normal", "knn", data_type, 0), normal_gt)
        cv.imwrite(file_io.get_output_file_name(file_idx, "normal", "knn", data_type, k_idx), normals)
        print(f"Saved {file_io.get_output_file_name(file_idx, 'normal', 'knn', data_type, k_idx)}")


# --------------------------- visualisation -------------------------------- #
# if k_max - k_min < 2:
#     exit()
#
# diffs = np.zeros((file_idx, k_max))
# file_idx = 0
# file_name = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
# while path.exists(file_name) and file_idx < max_file_num:
#     gt = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
#     if os.path.exists(gt):
#         normal_gt = cv.imread(gt)
#         valid_pixels = statistic.get_valid_pixels(normal_gt)
#     else:
#         continue
#
#     for j in range(k_min, k_max):
#         knn = file_io.get_output_file_name(file_idx, file_type="normal", method="knn", data_type=data_type, param=j)
#         if os.path.exists(knn):
#             normal_knn = cv.imread(knn)
#             diffs[file_idx, j] = statistic.mse(normal_gt, normal_knn)
#         else:
#             diffs[file_idx, j] = 0
#             continue
#
#     file_idx += 1
#     file_name = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
#
# # remove 0 elements
# diffs = diffs[:, k_min:k_max]
# print(f"mse: \n {diffs}")
# # visualisation
# chart.line_chart(diffs,
#                  title="normal_performance",
#                  x_scale=[k_min, 1],
#                  x_label="k_value",
#                  y_label="RGB_difference")
if __name__ == '__main__':
    file_idx = 0
    knn_normal(file_idx, k_min, k_max)
