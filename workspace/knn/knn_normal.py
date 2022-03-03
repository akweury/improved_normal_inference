from os import path
import json
import cv2 as cv
import numpy as np
import torch

import improved_normal_inference.help_funs.file_io
import improved_normal_inference.help_funs.mu
from improved_normal_inference.help_funs import file_io, mu
from improved_normal_inference.help_funs.mu import normal2RGB, copy_make_border
from improved_normal_inference.help_funs.ply_generator import generate_ply
from improved_normal_inference import config


# data_type = "synthetic_captured"
# data_type = "real"

def compute_normal(vertex, mask, k):
    normals = np.zeros(shape=vertex.shape)
    for i in range(k, vertex.shape[0]):
        for j in range(k, vertex.shape[1]):
            if mask[i, j]:
                neighbors = vertex[i - k:i + k, j - k:j + k]  # get its k neighbors
                neighbors = neighbors.reshape(neighbors.shape[0] * neighbors.shape[1], 3)
                neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)
                plane_vectors = neighbors - vertex[i, j]

                u, s, vh = np.linalg.svd(plane_vectors)
                normal = vh.T[:, -1]
                normal = mu.normal_point_to_view_point(normal, vertex[i][j], np.array([0, 0, 0]))
                if np.linalg.norm(normal) != 1:
                    normal = normal / np.linalg.norm(normal)
                normals[i, j] = normal

    return normals


def knn_normal(data_path, file_idx, k_min=1, k_max=3):
    print(f"knn start, k in range [{k_min}, {k_max}]. \n")

    # k loop
    for k_idx in range(k_min, k_max):
        print(f"k={k_idx} ")

        # get file names from dataset
        image_file, ply_file, json_file, depth_file, normal_file = file_io.get_file_name(file_idx, data_path)
        f = open(json_file)
        data = json.load(f)
        normal_gt = cv.imread(normal_file)
        # normal_gt = file_io.load_24bitNormal(normal_file)
        depth = file_io.load_scaled16bitImage(depth_file, data['minDepth'], data['maxDepth'])
        if file_idx == 0 and not path.exists(ply_file):
            print("No '.ply' file available, generate now...")
            generate_ply(file_idx, normal_gt, data_path=data_path)

        depth_padded = np.expand_dims(copy_make_border(depth, k_idx), axis=2)
        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)

        # calculate vertex from depth map
        vertex = mu.depth2vertex(torch.tensor(depth_padded).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())
        mask = np.sum(np.abs(depth_padded), axis=2) != 0

        # compute normals from neighbors
        normals = compute_normal(vertex, mask, k_idx)
        normals_rgb = normal2RGB(normals, mask)
        # generate point clouds, save calculated normal image and ground truth image
        generate_ply(file_idx, normals, data_path=data_path, param=k_idx)
        file_io.write_np2img(normals_rgb, file_io.get_output_file_name(file_idx, "normal", "knn", k_idx))
        cv.imwrite(file_io.get_output_file_name(file_idx, "normal", "knn", 0), normal_gt)

        # # calculate the difference between calculated image and ground truth image
        # angle_difference = np.zeros(normal_gt.shape[:2])
        # for i in range(angle_difference.shape[0]):
        #     for j in range(angle_difference.shape[1]):
        #         angle_difference[i, j] = mu.angle_between(normals[i, j], normal_gt[i, j])

        print(f"Saved {file_io.get_output_file_name(file_idx, 'normal', 'knn', k_idx)}")
        print(f"Saved {file_io.get_output_file_name(file_idx, 'normal', 'knn', 0)}")


def depth2normal(depth, k_idx, K, R, t):
    depth_padded = np.expand_dims(copy_make_border(depth, k_idx), axis=2)
    vertex = mu.depth2vertex(torch.tensor(depth_padded).permute(2, 0, 1),
                             torch.tensor(K),
                             torch.tensor(R).float(),
                             torch.tensor(t).float())
    mask = np.sum(np.abs(depth_padded), axis=2) != 0
    normals = compute_normal(vertex, mask, k_idx)
    normals_rgb = normal2RGB(normals, mask)
    return normals, normals_rgb


def knn_normal_pred(data_path, file_idx, depth_pred):
    k_idx = 2
    print(f"knn start, k = {k_idx}\n")

    # get file names from dataset
    image_file, ply_file, json_file, depth_gt_file, normal_file = file_io.get_file_name(file_idx, data_path)
    f = open(json_file)
    data = json.load(f)

    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    normal_gt = cv.imread(normal_file)
    # normal_gt = file_io.load_24bitNormal(normal_file)
    depth_gt = file_io.load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])
    # if not path.exists(ply_file):
    #     print("No '.ply' file available, generate now...")
    #     generate_ply(file_idx, normal_gt, data_path=data_path)


    depth_pred = file_io.scale16bitImage(depth_pred, data['minDepth'], data['maxDepth'])

    normals_gt, normals_rgb_gt = depth2normal(depth_gt, k_idx, data['K'], data['R'], data['t'])
    normals_pred, normals_rgb_pred = depth2normal(depth_pred, k_idx, data['K'], data['R'], data['t'])

    # generate point clouds, save calculated normal image and ground truth image
    generate_ply(file_idx, normals_pred, data_path=data_path, param=k_idx)

    return normals_rgb_gt, normals_rgb_pred

    # file_io.write_np2img(normals_rgb_pred, file_io.get_output_file_name(file_idx, "normal", "knn", k_idx))
    # cv.imwrite(file_io.get_output_file_name(file_idx, "normal", "knn", 0), normal_gt)

    # # calculate the difference between calculated image and ground truth image
    # angle_difference = np.zeros(normal_gt.shape[:2])
    # for i in range(angle_difference.shape[0]):
    #     for j in range(angle_difference.shape[1]):
    #         angle_difference[i, j] = mu.angle_between(normals[i, j], normal_gt[i, j])

    # print(f"Saved {file_io.get_output_file_name(file_idx, 'normal', 'knn', k_idx)}")
    # print(f"Saved {file_io.get_output_file_name(file_idx, 'normal', 'knn', 0)}")


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
    k_min, k_max = 2, 3
    data_path = config.real_data
    file_idx = 0
    knn_normal(file_idx, k_min, k_max)
