import os
from os import path
from pathlib import Path
import json
import numpy as np
import torch
import cv2 as cv
import glob
import shutil

import config
from help_funs import file_io, mu


def noisy_1channel(img):
    img = img.reshape(512, 512)
    h, w = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise_img = img * noise
    return noise_img


def noisy(img):
    h, w, c = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise = noise.reshape(h, w, 1)
    noise = np.repeat(noise, c, axis=-1)
    noise_img = img * noise
    return noise_img


def noisy_a_folder(folder_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx in range(500):
        image_file, ply_file, json_file, depth_gt_file, _, normal_file = file_io.get_file_name(idx, folder_path)
        if path.exists(image_file):
            f = open(json_file)
            data = json.load(f)
            # normal_gt = cv.imread(normal_file)
            depth = file_io.load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])

            # add noise
            noise_depth = noisy_1channel(depth)

            # save files to the new folders
            file_io.save_scaled16bitImage(noise_depth,
                                          str(output_path / (str(idx).zfill(5) + ".depth0_noise.png")),
                                          data['minDepth'], data['maxDepth'])
            shutil.copyfile(depth_gt_file, str(output_path / (str(idx).zfill(5) + ".depth0.png")))
            shutil.copyfile(image_file, str(output_path / (str(idx).zfill(5) + ".image0.png")))
            shutil.copyfile(json_file, str(output_path / (str(idx).zfill(5) + ".data0.json")))
            shutil.copyfile(normal_file, str(output_path / (str(idx).zfill(5) + ".normal0.png")))

            print(f'File {idx} added noise.')


def neighbor_vector(vertex, xu, xd, yl, yr, zf, zb):
    vertex_padded = np.pad(vertex, ((xd, xu), (yr, yl), (zf, zb)))
    h, w, c = vertex_padded.shape
    return vertex_padded[xu:h - xd, yl:w - yr, zf:c - zb] - vertex


def neighbor_vectors_k(vertex, k=2):
    vectors = np.zeros(shape=vertex.shape)
    for i in range(k):
        for j in range(k):
            if i == j and i == 0:
                continue
            vectors = np.concatenate((vectors, neighbor_vector(vertex, 0, i, 0, j, 0, 0)), axis=2)
            if j != 0:
                vectors = np.concatenate((vectors, neighbor_vector(vertex, 0, i, j, 0, 0, 0)), axis=2)
            if i != 0:
                vectors = np.concatenate((vectors, neighbor_vector(vertex, i, 0, j, 0, 0, 0)), axis=2)
                if j != 0:
                    vectors = np.concatenate((vectors, neighbor_vector(vertex, i, 0, 0, j, 0, 0)), axis=2)

    return vectors[:, :, 3:]


def neighbor_vectors(vertex, i=1):
    delta_right = np.pad(vertex, ((0, 0), (i, 0), (0, 0)))[:, :-i, :] - vertex
    delta_left = np.pad(vertex, ((0, 0), (0, i), (0, 0)))[:, i:, :] - vertex
    delta_down = np.pad(vertex, ((i, 0), (0, 0), (0, 0)))[:-i, :, :] - vertex
    delta_up = np.pad(vertex, ((0, i), (0, 0), (0, 0)))[i:, :, :] - vertex
    delta_down_right = np.pad(vertex, ((i, 0), (i, 0), (0, 0)))[:-i, :-i, :] - vertex
    delta_up_left = np.pad(vertex, ((0, i), (0, i), (0, 0)))[i:, i:, :] - vertex
    delta_up_right = np.pad(vertex, ((0, i), (i, 0), (0, 0)))[i:, :-i, :] - vertex
    delta_down_left = np.pad(vertex, ((i, 0), (0, i), (0, 0)))[:-i, i:, :] - vertex

    vectors = np.concatenate((delta_up_left, delta_left, delta_down_left, delta_down,
                              delta_down_right, delta_right, delta_up_right, delta_up), axis=2)

    return vectors


def convert2training_tensor(path, k):
    if not os.path.exists(str(path)):
        raise FileNotFoundError
    if not os.path.exists(str(path / "tensor")):
        os.makedirs(str(path / "tensor"))

    depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
    gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
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
        # move all the vertex as close to original point as possible,
        vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / vertex[:, :, :1][
            ~mask].max()
        vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / vertex[:, :, 1:2][
            ~mask].max()
        vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / vertex[:, :, 2:3][
            ~mask].max()

        # calculate delta x, y, z of between each point and its neighbors
        vectors = neighbor_vectors_k(vertex, k)
        vectors[mask] = 0

        input_torch = torch.from_numpy(vectors)  # (depth, dtype=torch.float)
        input_torch = input_torch.permute(2, 0, 1)

        gt = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
        gt = mu.normal2RGB(gt)
        gt = (gt).astype(np.float32)
        gt_torch = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        gt_torch = gt_torch.permute(2, 0, 1)

        # save tensors
        torch.save(input_torch, str(path / "tensor" / f"{str(item).zfill(5)}_input_{k}.pt"))
        torch.save(gt_torch, str(path / "tensor" / f"{str(item).zfill(5)}_gt_{k}.pt"))
        print(f'File {item} converted to tensor.')


if __name__ == '__main__':
    # # noisy a folder test code
    # noisy_a_folder(config.synthetic_captured_data, config.synthetic_captured_data_noise)
    for folder in ["selval", "test", "train"]:
        original_folder = config.synthetic_data / folder
        noisy_folder = config.synthetic_data_noise / folder
        noisy_a_folder(original_folder, noisy_folder)

    # # noisy test code
    # f = open(config.synthetic_captured_data / "00000.data0.json")
    # data = json.load(f)
    # f.close()
    #
    # img = file_io.load_scaled16bitImage(str(config.synthetic_captured_data / "00000.depth0.png"),
    #                                     data['minDepth'], data['maxDepth'])
    # noisy_img = noisy(img)
    #
    # cv.imshow('noise_depth', noisy_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
