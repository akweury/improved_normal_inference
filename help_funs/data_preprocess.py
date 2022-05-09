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
from pncnn.utils import args_parser


def noisy_1channel(img, noise_templete):
    img = img.reshape(512, 512)
    h, w = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise_img = img * noise * noise_templete
    return noise_img


def noisy(img):
    h, w, c = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise = noise.reshape(h, w, 1)
    noise = np.repeat(noise, c, axis=-1)
    noise_img = img * noise
    return noise_img


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, input_tensor, device):
    model.eval()  # Swith to evaluate mode

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)
        torch.cuda.synchronize()

        # Forward Pass
        output = model(input_tensor)

        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)[:, :, :3]
        output = output.to('cpu').numpy()

    return output


def noisy_a_folder(folder_path, output_path):
    # get noise model
    noise_model_path = config.ws_path / "noise_net" / "trained_model" / "output_2022-05-09_16_41_36" / "checkpoint-98.pth.tar"

    # load model
    checkpoint = torch.load(noise_model_path)

    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print('- Checkpoint was loaded successfully.')

    # Compare the checkpoint args with the json file in case I wanted to change some args
    # args_parser.compare_args_w_json(args, exp_dir, start_epoch + 1)
    args.evaluate = noise_model_path

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))

    model = checkpoint['model'].to(device)
    args_parser.print_args(args)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx in range(1000):
        image_file, ply_file, json_file, depth_gt_file, _, normal_file = file_io.get_file_name(idx, folder_path)
        if path.exists(image_file):
            f = open(json_file)
            data = json.load(f)
            depth = file_io.load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])

            # get noise mask
            img = np.expand_dims(file_io.load_16bitImage(image_file), axis=2)

            input_tensor = torch.from_numpy(img.astype(np.float32))  # (depth, dtype=torch.float)
            input_tensor = input_tensor.permute(2, 0, 1)

            img_noise = evaluate_epoch(model, input_tensor, device)
            noise_mask = img_noise.sum(axis=2) == 0
            # add noise
            depth[noise_mask] = 0

            # save files to the new folders
            file_io.save_scaled16bitImage(depth,
                                          str(output_path / (str(idx).zfill(5) + ".depth0_noise.png")),
                                          data['minDepth'], data['maxDepth'])
            img_noise = mu.normalise216bitImage(img_noise)
            file_io.save_16bitImage(img_noise, str(output_path / (str(idx).zfill(5) + ".image0_noise.png")))
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
    if k == 1:
        # use vertex itself as the input of network
        return vertex
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


def convert2training_tensor(path, k, output_type='normal'):
    if not os.path.exists(str(path)):
        raise FileNotFoundError
    if not os.path.exists(str(path / "tensor")):
        os.makedirs(str(path / "tensor"))
    if output_type == "normal":
        depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
    elif output_type == "normal_noise":
        depth_files = np.array(sorted(glob.glob(str(path / "*depth0_noise.png"), recursive=True)))
    else:
        raise ValueError("output_file is not supported. change it in args.json")
    gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
    img_files = np.array(sorted(glob.glob(str(path / "*image0.png"), recursive=True)))
    for item in range(len(data_files)):
        f = open(data_files[item])
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(depth_files[item],
                                              data['minDepth'],
                                              data['maxDepth'])
        img = file_io.load_16bitImage(img_files[item])
        data['R'] = np.identity(3)
        data['t'] = np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())
        mask = vertex.sum(axis=2) == 0
        # move all the vertex as close to original point as possible, and noramlized all the vertex
        range_0 = vertex[:, :, :1][~mask].max() - vertex[:, :, :1][~mask].min()
        range_1 = vertex[:, :, 1:2][~mask].max() - vertex[:, :, 1:2][~mask].min()
        range_2 = vertex[:, :, 2:3][~mask].max() - vertex[:, :, 2:3][~mask].min()

        vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / range_0
        vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / range_1
        vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / range_2

        # calculate delta x, y, z of between each point and its neighbors
        if k >= 2:
            vectors = neighbor_vectors_k(vertex, k)
        # case of ng
        elif k == 0:
            vectors = np.c_[vertex, np.expand_dims(img, axis=2)]
        elif k == 1:
            vectors = vertex
        else:
            raise ValueError

        vectors[mask] = 0

        input_torch = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
        input_torch = input_torch.permute(2, 0, 1)

        gt = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
        if output_type == 'rgb':
            gt = mu.normal2RGB(gt)
        gt = gt.astype(np.float32)
        gt_torch = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        gt_torch = gt_torch.permute(2, 0, 1)

        # save tensors
        torch.save(input_torch, str(path / "tensor" / f"{str(item).zfill(5)}_input_{k}_{output_type}.pt"))
        torch.save(gt_torch, str(path / "tensor" / f"{str(item).zfill(5)}_gt_{k}_{output_type}.pt"))
        print(f'File {item} converted to tensor.')


def high_pass_filter(img, threshold=20):
    img[img < threshold] = 0
    return img


def convert2training_tensor_noise(path):
    k = 0
    output_type = 'noise'
    if not os.path.exists(str(path)):
        raise FileNotFoundError
    if not os.path.exists(str(path / "tensor")):
        os.makedirs(str(path / "tensor"))

    img_files = np.array(sorted(glob.glob(str(path / "*image?.png"), recursive=True)))
    gt_files = np.array(sorted(glob.glob(str(path / "*normal?.png"), recursive=True)))
    for item in range(len(img_files)):
        img = file_io.load_16bitImage(img_files[item])
        idx = img_files[item].split(".")[-2][-1]
        prefix_idx = img_files[item].replace("\\", ".").split(".")[-3]
        img = high_pass_filter(img)
        img = np.expand_dims(img, axis=2)
        gt = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
        gt = gt.sum(axis=2, keepdims=True) != 0
        gt = gt.astype(np.float32)
        gt_torch = torch.from_numpy(gt)  # tensor(gt, dtype=torch.float)
        gt_torch = gt_torch.permute(2, 0, 1)

        input_torch = torch.from_numpy(img.astype(np.float32))  # (depth, dtype=torch.float)
        input_torch = input_torch.permute(2, 0, 1)
        # save tensors
        torch.save(input_torch, str(path / "tensor" / f"{str(item).zfill(5)}_input_{k}_{output_type}.pt"))
        torch.save(gt_torch, str(path / "tensor" / f"{str(item).zfill(5)}_gt_{k}_{output_type}.pt"))
        print(f'File {item} converted to tensor.')

        img_16bit = mu.normalise216bitImage(img)
        file_io.save_16bitImage(img_16bit, str(path / (prefix_idx + f".image{idx}_lowpass.png")))


# def convert2training_tensor2(path, k, input_size=1000):
#     if not os.path.exists(str(path)):
#         raise FileNotFoundError
#     if not os.path.exists(str(path / "tensor")):
#         os.makedirs(str(path / "tensor"))
#
#     depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
#     gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
#     data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
#     for item in range(len(data_files)):
#         f = open(data_files[item])
#         data = json.load(f)
#         f.close()
#
#         depth = file_io.load_scaled16bitImage(depth_files[item],
#                                               data['minDepth'],
#                                               data['maxDepth'])
#         data['R'] = np.identity(3)
#         data['t'] = np.zeros(3)
#         vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
#                                  torch.tensor(data['K']),
#                                  torch.tensor(data['R']).float(),
#                                  torch.tensor(data['t']).float())
#         mask = vertex.sum(axis=2) == 0
#         # move all the vertex as close to original point as possible,
#         vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / vertex[:, :, :1][
#             ~mask].max()
#         vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / vertex[:, :, 1:2][
#             ~mask].max()
#         vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / vertex[:, :, 2:3][
#             ~mask].max()
#
#         # gt
#         gt = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
#         gt[mask] = 0
#         gt = gt.reshape(-1, 3)
#         gt = gt[gt.sum(axis=1) != 0]
#         # calculate delta x, y, z of between each point and its neighbors
#         vectors = neighbor_vectors_k(vertex, k)
#         vectors[mask] = 0
#         vectors = vectors[vectors.sum(axis=2) != 0]
#         vectors = vectors.reshape(-1, (((k - 1) * 2 + 1) ** 2 - 1), 3)
#         vertex = vertex.reshape(-1, 3)
#         vertex = vertex[vertex.sum(axis=1) != 0]
#         assert gt.shape[0] == vectors.shape[0] == vertex.shape[0]
#
#         # align the input to the standard shape
#         random_idx = np.random.choice(vectors.shape[0], input_size)
#         vectors = vectors[random_idx]
#         vertex = vertex[random_idx]
#         gt = gt[random_idx]
#
#         normals, idx_gt, error = mu.generate_normals_all(vectors, vertex, gt)
#
#         # visualise to check if it is correct
#         best_error_avg = np.average(error)
#         error_validition = mu.angle_between(normals, gt)
#         error_avg = np.average(error_validition)
#
#         # save vectors and idx_gt
#         input_torch = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
#         input_torch = input_torch.permute(0, 2, 1)
#
#         idx_gt = (idx_gt).astype(np.float32)
#         gt_torch = torch.from_numpy(idx_gt)  # tensor(gt, dtype=torch.float)
#         gt_torch = gt_torch.permute(1, 0)
#
#         # save tensors
#         torch.save(input_torch, str(path / "tensor" / f"{str(item).zfill(5)}_input_x.pt"))
#         torch.save(gt_torch, str(path / "tensor" / f"{str(item).zfill(5)}_gt_x.pt"))
#         print(f'File {item} converted to tensor.')


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