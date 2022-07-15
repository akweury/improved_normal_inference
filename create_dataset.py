import argparse
import glob
import json
import os
import shutil

import numpy as np
import torch
import cv2 as cv

import config
from help_funs import file_io, mu
from help_funs.data_preprocess import noisy_a_folder, vectex_normalization

parser = argparse.ArgumentParser(description='Eval')

# Machine selection
parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                    help="loading dataset from local or dfki machine")
parser.add_argument('--data', type=str, default="synthetic",
                    help="choose dataset")
parser.add_argument('--max_k', type=str, default="0",
                    help="loading dataset from local or dfki machine")
parser.add_argument('--noise', type=str, default="true",
                    help="add noise to the dataset")
parser.add_argument('--clear', type=str, default="false",
                    help="flag that is used to clear old dataset")

args = parser.parse_args()


def convert2training_tensor(path, k, output_type='normal'):
    if not os.path.exists(str(path)):
        raise FileNotFoundError
    if not os.path.exists(str(path / "tensor")):
        os.makedirs(str(path / "tensor"))
    if output_type == "normal_noise":
        depth_files = np.array(sorted(glob.glob(str(path / "*depth0_noise.png"), recursive=True)))
        depth_gt_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
    elif output_type == "normal":
        depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
    else:
        raise ValueError("output_file is not supported. change it in args.json")

    gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
    img_files = np.array(sorted(glob.glob(str(path / "*image0.png"), recursive=True)))

    for item in range(len(data_files)):
        if os.path.exists(str(path / "tensor" / f"{str(item).zfill(5)}_{k}_{output_type}.pth.tar")):
            continue

        # input vertex
        f = open(data_files[item])
        data = json.load(f)
        f.close()
        light_pos = np.array(data['lightPos'])
        # light_pos = np.array(data['R']) @ light_pos.reshape(3, 1) - np.array(data['t']).reshape(3, 1)  # synthetic
        light_pos = np.array(data['R']) @ (light_pos.reshape(3, 1) + np.array(data['t']).reshape(3, 1))  # real

        light_pos = light_pos.reshape(3)
        depth = file_io.load_scaled16bitImage(depth_files[item],
                                              data['minDepth'],
                                              data['maxDepth'])
        if output_type == "normal_noise":
            depth_gt = file_io.load_scaled16bitImage(depth_gt_files[item],
                                                     data['minDepth'],
                                                     data['maxDepth'])
        else:
            depth_gt = depth
        mask = depth.sum(axis=2) == 0
        mask_gt = depth_gt.sum(axis=2) == 0

        img = file_io.load_16bitImage(img_files[item])
        img[mask_gt] = 0
        data['R'], data['t'] = np.identity(3), np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())
        vertex_gt = mu.depth2vertex(torch.tensor(depth_gt).permute(2, 0, 1),
                                    torch.tensor(data['K']),
                                    torch.tensor(data['R']).float(),
                                    torch.tensor(data['t']).float())
        vertex[mask] = 0
        vertex_gt[mask_gt] = 0
        vertex_norm, scale_factors, shift_vector = vectex_normalization(vertex, mask)

        vertex_gt_norm, scale_factors, shift_vector = vectex_normalization(vertex_gt, mask_gt)

        # gt normal
        gt_normal = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
        gt_normal = gt_normal / (
                np.linalg.norm(gt_normal, ord=2, axis=2, keepdims=True) + 1e-20)  # normal normalization

        # light
        # light_pos = (data['lightPos'] - shift_vector) / scale_factors

        lig_pos_norm = torch.from_numpy((light_pos - shift_vector) / scale_factors)
        light_direction = mu.vertex2light_direction(vertex, light_pos)
        light_direction_gt = mu.vertex2light_direction(vertex_gt, light_pos)
        light_direction_gt[mask_gt] = 0
        light_direction[mask] = 0

        # albedo
        G = np.sum(gt_normal * light_direction_gt, axis=-1)
        G[mask_gt] = 0
        albedo = img / (G)
        albedo[np.isnan(albedo)] = 0

        # mu.show_images(img, "img")
        # albedo_img = np.uint8(albedo)
        # albedo_img = cv.normalize(albedo_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # mu.show_images(albedo_img, "a")

        ###################  target  ####################
        target = np.c_[
            gt_normal,  # 0,1,2
            np.sum(gt_normal * light_direction_gt, axis=-1, keepdims=True),  # 3
            np.expand_dims(img, axis=2),  # 4
            light_direction_gt  # 5,6,7
        ]

        ################### input #######################
        vertex_norm[mask] = 0
        vectors = np.c_[vertex_norm,  # 0,1,2
                        np.expand_dims(img, axis=2),  # 3
                        light_direction]  # 4,5,6

        # convert to tensor
        input_tensor = torch.from_numpy(vectors.astype(np.float32)).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(target).permute(2, 0, 1)

        # save tensors
        training_case = {'input_tensor': input_tensor,
                         'gt_tensor': gt_tensor,
                         'scale_factors': scale_factors,
                         'light_source': lig_pos_norm,
                         'K': data['K'],
                         'R': data['R'],
                         't': data['t'],
                         'minDepth': data['minDepth'],
                         'maxDepth': data['maxDepth'],
                         }
        torch.save(training_case, str(path / "tensor" / f"{str(item).zfill(5)}_{k}_{output_type}.pth.tar"))
        print(f'File {item + 1}/{len(data_files)} converted to tensor. K = {k}')


if args.data in ["synthetic128", "synthetic256", "synthetic512", "synthetic64"]:
    for folder in ["selval", "test", "train"]:

        if args.machine == "remote":
            original_folder = config.synthetic_data_dfki / args.data / folder
            dataset_folder = config.synthetic_data_noise_dfki / args.data / folder
        elif args.machine == 'local':
            original_folder = config.synthetic_data / args.data / folder
            dataset_folder = config.synthetic_data_noise_local / args.data / folder
        else:
            raise ValueError

        if args.noise == "true":
            noisy_a_folder(original_folder, dataset_folder)
            if not os.path.exists(str(dataset_folder / "tensor")):
                os.makedirs(str(dataset_folder / "tensor"))
            if args.clear == "true":
                print("remove the old dataset...")
                shutil.rmtree(str(dataset_folder / "tensor"))
            for k in args.max_k.split(','):
                print(f"K = {k}, {dataset_folder}")
                convert2training_tensor(dataset_folder, k=int(k), output_type="normal_noise")
        else:
            if not os.path.exists(str(original_folder / "tensor")):
                os.makedirs(str(original_folder / "tensor"))
            if args.clear == "true":
                print("remove the old dataset...")
                shutil.rmtree(str(original_folder / "tensor"))
            for k in args.max_k.split(','):
                print(f"K = {k}, {original_folder}")
                convert2training_tensor(original_folder, k=int(k), output_type="normal")


elif args.data == "real":
    dataset_folder = config.real_data
    for k in args.max_k.split(','):
        print(f"K = {k}, {dataset_folder}")
        convert2training_tensor(dataset_folder, k=int(k), output_type="normal")
else:
    raise ValueError
