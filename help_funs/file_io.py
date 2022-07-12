import os
import glob
import json
import cv2
import numpy as np
import torch
from PIL import Image
from pathlib import Path

import config
from help_funs import mu, chart
from help_funs.mu import scale16bitImage


def file_path_abs(folder_path, file_name):
    return str(Path(folder_path) / str(file_name))


# --------------------------------------- write ----------------------------------------------------------------

def writePLY(vertex, normal, image, mask, filename, cameraPoints=None, lightPoints=None):
    numPoints = np.sum(mask)
    if cameraPoints is not None:
        numPoints += len(cameraPoints)
    if lightPoints is not None:
        numPoints += len(lightPoints)

    ply_file = open(filename, "w")
    ply_file.write('ply\n')
    ply_file.write('format ascii 1.0\n')
    ply_file.write('element vertex ' + str(numPoints) + '\n')
    ply_file.write('property float x\n')
    ply_file.write('property float y\n')
    ply_file.write('property float z\n')
    ply_file.write('property float nx\n')
    ply_file.write('property float ny\n')
    ply_file.write('property float nz\n')
    ply_file.write('property uchar red\n')
    ply_file.write('property uchar green\n')
    ply_file.write('property uchar blue\n')
    ply_file.write('end_header\n')

    for i in range(vertex.shape[0]):
        for j in range(vertex.shape[1]):
            if mask[i, j]:
                ply_file.write(f"{' '.join(map(str, vertex[i, j, :]))} "
                               f"{' '.join(map(str, normal[i, j, :]))} "
                               f"{' '.join(map(str, image[i, j].repeat(3).astype(np.int32)))}\n")
    if cameraPoints is not None:
        for i in range(len(cameraPoints)):
            ply_file.write(f"{' '.join(map(str, cameraPoints[i]))} 0.0 0.0 0.0 255 0 0\n")
    if lightPoints is not None:
        for i in range(len(lightPoints)):
            ply_file.write(f"{' '.join(map(str, lightPoints[i]))} 0.0 0.0 0.0 255 255 0\n")

    ply_file.close()


def save_scaled16bitImage(image, img_name, minVal, maxVal):
    if len(image.shape) != 2:
        raise ValueError
    img = image.copy()
    img[np.isnan(img) != 0] = 0
    mask = (img == 0)

    img[~mask] = (img[~mask] - minVal) / (maxVal - minVal) * 65535
    img = np.array(img, dtype=np.int32)
    write_np2img(img, img_name)
    return img


def write_np2rgbimg(img, img_name):
    img = Image.fromarray(img.astype(np.uint8))
    img.save(img_name)
    return img


def write_np2img(np_array, img_name):
    img = Image.fromarray(np_array.astype(np.uint32))
    img.save(img_name)
    return img


# --------------------------------------- read ----------------------------------------------------------------

def load_24bitImage(root):
    img = cv2.imread(root, -1)
    img[np.isnan(img)] = 0

    return img


def load_16bitImage(root):
    img = cv2.imread(root, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    return img

def save_16bitImage(img, img_name):
    img = img.reshape(512, 512)
    img[np.isnan(img) != 0] = 0
    mask = (img == 0)

    img = np.array(img, dtype=np.uint16)
    cv2.imwrite(img_name, img)

    return img

def load_24bitNormal(root):
    R = torch.tensor(np.identity(3)).float()

    nml = cv2.imread(root, -1)
    nml = np.array(nml, dtype=np.float32)
    nml = np.array(nml)
    nml[np.isnan(nml)] = 0
    nml = nml[:, :, [2, 1, 0]]
    mask = (nml[:, :, 0] == 0) & (nml[:, :, 1] == 0) & (nml[:, :, 2] == 0)
    w, h = mask.shape
    nml[:, :, 0] = (~mask) * (nml[:, :, 0] / 255.0 * 2.0 - 1.0)
    nml[:, :, 1] = (~mask) * (nml[:, :, 1] / 255.0 * 2.0 - 1.0)
    nml[:, :, 2] = (~mask) * (1.0 - nml[:, :, 2] / 255.0 * 2.0)

    nml = R.transpose(0, 1) @ torch.tensor(nml).permute(2, 0, 1).reshape(3, -1)
    nml = nml.reshape(3, w, h).permute(1, 2, 0)

    return np.array(nml).astype(np.float32)


def load_scaled16bitImage(root, minVal, maxVal):
    img = cv2.imread(root, -1)
    return scale16bitImage(img, minVal, maxVal)


def load_single_data(data_path, idx):
    # get file names from dataset
    image_file, ply_file, json_file, depth_gt_file, depth_noise_file, normal_file = get_file_name(idx, data_path)

    f = open(json_file)
    data = json.load(f)

    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    depth_gt = load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])
    if data_path == config.real_data:
        depth_noise = depth_gt
    else:
        depth_noise = load_scaled16bitImage(depth_noise_file, data['minDepth'], data['maxDepth'])
    #

    normal = load_24bitNormal(normal_file)
    image = load_16bitImage(image_file)
    return data, depth_gt, depth_noise, normal, image


def get_file_name(idx, data_path):
    image_file = str(data_path / str(idx).zfill(5)) + ".image0.png"
    ply_file = str(data_path / str(idx).zfill(5)) + ".pointcloud0.ply"
    json_file = str(data_path / str(idx).zfill(5)) + f".data0.json"
    depth_gt_file = str(data_path / str(idx).zfill(5)) + f".depth0.png"
    depth_noise_file = str(data_path / str(idx).zfill(5)) + f".depth0_noise.png"
    normal_file = str(data_path / str(idx).zfill(5)) + f".normal0.png"

    return image_file, ply_file, json_file, depth_gt_file, depth_noise_file, normal_file


def get_output_file_name(idx, file_type=None, method=None, param=0):
    if file_type in ["normal", "depth"]:
        suffix = ".png"
    elif file_type == 'pointcloud':
        suffix = ".ply"
    else:
        raise ValueError

    folder_path = config.output_path
    file_name = str(idx).zfill(5) + f".{file_type}_{method}_{param}{suffix}"
    file_path = file_path_abs(folder_path, file_name)

    return file_path


def save_input():
    data_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.data0.json"
    depth_file = "D:\\TUK\\improved_normal_inference\\paper\\pic\\00440.depth0_noise.png"

    f = open(data_file)
    data = json.load(f)
    f.close()

    depth = load_scaled16bitImage(depth_file,
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

        depth = load_scaled16bitImage(depth_files[item],
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
