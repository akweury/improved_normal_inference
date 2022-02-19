import PIL.Image
import cv2
import numpy as np
import torch
from PIL import Image

from improved_normal_inference.config import synthetic_basic_data, synthetic_captured_data, real_data, output_path


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
                               f"{' '.join(map(str, image[i, j, :].repeat(3).astype(np.int32)))}\n")
    if cameraPoints is not None:
        for i in range(len(cameraPoints)):
            ply_file.write(f"{' '.join(map(str, cameraPoints[i]))} 0.0 0.0 0.0 255 0 0\n")
    if lightPoints is not None:
        for i in range(len(lightPoints)):
            ply_file.write(f"{' '.join(map(str, lightPoints[i]))} 0.0 0.0 0.0 255 255 0\n")

    ply_file.close()


def load_8bitImage(root):
    img = cv2.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    img = torch.tensor(img).unsqueeze(2)
    img = np.array(img)
    img[np.isnan(img)] = 0

    return img.astype(np.float32)


def load_24bitNormal(root):
    R = torch.tensor(np.identity(3)).float()

    nml = cv2.imread(root, -1)
    nml = np.array(nml, dtype=np.float32)
    nml = np.array(nml)
    nml[np.isnan(nml)] = 0
    nml = nml[:, :, [2, 1, 0]]
    mask = (nml[:, :, 0] == 0) & (nml[:, :, 1] == 0) & (nml[:, :, 2] == 0)

    nml[:, :, 0] = (~mask) * (nml[:, :, 0] / 255.0 * 2.0 - 1.0)
    nml[:, :, 1] = (~mask) * (nml[:, :, 1] / 255.0 * 2.0 - 1.0)
    nml[:, :, 2] = (~mask) * (1.0 - nml[:, :, 2] / 255.0 * 2.0)

    nml = R.transpose(0, 1) @ torch.tensor(nml).permute(2, 0, 1).reshape(3, -1)
    nml = nml.reshape(3, 512, 512).permute(1, 2, 0)

    return np.array(nml).astype(np.float32)


def load_scaled16bitImage(root, minVal, maxVal):
    img = cv2.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0

    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img)

    return img.astype(np.float32)


def save_scaled16bitImage(img, img_name, minVal, maxVal):
    img = img.reshape(512, 512)
    img[np.isnan(img)] = 0
    mask = (img == 0)

    img[~mask] = (img[~mask] - minVal) / (maxVal - minVal) * 65535
    img = np.array(img, dtype=np.int32)
    write_np2img(img, img_name)


def get_file_name(idx, data_type):
    if data_type == "synthetic_basic":
        file_path = synthetic_basic_data
        image_file = str(file_path / str(idx).zfill(5)) + ".image0Gray.png"
    elif data_type == "synthetic_captured":
        file_path = synthetic_captured_data
        image_file = str(file_path / str(idx).zfill(5)) + ".image0Gray.png"
    elif data_type == "real":
        file_path = real_data
        image_file = str(file_path / str(idx).zfill(5)) + ".image0.png"
    else:
        raise ValueError

    ply_file = str(file_path / str(idx).zfill(5)) + ".pointcloud0.ply"
    json_file = str(file_path / str(idx).zfill(5)) + f".data0.json"
    depth_file = str(file_path / str(idx).zfill(5)) + f".depth0.png"
    normal_file = str(file_path / str(idx).zfill(5)) + f".normal0.png"

    return image_file, ply_file, json_file, depth_file, normal_file


def get_output_file_name(idx, file_type=None, method=None, data_type=None, param=0):
    file_path = output_path

    if file_type in ["normal", "depth"]:
        suffix = ".png"
    elif file_type == 'pointcloud':
        suffix = ".ply"
    else:
        raise ValueError

    file_name = str(file_path / str(idx).zfill(5)) + f".{file_type}_{method}_{param}{suffix}"
    return file_name


def write_np2img(np_array, img_name):
    img = Image.fromarray(np_array.astype(np.uint32))
    img.save(img_name)
    return img
