import cv2
import cv2 as cv
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


def write_np2img(np_array, img_name):
    img = Image.fromarray(np_array.astype(np.uint8), "RGB")
    img.save(img_name)
    return img


def depth2vertex(depth, K, R, t):
    c, h, w = depth.shape

    camOrig = -R.transpose(0, 1) @ t.unsqueeze(1)

    X = torch.arange(0, depth.size(2)).repeat(depth.size(1), 1) - K[0, 2]
    Y = torch.transpose(torch.arange(0, depth.size(1)).repeat(depth.size(2), 1), 0, 1) - K[1, 2]
    Z = torch.ones(depth.size(1), depth.size(2)) * K[0, 0]
    Dir = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), 2)

    vertex = Dir * (depth.squeeze(0) / torch.norm(Dir, dim=2)).unsqueeze(2).repeat(1, 1, 3)
    vertex = R.transpose(0, 1) @ vertex.permute(2, 0, 1).reshape(3, -1)
    vertex = camOrig.unsqueeze(1).repeat(1, h, w) + vertex.reshape(3, h, w)
    vertex = vertex.permute(1, 2, 0)
    return np.array(vertex)


def cameraVisualization():
    points = []
    for p in range(-50, 51):
        ps = float(p) / 100.0
        points.append([ps, 0.5, 0.5])
        points.append([ps, -0.5, 0.5])
        points.append([ps, 0.5, -0.5])
        points.append([ps, -0.5, -0.5])
        points.append([0.5, ps, 0.5])
        points.append([-0.5, ps, 0.5])
        points.append([0.5, ps, -0.5])
        points.append([-0.5, ps, -0.5])
        points.append([0.5, 0.5, ps])
        points.append([0.5, -0.5, ps])
        points.append([-0.5, 0.5, ps])
        points.append([-0.5, -0.5, ps])

    for p in range(-30, 31):
        ps = float(p) / 100.0
        points.append([ps, 0.3, 0.3 + 0.8])
        points.append([ps, -0.3, 0.3 + 0.8])
        points.append([ps, 0.3, -0.3 + 0.8])
        points.append([ps, -0.3, -0.3 + 0.8])
        points.append([0.3, ps, 0.3 + 0.8])
        points.append([-0.3, ps, 0.3 + 0.8])
        points.append([0.3, ps, -0.3 + 0.8])
        points.append([-0.3, ps, -0.3 + 0.8])
        points.append([0.3, 0.3, ps + 0.8])
        points.append([0.3, -0.3, ps + 0.8])
        points.append([-0.3, 0.3, ps + 0.8])
        points.append([-0.3, -0.3, ps + 0.8])

    return points


def lightVisualization():
    points = []
    for px in range(-5, 6):
        for py in range(-5, 6):
            for pz in range(-5, 6):
                p = np.array([px, py, pz]).astype(np.float32) / 10
                if np.linalg.norm(p) > 0:
                    points.append(p / np.linalg.norm(p))
    return points


def get_file_name(idx, data_type):
    if data_type == "synthetic_basic":
        file_path = synthetic_basic_data
    elif data_type == "synthetic_captured":
        file_path = synthetic_captured_data
    elif data_type == "real":
        file_path = real_data
    else:
        raise ValueError

    image_file = str(file_path / str(idx).zfill(5)) + ".image0Gray.png"
    ply_file = str(file_path / str(idx).zfill(5)) + ".pointcloud0.ply"
    json_file = str(file_path / str(idx).zfill(5)) + f".data0.json"
    depth_file = str(file_path / str(idx).zfill(5)) + f".depth0.png"
    normal_file = str(file_path / str(idx).zfill(5)) + f".normal0.png"

    return image_file, ply_file, json_file, depth_file, normal_file


def get_output_file_name(idx, file_type=None, method=None, data_type=None, param=0):
    file_path = output_path

    if file_type == "normal":
        suffix = ".png"
    elif file_type == 'pointcloud':
        suffix = ".ply"
    else:
        raise ValueError

    file_name = str(file_path / str(idx).zfill(5)) + f".{file_type}_{method}_{param}{suffix}"
    return file_name


def get_valid_pixels_idx(img):
    return np.sum(img, axis=2) != 0


def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int32(patch_width / 2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)
