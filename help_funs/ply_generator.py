import json

import numpy as np
import torch

import config
from help_funs.file_io import writePLY, load_24bitNormal, load_scaled16bitImage, load_16bitImage, \
    get_file_name
from help_funs.mu import lightVisualization, cameraVisualization, depth2vertex


def generate_ply(file_idx, normal, data_path, param=0):
    image_file, ply_file, json_file, depth_file, depth_noise_file, normal_file = get_file_name(file_idx, data_path)

    f = open(json_file)
    data = json.load(f)
    f.close()

    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    image = load_16bitImage(image_file)
    depth = load_scaled16bitImage(depth_file, data['minDepth'], data['maxDepth'])
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(data['K']),
                          torch.tensor(data['R']).float(),
                          torch.tensor(data['t']).float())

    mask = np.sum(np.abs(normal), axis=2) != 0
    cameraPoints = cameraVisualization()
    for i in range(len(cameraPoints)):
        cameraPoints[i] = np.array(data['R']).transpose() @ cameraPoints[i] / 4 - np.array(
            data['R']).transpose() @ np.array(data['t'])
    lightPoints = lightVisualization()
    for i in range(len(lightPoints)):
        lightPoints[i] = lightPoints[i] / 8 + np.array(data['lightPos0'])

        # lightPoints[i] = np.array(data['R']) @ (lightPoints[i] - np.array(data['t']))
    writePLY(vertex, normal, image, mask, ply_file,
             cameraPoints=cameraPoints,
             lightPoints=lightPoints)
    print(f"Saved {file_idx}-th ply file {ply_file}...")


if __name__ == '__main__':
    file_idx = 3
    data_path = config.synthetic_data / "synthetic512" / 'train'

    image_file, ply_file, json_file, depth_file, depth_noise_file, normal_file = get_file_name(file_idx, data_path)
    normal = load_24bitNormal(normal_file)
    generate_ply(file_idx, normal, data_path=data_path)
