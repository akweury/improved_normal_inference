import os.path

import numpy as np
import torch
import json

from improved_normal_inference.file_io import writePLY, load_8bitImage, load_24bitNormal, load_scaled16bitImage, \
    depth2vertex, cameraVisualization, lightVisualization, get_file_name, get_output_file_name


def generate_ply(file_idx, normal, method, data_type, param=0):
    json_file = get_file_name(file_idx, "data", data_type)

    f = open(json_file)
    data = json.load(f)
    f.close()

    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    image = load_8bitImage(get_file_name(file_idx, "image", data_type))
    depth = load_scaled16bitImage(get_file_name(file_idx, "depth", data_type),
                                  data['minDepth'],
                                  data['maxDepth'])
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(data['K']),
                          torch.tensor(data['R']).float(),
                          torch.tensor(data['t']).float())

    # mask = np.sum(np.abs(vertex), axis=2) != 0
    mask = np.sum(np.abs(normal), axis=2) != 0
    cameraPoints = cameraVisualization()
    for i in range(len(cameraPoints)):
        cameraPoints[i] = np.array(data['R']).transpose() @ cameraPoints[i] / 4 - np.array(
            data['R']).transpose() @ np.array(data['t'])
    lightPoints = lightVisualization()
    for i in range(len(lightPoints)):
        lightPoints[i] = lightPoints[i] / 8 + np.array(data['lightPos'])

    point_cloud_file = get_output_file_name(file_idx, file_type="pointcloud", method=method, data_type=data_type, param=param)
    writePLY(vertex, normal, image, mask, point_cloud_file,
             cameraPoints=cameraPoints,
             lightPoints=lightPoints)
    print(f"Saved {file_idx}-th ply file {point_cloud_file}...")


if __name__ == '__main__':
    file_idx = 0
    data_type = "synthetic_basic"
    method = "gt"
    normal = load_24bitNormal(get_output_file_name(file_idx, "normal", "gt", data_type))
    generate_ply(file_idx, normal, method=method, data_type=data_type)
