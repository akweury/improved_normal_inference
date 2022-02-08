import os.path

import numpy as np
import torch
import json

from improved_normal_inference.file_io import writePLY, load_8bitImage, load_24bitNormal, load_scaled16bitImage, \
    depth2vertex, cameraVisualization, lightVisualization, get_file_name, get_output_file_name

data_type = "synthetic_basic"

# ----------------------------------------------------------------------------------
file_idx = 0
file = get_file_name(file_idx, "data", data_type)

while os.path.isfile(file):

    f = open(file)
    data = json.load(f)
    f.close()

    image = load_8bitImage(get_file_name(file_idx, "image", data_type))
    depth = load_scaled16bitImage(get_file_name(file_idx, "depth", data_type),
                                  data['minDepth'],
                                  data['maxDepth'])
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(data['K']),
                          torch.tensor(data['R']),
                          torch.tensor(data['t']))
    normal = load_24bitNormal(get_output_file_name(file_idx, "normal", "gt", data_type), torch.tensor(data['R']))
    # mask = np.sum(np.abs(vertex), axis=2) != 0
    mask = np.sum(np.abs(depth), axis=2) != 0
    cameraPoints = cameraVisualization()
    for i in range(len(cameraPoints)):
        cameraPoints[i] = np.array(data['R']).transpose() @ cameraPoints[i] / 4 - np.array(
            data['R']).transpose() @ np.array(data['t'])
    lightPoints = lightVisualization()
    for i in range(len(lightPoints)):
        lightPoints[i] = lightPoints[i] / 8 + np.array(data['lightPos'])

    point_cloud_file = get_file_name(file_idx, "pointcloud", data_type)
    writePLY(vertex, normal, image, mask, point_cloud_file,
             cameraPoints=cameraPoints,
             lightPoints=lightPoints)
    print(f"Saved {file_idx}-th ply file {point_cloud_file}...")
    file_idx += 1
    file = get_file_name(file_idx, "data", data_type)
