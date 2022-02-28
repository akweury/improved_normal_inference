import numpy as np
import torch
import json

from improved_normal_inference.help_funs.file_io import writePLY, load_8bitImage, load_24bitNormal, load_scaled16bitImage, \
    get_file_name, get_output_file_name
from improved_normal_inference.help_funs.mu import lightVisualization, cameraVisualization, depth2vertex
from improved_normal_inference import config

def generate_ply(file_idx, normal, data_path, param=0):
    image_file, ply_file, json_file, depth_file, normal_file = get_file_name(file_idx, data_path)

    f = open(json_file)
    data = json.load(f)
    f.close()

    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    image = load_8bitImage(image_file)
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
        lightPoints[i] = lightPoints[i] / 8 + np.array(data['lightPos'])

    writePLY(vertex, normal, image, mask, ply_file,
             cameraPoints=cameraPoints,
             lightPoints=lightPoints)
    print(f"Saved {file_idx}-th ply file {ply_file}...")


if __name__ == '__main__':
    file_idx = 0
    data_path = config.synthetic_data / 'train'

    image_file, ply_file, json_file, depth_file, normal_file = get_file_name(file_idx, data_path)
    normal = load_24bitNormal(normal_file)
    generate_ply(file_idx, normal, data_path=data_path)
