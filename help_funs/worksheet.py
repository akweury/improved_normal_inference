import json

import cv2 as cv
import numpy as np
import torch

import config
from help_funs import file_io, mu, data_preprocess


def test_torch():
    # depth_file = str(config.ws_path / "a.png")
    data_file = str(config.synthetic_data / "train" / '00102.normal0.png')
    normal = file_io.load_24bitNormal(data_file).astype(np.float32)
    data_tensor = torch.from_numpy(normal).permute(2, 0, 1)
    img, img_extended = mu.hpf_torch(data_tensor)
    cv.imwrite(str(config.ws_path / f"aa.png"), img.numpy())


#
#
# def test_png():
#     depth_file = str(
#         config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / "train_epoch_199_0_loss_0.16585270.png")
#     img = mu.hpf(depth_file)
#     cv.imwrite(str(config.ws_path / "degares" / "output" / "output_2022-05-24_15_14_04" / f"aa.png"), img)


def print_cuda_info():
    print(f"Cuda is available: {torch.cuda.is_available()}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


if __name__ == '__main__':
    # test_torch()
    # print_cuda_info()
    folder_path = config.paper_pic
    depth_file = str(config.dataset / "data_synthetic" / "test" / "00601.depth0.png")
    data_file = str(config.dataset / "data_synthetic" / "test" / "00601.data0.json")
    img_file = str(config.dataset / "data_synthetic" / "test" / "00601.image0.png")

    # input vertex
    f = open(data_file)
    data = json.load(f)
    f.close()

    depth = file_io.load_scaled16bitImage(depth_file,
                                          data['minDepth'],
                                          data['maxDepth'])

    mask = depth.sum(axis=2) == 0

    img = file_io.load_16bitImage(img_file)
    img[mask] = 0
    data['R'], data['t'] = np.identity(3), np.zeros(3)
    vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                             torch.tensor(data['K']),
                             torch.tensor(data['R']).float(),
                             torch.tensor(data['t']).float())

    vertex, scale_factors, shift_vector = data_preprocess.vectex_normalization(vertex, mask)
    cv.imwrite(str(folder_path / f"fancy_eval_point_cloud.png"), mu.visual_vertex(vertex, ""))
