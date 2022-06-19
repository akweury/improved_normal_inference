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
    depth_file = str(config.dataset / "data_synthetic" / "train" / "00042.depth0.png")
    data_file = str(config.dataset / "data_synthetic" / "train" / "00042.data0.json")

    img_list = []
    for noised_factor in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
        # input vertex
        f = open(data_file)
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(depth_file,
                                              data['minDepth'],
                                              data['maxDepth'])

        mask = depth.sum(axis=2) == 0

        depth_noised, noised_factor = data_preprocess.noisy_1channel(depth, noised_factor)
        img = file_io.save_scaled16bitImage(depth_noised,
                                            str(folder_path / f"depth_noised_f_{noised_factor}.png"),
                                            data['minDepth'], data['maxDepth'])
        img = img / 65535
        img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        img = cv.merge((img, img, img))
        img_list.append(img)
    output = cv.hconcat(img_list)
    cv.imwrite(str(folder_path / f"add_noise_depth.png"), output)
