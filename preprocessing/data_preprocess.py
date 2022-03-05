import os
from os import path
from pathlib import Path
import json
import numpy as np
import cv2 as cv
import shutil

import config
from help_funs import file_io


def noisy_1channel(img):
    img = img.reshape(512, 512)
    h, w = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise_img = img * noise
    return noise_img


def noisy(img):
    h, w, c = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise = noise.reshape(h, w, 1)
    noise = np.repeat(noise, c, axis=-1)
    noise_img = img * noise
    return noise_img


def noisy_a_folder(folder_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx in range(500):
        image_file, ply_file, json_file, depth_gt_file,_, normal_file = file_io.get_file_name(idx, folder_path)
        if path.exists(image_file):
            f = open(json_file)
            data = json.load(f)
            # normal_gt = cv.imread(normal_file)
            depth = file_io.load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])

            # add noise
            noise_depth = noisy_1channel(depth)

            # save files to the new folders
            file_io.save_scaled16bitImage(noise_depth,
                                          str(output_path / (str(idx).zfill(5) + ".depth0_noise.png")),
                                          data['minDepth'], data['maxDepth'])
            shutil.copyfile(depth_gt_file, str(output_path / (str(idx).zfill(5) + ".depth0.png")))
            shutil.copyfile(image_file, str(output_path / (str(idx).zfill(5) + ".image0.png")))
            shutil.copyfile(json_file, str(output_path / (str(idx).zfill(5) + ".data0.json")))
            shutil.copyfile(normal_file, str(output_path / (str(idx).zfill(5) + ".normal0.png")))

            print(f'File {idx} added noise.')


if __name__ == '__main__':
    # # noisy a folder test code
    # noisy_a_folder(config.synthetic_captured_data, config.synthetic_captured_data_noise)
    for folder in ["selval","test", "train" ]:
        original_folder = config.synthetic_data / folder
        noisy_folder = config.synthetic_data_noise / folder
        noisy_a_folder(original_folder, noisy_folder)

    # # noisy test code
    # f = open(config.synthetic_captured_data / "00000.data0.json")
    # data = json.load(f)
    # f.close()
    #
    # img = file_io.load_scaled16bitImage(str(config.synthetic_captured_data / "00000.depth0.png"),
    #                                     data['minDepth'], data['maxDepth'])
    # noisy_img = noisy(img)
    #
    # cv.imshow('noise_depth', noisy_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
