import os
import numpy as np
import cv2 as cv

import config
import chart

data_type = "synthetic"


# data_type = "real"


def mse(img_1, img_2):
    """

    :param img_1: np_array of image with shape height*width*channel
    :param img_2: np_array of image with shape height*width*channel
    :return: mse error of two images in range [0,1]
    """
    if img_1.shape != img_2.shape:
        print("MSE Error: img 1 and img 2 do not have the same shape!")
        raise ValueError

    h, w, c = img_1.shape
    diff = np.sum(np.abs(img_1 - img_2)) / (h * w * c * 255)
    return diff


diffs = np.zeros((3, 100 - 3))
for i in range(3):
    gt = config.get_file_name(i, "normal", data_type)
    if os.path.exists(gt):
        normal_gt = cv.imread(gt)
    else:
        continue

    for j in range(3, 100):
        knn = config.get_output_file_name(i, "knn", j)
        if os.path.exists(knn):
            normal_knn = cv.imread(knn)
            diffs[i, j - 3] = mse(normal_gt, normal_knn)
        else:
            diffs[i, j - 3] = 0
            continue
    print(f"data {i}: mse {diffs}")
chart.line_chart(diffs[:, :np.where(diffs[0, :] == 0)[0][0]-1],
                 title="knn performance",
                 x_scale=[3, 1],
                 y_scale=[1, 1])

