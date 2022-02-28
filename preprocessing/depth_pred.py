"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import json
import numpy as np
import cv2 as cv
from PIL import Image
from improved_normal_inference.help_funs import file_io, mu
from improved_normal_inference import config
from improved_normal_inference.preprocessing import data_preprocess


def median_filter(depth):
    # TODO: pred the depth of each pixel
    # use a simple CNN to predict depth in each pixel
    padding = 2  # 2 optimal

    # add padding
    depth_padded = np.expand_dims(mu.copy_make_border(depth, padding * 2), axis=2)
    h, w, c = depth_padded.shape

    mask = (depth_padded == 0)

    # predict
    for i in range(padding, h - padding):
        for j in range(padding, w - padding):
            if mask[i, j]:
                neighbor = depth_padded[i - padding:i + padding + 1, j - padding:j + padding + 1].flatten()
                neighbor = np.delete(neighbor, np.floor((padding * 2 + 1) ** 2 / 2).astype(np.int32))
                # depth_padded[i, j] = mu.bi_interpolation(lower_left, lower_right, upper_left, upper_right, 0.5, 0.5)
                depth_padded[i, j] = np.median(neighbor)

    # remove the padding
    pred_depth = depth_padded[padding:-padding, padding:-padding]

    return pred_depth.reshape(512, 512)

if __name__ == '__main__':
    # img_path = str(config.dataset / "gradual2d_512.png")  # basic image
    # img_path = str(config.data_3 / "00003.depth0_gt.png")  # basic image
    img_path = str(config.synthetic_data / "TrainData" / "00000.depth0.png")  # synthetic image

    original_16bit = file_io.load_16bitImage(img_path)
    original_noise_16bit = data_preprocess.noisy_1channel(original_16bit)

    original = cv.normalize(original_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    original_noise = cv.normalize(original_noise_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    original_denoise = median_filter(original_noise)
    difference_16bit = original - original_denoise
    difference = cv.normalize(difference_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    diff_noise = np.sum(original).astype(np.int64) - np.sum(original_noise).astype(np.int64)
    diff_denoise = np.sum(original).astype(np.int64) - np.sum(original_denoise).astype(np.int64)

    print(diff_noise)
    print(diff_denoise)

    final_output = cv.hconcat([original, original_noise, original_denoise, difference])

    cv.imshow("original_noise", final_output)
    cv.waitKey(0)
    cv.destroyAllWindows()
