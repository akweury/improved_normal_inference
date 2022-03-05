"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import numpy as np
import cv2 as cv

import help_funs.mu
from help_funs import file_io, mu
import config



def median_filter(depth):
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


def pred_filter(img, pred_img):
    h, w = img.shape[:2]
    mask = ~(img == 0)
    img[mask] = pred_img[mask]

    return img


if __name__ == '__main__':
    # img_path = str(config.dataset / "gradual2d_512.png")  # basic image
    # img_path = str(config.data_3 / "00003.depth0_gt.png")  # basic image

    path = config.synthetic_data_noise / "train"
    idx = 4
    data, depth, depth_noise = file_io.load_single_data(path, idx)

    # original image
    depth_8bit = mu.normalize2_8bit(depth, data)
    normal_knn, normal_knn_8bit = help_funs.mu.depth2normal(depth, 2, data['K'], data['R'], data['t'])

    # noised image
    depth_noise_8bit = mu.normalize2_8bit(depth_noise, data)
    normal_noise_knn, normal_noise_knn_8bit = help_funs.mu.depth2normal(depth_noise, 2, data['K'], data['R'], data['t'])

    # # denoised image
    # # denoise_path = str(config.ws_path / "pncnn" / "output" / "epoch_40_synthetic_selval_output" / "00002.png")
    # # depth_denoise_8bit = depth16bit_2_8bit(denoise_path)
    # # depth_denoise_8bit = pred_filter(depth_8bit, depth_denoise_8bit)
    #
    # # difference between original and denoised image
    # depth_difference = depth_8bit - depth_denoise_8bit
    #
    # diff_noise = np.sum(depth_8bit).astype(np.int64) - np.sum(depth_noise_8bit).astype(np.int64)
    # diff_denoise = np.sum(depth_8bit).astype(np.int64) - np.sum(depth_denoise_8bit).astype(np.int64)
    #
    # print(diff_noise)
    # print(diff_denoise)

    mu.addText(depth_8bit, "Original")
    mu.addText(depth_noise_8bit, 'Adding Noise')
    final_output_1 = cv.hconcat([depth_8bit, depth_noise_8bit])
    cv.imshow("original_noise", final_output_1)
    cv.waitKey(0)
    cv.destroyAllWindows()

    mu.addText(normal_knn_8bit, 'Normal (Ground Truth)')
    mu.addText(normal_noise_knn_8bit, "Original")
    final_output_2 = cv.hconcat([normal_knn_8bit, normal_noise_knn_8bit])

    cv.imshow("original_noise_2", final_output_2)
    cv.waitKey(0)
    cv.destroyAllWindows()
