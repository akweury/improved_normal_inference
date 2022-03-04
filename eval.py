"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import numpy as np
import cv2 as cv
from help_funs import file_io, mu
import config
from workspace.knn import knn_normal


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


def concat_vh(list_2d):
    # return final image
    return cv.vconcat([cv.hconcat(list_h) for list_h in list_2d])


if __name__ == '__main__':
    # img_path = str(config.dataset / "gradual2d_512.png")  # basic image
    # img_path = str(config.data_3 / "00003.depth0_gt.png")  # basic image

    # original image
    original_path = str(config.synthetic_data_noise / "train" / "00004.depth0.png")  # synthetic image
    original_16bit = file_io.load_16bitImage(original_path)
    original = cv.normalize(original_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # noised image
    noise_img_path = str(config.synthetic_data_noise / "train" / "00004.depth0_noise.png")  # synthetic image
    # noise_img_16bit = data_preprocess.noisy_1channel(original_16bit)
    noise_img_16bit = file_io.load_16bitImage(noise_img_path)
    noise_img = cv.normalize(noise_img_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    # denoised image
    pred_img_path = str(
        config.ws_path / "pncnn" / "output" / "epoch_99_synthetic_selval_output" / "00008.png")  # synthetic image
    pred_img_16bit = file_io.load_16bitImage(pred_img_path)
    denoise_img_16_bit = pred_filter(original_16bit, pred_img_16bit)
    denoise_img = cv.normalize(denoise_img_16_bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # denoise_img = median_filter(noise_img)

    # difference between original and denoised image
    difference_16bit = original - denoise_img
    difference = cv.normalize(difference_16bit, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    diff_noise = np.sum(original).astype(np.int64) - np.sum(noise_img).astype(np.int64)
    diff_denoise = np.sum(original).astype(np.int64) - np.sum(denoise_img).astype(np.int64)

    print(diff_noise)
    print(diff_denoise)

    # normal ground truth
    normals_gt_img, normals_pred_img = knn_normal.knn_normal_pred(config.synthetic_data_noise / "train",
                                                                  4,
                                                                  denoise_img_16_bit)
    normals_gt_img = normals_gt_img[1:-1,1:-1,:]
    normals_pred_img = normals_pred_img[1:-1,1:-1,:]


    # text annotations

    cv.putText(original, text='Original', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)
    cv.putText(noise_img, text='Adding Noise', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)
    cv.putText(denoise_img, text='Denoised', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)
    cv.putText(difference, text='Original-Denoised', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)
    cv.putText(normals_gt_img, text='Normal (Ground Truth)', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)
    cv.putText(normals_pred_img, text='Normal (Prediction)', org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)

    final_output_1 = cv.hconcat([original, noise_img, denoise_img, difference])
    final_output_2 = cv.hconcat([normals_gt_img, normals_pred_img])
    #
    # final_output = concat_vh([[original, noise_img, denoise_img],
    #                           [difference, normals_gt_img, normals_pred_img]])

    cv.imshow("original_noise", final_output_1)
    cv.imshow("original_noise_2", final_output_2)

    cv.waitKey(0)
    cv.destroyAllWindows()
