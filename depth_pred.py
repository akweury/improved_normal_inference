"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import json
import numpy as np

from improved_normal_inference.help_funs import file_io, mu


def pred(image, depth_padded, mask_unknown_depth):
    # TODO: pred the depth of each pixel
    # use a simple CNN to predict depth in each pixel
    h, w, c = depth_padded.shape

    pass


def remove_padding(depth_mended):
    pass


def repair_depth_map(idx, data_type, padding=1):
    # get file names from dataset
    image_file, ply_file, json_file, depth_file, normal_file = file_io.get_file_name(idx, data_type)
    f = open(json_file)
    data = json.load(f)
    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)

    image = file_io.load_8bitImage(image_file)
    depth = file_io.load_scaled16bitImage(depth_file, data['minDepth'], data['maxDepth'])

    depth_padded = np.expand_dims(mu.copy_make_border(depth, padding), axis=2)
    img_padded = mu.copy_make_border(image, padding)

    h, w, c = depth_padded.shape
    # calculate vertex from depth map
    mask_padded = np.sum(np.abs(depth_padded), axis=2) != 0

    # img = cv.imread(image_file)
    # sobelx = cv.Sobel(img, cv.CV_64F, 1, 0, ksize=5)
    # sobely = cv.Sobel(img, cv.CV_64F, 0, 1, ksize=5)
    # laplacian = cv.Laplacian(img, cv.CV_64F)
    # cv.imshow('sobelx', sobelx)
    # cv.imshow('sobely', sobely)
    # cv.imshow('laplacian', laplacian)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    pred_depth = pred(img_padded, depth_padded, ~mask_padded)

    depth_mended = np.zeros(shape=depth.shape)
    for i in range(h):
        for j in range(w):
            if mask_padded[i, j]:
                depth_mended[i, j] = depth_padded[i, j]
            elif mask_padded[i, j]:
                depth_mended[i, j] = pred_depth[i, j]

    # save depth image
    depth_mended_file = file_io.get_output_file_name(idx, "depth", "cnn", data_type)
    file_io.write_np2img(depth_mended, depth_mended_file)

    depth = remove_padding(depth_mended)

    return depth


if __name__ == '__main__':
    data_type = "real"
    file_idx = 0
    depth_map = repair_depth_map(file_idx, data_type)
