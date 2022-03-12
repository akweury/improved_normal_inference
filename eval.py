"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import cv2 as cv
import torch
from help_funs import file_io, mu
import config
from workspace.svd import eval as svd


def main():
    path = config.synthetic_data_noise / "train"
    idx = 0
    data, depth, depth_noise, normal_gt = file_io.load_single_data(path, idx)

    # ground truth normal
    normal_gt_img = mu.normal2RGB(normal_gt)
    mu.addText(normal_gt_img, 'Ground Truth')

    # vertex
    vertex_gt = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                torch.tensor(data['K']),
                                torch.tensor(data['R']).float(),
                                torch.tensor(data['t']).float())

    # svd normal
    normal_svd, normal_svd_img = svd.eval(vertex_gt, farthest_neighbour=2)
    mu.addText(normal_svd_img, 'SVD')

    output = cv.hconcat([normal_gt_img, normal_svd_img])
    mu.show_images(output, "evaluation")


if __name__ == '__main__':
    main()
