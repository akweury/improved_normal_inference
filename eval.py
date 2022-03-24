"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import cv2 as cv
import torch
import datetime

from help_funs import file_io, mu
import config
from workspace.svd import eval as svd
from workspace import eval


def main():
    # path = config.synthetic_data_noise / "train"
    #
    path = config.geo_data / "train"
    data, depth, depth_noise, normal_gt = file_io.load_single_data(path, idx=0)

    # vertex
    vertex_gt = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                torch.tensor(data['K']),
                                torch.tensor(data['R']).float(),
                                torch.tensor(data['t']).float())

    img_list = []

    # ground truth normal
    normal_gt_img = mu.normal2RGB(normal_gt)
    mu.addText(normal_gt_img, 'Ground Truth')
    img_list.append(normal_gt_img)

    # svd normal
    normal_svd, normal_svd_img = svd.eval(vertex_gt, farthest_neighbour=2)
    mu.addText(normal_svd_img, 'SVD')
    img_list.append(normal_svd_img)

    # neighbor normal
    # neighbor_model_path = config.ws_path / "nnn" / "trained_model" / "neighbor_model.pth.tar"
    # normal_neighbor, normal_neighbor_img = eval.eval(vertex_gt, neighbor_model_path)
    # mu.addText(normal_svd_img, 'Neighbor')
    # img_list.append(normal_neighbor_img)

    # vertex normal
    vertex_model_path = config.ws_path / "nnn" / "trained_model" / "vertex_model.pth.tar"
    normal_vertex, normal_vertex_img = eval.eval(vertex_gt, vertex_model_path, k=1)
    mu.addText(normal_vertex_img, 'Vertex')
    img_list.append(normal_vertex_img)

    # show the results
    output = cv.hconcat(img_list)
    mu.show_images(output, "evaluation")
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    date_now = datetime.datetime.today().date()
    cv.imwrite(str(config.ws_path / f"evaluation_{date_now}_{time_now}.png"), output)


if __name__ == '__main__':
    main()
