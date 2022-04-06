"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import cv2 as cv
import torch
import datetime
import numpy as np
from help_funs import file_io, mu
import config
from workspace.svd import eval as svd
from workspace import eval


def eval_post_processing(normal, normal_img, normal_gt, name):
    out_ranges = mu.addHist(normal_img)
    mu.addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)
    mu.addText(normal_img, name)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle))

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img


def main():
    path = config.synthetic_data_noise / "train"

    # path = config.geo_data / "train"
    data, depth, depth_noise, normal_gt = file_io.load_single_data(path, idx=169)

    # vertex
    vertex_gt = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                torch.tensor(data['K']),
                                torch.tensor(data['R']).float(),
                                torch.tensor(data['t']).float())

    img_list = []
    diff_list = []
    # ground truth normal
    normal_gt_img = mu.normal2RGB(normal_gt)
    normal_gt_ = mu.rgb2normal(normal_gt_img)
    gt_img, gt_diff = eval_post_processing(normal_gt_, normal_gt_img, normal_gt, "GT")
    img_list.append(gt_img)
    diff_list.append(gt_diff)

    # svd normal
    normal_svd, svd_img = svd.eval(vertex_gt, farthest_neighbour=2)
    svd_img, svd_diff = eval_post_processing(normal_svd, svd_img, normal_gt, "SVD")
    img_list.append(svd_img)
    diff_list.append(svd_diff)

    # neighbor rgb
    neighbor_model_path = config.ws_path / "nnn24" / "trained_model" / "full_nnn24_05_04_2022" / "checkpoint-814.pth.tar"
    normal_neighbor, normal_neighbor_img, normal_neighbor_pn, normal_neighbor_time = eval.eval(vertex_gt,
                                                                                               neighbor_model_path, k=2)
    neighbor_img, neighbor_diff = eval_post_processing(normal_neighbor, normal_neighbor_img, normal_gt, "FULL_RGB")
    img_list.append(neighbor_img)
    diff_list.append(neighbor_diff)

    # neighbor normal
    neighbor_model_path = config.ws_path / "nnn24" / "trained_model" / "2022_04_06_10_07_33" / "checkpoint-572.pth.tar"
    normal_neighbor, normal_neighbor_img, normal_neighbor_pn, normal_neighbor_time = eval.eval(vertex_gt,
                                                                                               neighbor_model_path,
                                                                                               k=2,
                                                                                               output_type='normal')
    neighbor_img, neighbor_diff = eval_post_processing(normal_neighbor, normal_neighbor_img, normal_gt, "32_Normal")
    img_list.append(neighbor_img)
    diff_list.append(neighbor_diff)

    # # # vertex normal
    # vertex_model_path = config.ws_path / "nnn" / "trained_model" / "checkpoint.pth.tar"
    # normal_vertex, normal_vertex_img, normal_vertex_p_num, normal_vertex_time = eval.eval(vertex_gt, vertex_model_path,
    #                                                                                       k=1, output_type='rgb')
    # vertex_img, vertex_diff = eval_post_processing(normal_vertex, normal_vertex_img, normal_gt, "Vertex")
    # img_list.append(vertex_img)
    # diff_list.append(vertex_diff)

    # show the results
    output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
    output_diff = cv.hconcat(diff_list)
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    date_now = datetime.datetime.today().date()
    cv.imwrite(str(config.ws_path / "eval_output" / f"evaluation_{date_now}_{time_now}.png"), output)
    cv.imwrite(str(config.ws_path / "eval_output" / f"diff{date_now}_{time_now}.png"), output_diff)


if __name__ == '__main__':
    main()
