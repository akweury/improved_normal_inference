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
    mu.addText(normal_img, name, font_size=0.8)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle))

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img


def main():
    # path = config.synthetic_data_noise / "test"
    # path = config.synthetic_data / "test"  # test 1108, 1164
    # path = config.geo_data / "train"

    path = config.real_data  # test 103, 166, 189,9
    eval_list = [9, 66, 110, 103, 166, 189]
    for idx in eval_list:
        data, depth, depth_noise, normal_gt = file_io.load_single_data(path, idx=idx)
        vertex_filted_gt = None
        if path == config.real_data:
            depth_filtered = mu.median_filter(depth)
            vertex_filted_gt = mu.depth2vertex(torch.tensor(depth_filtered).permute(2, 0, 1),
                                               torch.tensor(data['K']),
                                               torch.tensor(data['R']).float(),
                                               torch.tensor(data['t']).float())
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
        # diff_list.append(gt_diff)

        # svd normal
        normal_svd, svd_img = svd.eval(vertex_gt, farthest_neighbour=2)
        svd_img, svd_diff = eval_post_processing(normal_svd, svd_img, normal_gt, "SVD")
        img_list.append(svd_img)
        diff_list.append(svd_diff)

        # nnn24 9999
        normal_model_path = config.ws_path / "nnn24" / "trained_model" / "full_normal_2999" / "checkpoint-9999.pth.tar"
        if path == config.real_data:
            normal_noraml, normal_img, _, _ = eval.eval(vertex_filted_gt, normal_model_path, k=2, output_type='normal')
        else:
            normal_noraml, normal_img, _, _ = eval.eval(vertex_gt, normal_model_path, k=2, output_type='normal')
        normal_img_final, birnak_diff_img = eval_post_processing(normal_noraml, normal_img, normal_gt, "Neighbor_9999")
        img_list.append(normal_img_final)
        diff_list.append(birnak_diff_img)

        # nnnn 99999
        normal_model_path = config.ws_path / "nnnn" / "trained_model" / "output_2022-05-02_08_09_16" / "checkpoint-10082.pth.tar"
        nnnn_noraml, nnnn_img, _, _ = eval.eval(vertex_gt, normal_model_path, k=1, output_type='normal')
        nnnn_img_final, birnak_diff_img = eval_post_processing(nnnn_noraml, nnnn_img, normal_gt, "Vertex_10082")
        img_list.append(nnnn_img_final)
        diff_list.append(birnak_diff_img)

        # show the results
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        output_diff = cv.hconcat(diff_list)
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        date_now = datetime.datetime.today().date()
        cv.imwrite(str(config.ws_path / "eval_output" / f"evaluation_{date_now}_{time_now}.png"), output)
        cv.imwrite(str(config.ws_path / "eval_output" / f"diff{date_now}_{time_now}.png"), output_diff)
        print(f"{idx} has been evaluated.")


if __name__ == '__main__':
    main()
