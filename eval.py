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
from workspace.svdn import eval as svdn
from workspace.deepfit import eval as deepfit


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

    # svd network normal
    normal_img, eval_point_counter, total_time = svdn.eval(vertex_gt)
    mu.addText(normal_img, 'SVD_Net')
    mu.addText(normal_img, f"{eval_point_counter} points. processing time: {total_time:.3f} s", pos="lower_left",
               font_size=0.5)
    img_list.append(normal_img)

    # deepfit network normal
    deepfit_normal, eval_point_counter, total_time = deepfit.eval(vertex_gt)
    mu.addText(deepfit_normal, 'DeepFit')
    mu.addText(deepfit_normal, f"{eval_point_counter} points. processing time: {total_time:.3f} s", pos="lower_left",
               font_size=0.5)
    img_list.append(deepfit_normal)

    # show the results
    output = cv.hconcat(img_list)
    mu.show_images(output, "evaluation")
    time_now = datetime.datetime.now().strftime("%H_%M_%S")
    date_now = datetime.datetime.today().date()
    cv.imwrite(str(config.ws_path / f"evaluation_{date_now}_{time_now}.png"), output)


if __name__ == '__main__':
    main()
