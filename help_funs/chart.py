import datetime
import os.path
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from help_funs import mu

date_now = datetime.datetime.today().date()
time_now = datetime.datetime.now().strftime("%H_%M_%S")


def line_chart(data, path, title=None, x_scale=None, y_scale=None, x_label=None, y_label=None, show=False, log_y=False):
    if data.shape[1] <= 1:
        return

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for row in data:
        x = np.arange(row.shape[0]) * x_scale[1] + x_scale[0]
        y = row
        plt.plot(x, y)

    if title is not None:
        plt.title(title)

    if x_label is not None:
        plt.xlabel(x_label)
    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"line_{title}_{x_label}_{y_label}_{date_now}_{time_now}.png"))

    if show:
        plt.show()


# --------------------------- visualisation -------------------------------- #
# if k_max - k_min < 2:
#     exit()
#
# diffs = np.zeros((file_idx, k_max))
# file_idx = 0
# file_name = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
# while path.exists(file_name) and file_idx < max_file_num:
#     gt = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
#     if os.path.exists(gt):
#         normal_gt = cv.imread(gt)
#         valid_pixels = statistic.get_valid_pixels(normal_gt)
#     else:
#         continue
#
#     for j in range(k_min, k_max):
#         knn = file_io.get_output_file_name(file_idx, file_type="normal", method="knn", data_type=data_type, param=j)
#         if os.path.exists(knn):
#             normal_knn = cv.imread(knn)
#             diffs[file_idx, j] = statistic.mse(normal_gt, normal_knn)
#         else:
#             diffs[file_idx, j] = 0
#             continue
#
#     file_idx += 1
#     file_name = file_io.get_output_file_name(file_idx, file_type="normal", method="gt", data_type=data_type)
#
# # remove 0 elements
# diffs = diffs[:, k_min:k_max]
# print(f"mse: \n {diffs}")
# # visualisation
# chart.line_chart(diffs,
#                  title="normal_performance",
#                  x_scale=[k_min, 1],
#                  x_label="k_value",
#                  y_label="RGB_difference")


def draw_output_svd(x0, xout, target, exp_path, loss, epoch, i, prefix):
    target = target[0, :]
    xout = xout[0, :]
    # xout = out[:, :3, :, :]
    # cout = out[:, 3:6, :, :]

    # input normal
    input = mu.tenor2numpy(x0[:1, :, :, :])
    x0_normalized_8bit = mu.normal2RGB(input)
    x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
    mu.addText(x0_normalized_8bit, "Input(Normals)")

    # gt normal
    target = target.numpy()
    target_color = mu.normal2RGB_single(target).reshape(3)
    normal_gt_8bit = mu.pure_color_img(target_color, (512, 512, 3))
    mu.addText(normal_gt_8bit, "gt")

    # minimum difference between input normal and gt normal
    best_angle, diff = mu.choose_best(input, target)
    diff_min = np.min(diff)
    mu.addText(normal_gt_8bit, f'e={diff_min:.2f}', pos='lower_right', font_size=1.0)

    # normalize output normal
    xout_normal = xout.detach().numpy() / np.linalg.norm(xout.detach().numpy())
    xout_color = mu.normal2RGB_single(xout_normal).reshape(3)
    normal_cnn_8bit = mu.pure_color_img(xout_color, (512, 512, 3))
    mu.addText(normal_cnn_8bit, "output")

    # angle difference between output and target
    xout_normal = mu.rgb2normal(xout_color)
    tartget_normal = mu.rgb2normal(target_color)
    difference_angle = mu.angle_between(xout_normal, tartget_normal).item()
    mu.addText(normal_cnn_8bit, f'e={difference_angle:.2f}', pos='lower_right', font_size=1.0)

    # ------------------ combine together ----------------------------------------------

    output = cv.hconcat([x0_normalized_8bit, normal_gt_8bit, normal_cnn_8bit])
    cv.imwrite(str(exp_path / f"{prefix}_epoch_{epoch}_{i}_loss_{loss:.3f}.png"), output)
    # mu.show_images(output, f"svdn")


def draw_output(x0, xout, cout, c0, target, exp_path, loss, epoch, i, prefix):
    # c0 = out[:, 6:, :, :]
    # xout = out[:, :3, :, :]
    # cout = out[:, 3:6, :, :]

    x0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(x0[:1, :, :, :]))
    mu.addText(x0_normalized_8bit, "Input(Vertex)")

    normal_gt_8bit = mu.normalize2_8bit(mu.tenor2numpy(target[:1, :, :, :]))
    mu.addText(normal_gt_8bit, "gt")

    normal_cnn_8bit_norm = mu.normalize2_8bit(mu.tenor2numpy(xout[:1, :, :, :]))
    mu.addText(normal_cnn_8bit_norm, "output_shifted")

    first_row = [x0_normalized_8bit, normal_gt_8bit, normal_cnn_8bit_norm]

    # normalize output normal
    normal_cnn_8bit = mu.tenor2numpy(xout[:1, :, :, :]).astype(np.uint8)
    mu.addText(normal_cnn_8bit, "output")

    second_row = [normal_cnn_8bit]

    if c0 is not None:
        c0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(c0[:1, :, :, :]))
        mu.addText(c0_normalized_8bit, "Input Confidence")
        second_row.append(c0_normalized_8bit)

    if cout is not None:
        conf_cnn_8_bit = mu.tenor2numpy(cout[:1, :, :, :]).astype(np.uint8)
        mu.addText(conf_cnn_8_bit, "cout")
        second_row.append(conf_cnn_8_bit)

        conf_cnn_8_bit_norm = mu.normalize2_8bit(mu.tenor2numpy(cout[:1, :, :, :]))
        mu.addText(conf_cnn_8_bit_norm, "cout_shifted")
        second_row.append(conf_cnn_8_bit_norm)

    # ------------------ combine together ----------------------------------------------

    output = mu.concat_tile_resize([first_row, second_row])
    output = cv.resize(output, (1440, 1080))

    folder_path = exp_path / f"output_{date_now}_{time_now}"

    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    cv.imwrite(str(folder_path / f"{prefix}_epoch_{epoch}_{i}_loss_{loss:.3f}.png"), output)

    # np_array1, np_array2 = mu.tenor2numpy(out[:1, :, :, :]), mu.tenor2numpy(target[:1, :, :, :])
    # b1, g1, r1 = cv2.split(np_array1)
    # b2, g2, r2 = cv2.split(np_array2)
    # output_3 = mu.concat_vh([[b1, g1, r1], [b2, g2, r2]])
    # cv2.imwrite(str(nn_model.exp_dir / "output" / f"{prefix}_NNN_epoch_{epoch}_{i}_loss_{loss}_tensor.png"), output_3)

    # mu.show_images(output_3, f"tensor_different")
    # mu.show_images(output, f"{prefix}_NNN_epoch_{epoch}_{i}.png")
