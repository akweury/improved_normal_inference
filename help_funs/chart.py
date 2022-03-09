import datetime
import numpy as np
import matplotlib.pyplot as plt
from help_funs import mu
import cv2 as cv

time_now = datetime.datetime.today().date()


def line_chart(data, path, title=None, x_scale=None, y_scale=None, x_label=None, y_label=None, show=False):
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

    plt.savefig(str(path / f"line_{title}_{x_label}_{y_label}_{time_now}.png"))

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

def draw_output(x0, out, target, exp_path, loss, epoch, i, prefix):
    c0 = out[:, 6:, :, :]
    xout = out[:, :3, :, :]
    cout = out[:, 3:6, :, :]

    # ------------------ input ----------------------------------------------
    x0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(x0[:1, :, :, :]))
    mu.addText(x0_normalized_8bit, "Input(Vertex)")

    c0_normalized_8bit = mu.normalize2_8bit(mu.tenor2numpy(c0[:1, :, :, :]))
    mu.addText(c0_normalized_8bit, "Input Confidence")

    input_img = [x0_normalized_8bit, c0_normalized_8bit]

    # ------------------ output ----------------------------------------------

    # normalize output normal
    conf_cnn_8_bit = mu.normalize2_8bit(mu.tenor2numpy(cout[:1, :, :, :]))
    mu.addText(conf_cnn_8_bit, "cout")

    normal_cnn_8bit = mu.normalize2_8bit(mu.tenor2numpy(xout[:1, :, :, :]))
    mu.addText(normal_cnn_8bit, "output")

    normal_gt_8bit = mu.tenor2numpy(target[:1, :, :, :]).astype(np.uint8)
    mu.addText(normal_gt_8bit, "gt")

    output_img = [conf_cnn_8_bit, normal_cnn_8bit, normal_gt_8bit]

    # ------------------ combine together ----------------------------------------------

    output = mu.concat_tile_resize([input_img, output_img])
    output = cv.resize(output, (1000, 1000))
    cv.imwrite(str(exp_path / "output" / f"{prefix}_NNN_epoch_{epoch}_{i}_loss_{loss:.3f}.png"), output)

    # np_array1, np_array2 = mu.tenor2numpy(out[:1, :, :, :]), mu.tenor2numpy(target[:1, :, :, :])
    # b1, g1, r1 = cv2.split(np_array1)
    # b2, g2, r2 = cv2.split(np_array2)
    # output_3 = mu.concat_vh([[b1, g1, r1], [b2, g2, r2]])
    # cv2.imwrite(str(nn_model.exp_dir / "output" / f"{prefix}_NNN_epoch_{epoch}_{i}_loss_{loss}_tensor.png"), output_3)

    # mu.show_images(output_3, f"tensor_different")
    # mu.show_images(output, f"{prefix}_NNN_epoch_{epoch}_{i}.png")
