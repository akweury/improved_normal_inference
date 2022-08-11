import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os.path
from pathlib import Path

path = "/Users/jing/Downloads/feature_maps/cylinder/"


def line_chart(data, path, labels, x=None, title=None, x_scale=None, y_scale=None, y_label=None, show=False,
               log_y=False, cla_leg=False):
    if data.shape[1] <= 1:
        return

    if y_scale is None:
        y_scale = [1, 1]
    if x_scale is None:
        x_scale = [1, 1]

    for i, row in enumerate(data):
        if x is None:
            x = np.arange(row.shape[0]) * x_scale[1] + x_scale[0]
        y = row
        plt.plot(x, y, label=labels[i], lw=1)

    if title is not None:
        plt.title(title)

    if y_label is not None:
        plt.ylabel(y_label)

    if log_y:
        plt.yscale('log')

    plt.legend()
    plt.grid(True)

    if not os.path.exists(str(path)):
        os.mkdir(path)
    plt.savefig(
        str(Path(path) / f"{title}_{y_label}.png"))

    if show:
        plt.show()
    if cla_leg:
        plt.cla()


threshold_bright = 50
bright_percent_list = np.zeros(shape=(3, 128))
# read a feature map
for model_idx, prefix in enumerate(["feature_map_gcnn-noc_", "feature_map_gcnn-gcnn_", "feature_map_gcnn-cnn_"]):
    for idx in range(128):
        print(f"current fm idx: {idx}")
        fm_file = path + prefix + str(idx) + ".png"
        fm_512_img = Image.open(fm_file)
        fm_512 = np.asarray(fm_512_img)

        fm_32 = np.zeros(shape=(32, 32))
        for row_idx in range(16):
            for col_idx in range(16):
                patch_3c = fm_512[row_idx * 32:(row_idx + 1) * 32, col_idx * 32:(col_idx + 1) * 32, :]
                patch_1c = (patch_3c[:, :, 0] + patch_3c[:, :, 1] * 2 + patch_3c[:, :, 2]) / 4
                median_value = np.median(patch_1c)
                fm_32[row_idx, col_idx] = median_value
        mask = fm_32 == 0

        bright_pixel_percent = fm_32.sum()
        bright_percent_list[model_idx, idx] = bright_pixel_percent

# visual the chart
output_path = "/Users/jing/Downloads/feature_maps/"
line_chart(bright_percent_list, output_path, labels=["noc","gcnn","cnn"], title=f"Percentage greater than {threshold_bright}")
