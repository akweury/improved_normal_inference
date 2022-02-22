import numpy as np
import file_io
from improved_normal_inference import config

height, width = 512, 512


def gray_grid1d(h, w, grid, name):
    """

    :param h: image height
    :param w: image width
    :param grid: grid numbers
    :param name: image name
    :return: 16-bit gray gradual image
    """

    img_array = np.zeros(shape=(h, w))
    grid_width = width // grid
    for i in range(grid):
        grid_color = 65535 // grid * i
        img_array[:, i * grid_width:(i + 1) * grid_width] = grid_color
    file_io.write_np2img(img_array, name)


def gray_grid2d(h, w, grid, name):
    img_array = np.zeros(shape=(h, w))
    grid_width = width // grid
    for i in range(grid):
        grid_color_i = 65535 // grid * i
        for j in range(grid):
            grid_color_j = 65535 // grid * j
            grid_color = grid_color_i + grid_color_j
            img_array[i * grid_width:(i + 1) * grid_width, j * grid_width:(j + 1) * grid_width] = grid_color
    file_io.write_np2img(img_array, name)


if __name__ == '__main__':
    gray_grid1d(height, width, 10, str(config.dataset / "gradual1d_10.png"))
    gray_grid2d(height, width, 10, str(config.dataset / "gradual2d_10.png"))
    gray_grid1d(height, width, 512, str(config.dataset / "gradual1d_512.png"))
    gray_grid2d(height, width, 512, str(config.dataset / "gradual2d_512.png"))
