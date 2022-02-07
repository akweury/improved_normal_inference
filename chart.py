import numpy as np
import matplotlib.pyplot as plt


def line_chart(data, title=None, x_scale=None, y_scale=None):
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
    plt.show()
