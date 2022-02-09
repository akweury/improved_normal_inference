import datetime
import numpy as np
import matplotlib.pyplot as plt

time_now = datetime.datetime.today().date()


def line_chart(data, title=None, x_scale=None, y_scale=None, x_label=None, y_label=None):
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

    plt.savefig(f"line_{title}_{x_label}_{y_label}_{time_now}.png")
    plt.show()
