import numpy as np


def normal_to_view_point(normal, point, view_point):
    if np.dot(normal, (view_point - point)) <= 0:
        normal = -normal

    return normal
