import numpy as np


def normal_point_to_view_point(normal, point, view_point):
    if np.dot(normal, (point - view_point)) > 0:
        normal = -normal

    return normal


# https://stackoverflow.com/a/13849249
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    if vector.sum() == 0:
        return np.zeros(vector.shape)
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if np.sum(v1_u - v2_u) == 0:
        return 0
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def mse(img_1, img_2, valid_pixels=None):
    """

    :param img_1: np_array of image with shape height*width*channel
    :param img_2: np_array of image with shape height*width*channel
    :return: mse error of two images in range [0,1]
    """
    if img_1.shape != img_2.shape:
        print("MSE Error: img 1 and img 2 do not have the same shape!")
        raise ValueError

    h, w, c = img_1.shape
    diff = np.sum(np.abs(img_1 - img_2))
    if valid_pixels is not None:
        # only calculate the difference of valid pixels
        diff /= (valid_pixels * c)
    else:
        # calculate the difference of whole image
        diff /= (h * w * c)

    return diff


def get_valid_pixels(img):
    return np.count_nonzero(np.sum(img, axis=2) > 0)


def normal2RGB(normals, mask):
    # convert normal to RGB color
    h, w, c = normals.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                normals[i, j] = normals[i, j] * 0.5 + 0.5
                normals[i, j, 2] = 1 - normals[i, j, 2]
                # normals[i, j,2] = 1 - normals[i, j,2]
                # normals[i, j,0] = 1 - normals[i, j,0]
                normals[i, j] = (normals[i, j] * 255).astype(np.int32)

    return normals