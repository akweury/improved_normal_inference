import cv2 as cv
import numpy as np
import torch


def binary(img):
    # h, w = img.shape[:2]
    c = torch.zeros(size=img.shape)
    mask = ~(img == 0)
    c[mask] = 1

    return c


def bi_interpolation(lower_left, lower_right, upper_left, upper_right, x, y):
    return lower_left * (1 - x) * (1 - y) + lower_right * x * (1 - y) + upper_left * (1 - x) * y + upper_right * x * y


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


def array2RGB(numpy_array, mask):
    min, max = numpy_array.min(), numpy_array.max()

    # convert normal to RGB color
    # h, w, c = numpy_array.shape
    numpy_array[mask] = ((numpy_array[mask] - min) / (max - min) * 255).astype(np.uint8)
    # for i in range(h):
    #     for j in range(w):
    #         if mask[i, j]:
    #             numpy_array[i, j] = (numpy_array[i, j] - min) / (max - min)
    #             numpy_array[i, j] = (numpy_array[i, j] * 255).astype(np.uint8)

    return numpy_array


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


def normalize(numpy_array):
    min, max = numpy_array.min(), numpy_array.max()
    numpy_array = (numpy_array - min) / (max - min)
    return numpy_array


def copy_make_border(img, patch_width):
    """
    This function applies cv.copyMakeBorder to extend the image by patch_width/2
    in top, bottom, left and right part of the image
    Patches/windows centered at the border of the image need additional padding of size patch_width/2
    """
    offset = np.int32(patch_width / 2.0)
    return cv.copyMakeBorder(img,
                             top=offset, bottom=offset,
                             left=offset, right=offset,
                             borderType=cv.BORDER_REFLECT)


def get_valid_pixels_idx(img):
    return np.sum(img, axis=2) != 0


def lightVisualization():
    points = []
    for px in range(-5, 6):
        for py in range(-5, 6):
            for pz in range(-5, 6):
                p = np.array([px, py, pz]).astype(np.float32) / 10
                if np.linalg.norm(p) > 0:
                    points.append(p / np.linalg.norm(p))
    return points


def cameraVisualization():
    points = []
    for p in range(-50, 51):
        ps = float(p) / 100.0
        points.append([ps, 0.5, 0.5])
        points.append([ps, -0.5, 0.5])
        points.append([ps, 0.5, -0.5])
        points.append([ps, -0.5, -0.5])
        points.append([0.5, ps, 0.5])
        points.append([-0.5, ps, 0.5])
        points.append([0.5, ps, -0.5])
        points.append([-0.5, ps, -0.5])
        points.append([0.5, 0.5, ps])
        points.append([0.5, -0.5, ps])
        points.append([-0.5, 0.5, ps])
        points.append([-0.5, -0.5, ps])

    for p in range(-30, 31):
        ps = float(p) / 100.0
        points.append([ps, 0.3, 0.3 + 0.8])
        points.append([ps, -0.3, 0.3 + 0.8])
        points.append([ps, 0.3, -0.3 + 0.8])
        points.append([ps, -0.3, -0.3 + 0.8])
        points.append([0.3, ps, 0.3 + 0.8])
        points.append([-0.3, ps, 0.3 + 0.8])
        points.append([0.3, ps, -0.3 + 0.8])
        points.append([-0.3, ps, -0.3 + 0.8])
        points.append([0.3, 0.3, ps + 0.8])
        points.append([0.3, -0.3, ps + 0.8])
        points.append([-0.3, 0.3, ps + 0.8])
        points.append([-0.3, -0.3, ps + 0.8])

    return points


def depth2vertex(depth, K, R, t):
    c, h, w = depth.shape

    camOrig = -R.transpose(0, 1) @ t.unsqueeze(1)

    X = torch.arange(0, depth.size(2)).repeat(depth.size(1), 1) - K[0, 2]
    Y = torch.transpose(torch.arange(0, depth.size(1)).repeat(depth.size(2), 1), 0, 1) - K[1, 2]
    Z = torch.ones(depth.size(1), depth.size(2)) * K[0, 0]
    Dir = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), 2)

    vertex = Dir * (depth.squeeze(0) / torch.norm(Dir, dim=2)).unsqueeze(2).repeat(1, 1, 3)
    vertex = R.transpose(0, 1) @ vertex.permute(2, 0, 1).reshape(3, -1)
    vertex = camOrig.unsqueeze(1).repeat(1, h, w) + vertex.reshape(3, h, w)
    vertex = vertex.permute(1, 2, 0)
    return np.array(vertex)
