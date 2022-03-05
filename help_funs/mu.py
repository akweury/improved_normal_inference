import cv2
import cv2 as cv
import numpy as np
import torch


# --------------------------- evaluate operations ----------------------------------------------------------------------

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


def get_valid_pixels_idx(img):
    return np.sum(img, axis=2) != 0


# --------------------------- filter operations -----------------------------------------------------------------------
def binary(img):
    # h, w = img.shape[:2]
    img_permuted = img.permute(2, 3, 1, 0)

    mask = img_permuted.sum(dim=2).sum(dim=2) == 0
    c_permute = torch.zeros(size=img_permuted.shape)

    c_permute[~mask] = 1

    c = c_permute.permute(3, 2, 0, 1)
    return c


def bi_interpolation(lower_left, lower_right, upper_left, upper_right, x, y):
    return lower_left * (1 - x) * (1 - y) + lower_right * x * (1 - y) + upper_left * (1 - x) * y + upper_right * x * y


def normalize(numpy_array, data):
    mask = numpy_array.sum(axis=2) == 0
    if data is not None:
        numpy_array[~mask] = (numpy_array[~mask] - data["minDepth"]) / (data["maxDepth"] - data["minDepth"])
    else:
        min, max = numpy_array[~mask].min(), numpy_array.max()
        if min != max:
            numpy_array[~mask] = (numpy_array[~mask] - min) / (max - min)
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


def normalize2_8bit(img_scaled, data=None):
    normalized_img = normalize(img_scaled, data)
    img_8bit = cv.normalize(normalized_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_8bit


# --------------------------- convert operations -----------------------------------------------------------------------


def normal_point2view_point(normal, point, view_point):
    if np.dot(normal, (point - view_point)) > 0:
        normal = -normal

    return normal


def compute_normal(vertex, mask, k):
    normals = np.zeros(shape=vertex.shape)
    for i in range(k, vertex.shape[0]):
        for j in range(k, vertex.shape[1]):
            if mask[i, j]:
                neighbors = vertex[i - k:i + k, j - k:j + k]  # get its k neighbors
                neighbors = neighbors.reshape(neighbors.shape[0] * neighbors.shape[1], 3)
                neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)
                plane_vectors = neighbors - vertex[i, j]

                u, s, vh = np.linalg.svd(plane_vectors)
                normal = vh.T[:, -1]
                normal = normal_point2view_point(normal, vertex[i][j], np.array([0, 0, 0]))
                if np.linalg.norm(normal) != 1:
                    normal = normal / np.linalg.norm(normal)
                normals[i, j] = normal

    return normals


# --------------------------- convert functions -----------------------------------------------------------------------

def array2RGB(numpy_array, mask):
    # convert normal to RGB color
    min, max = numpy_array.min(), numpy_array.max()
    if min != max:
        numpy_array[mask] = ((numpy_array[mask] - min) / (max - min))

    return (numpy_array * 255).astype(np.uint8)


def normal2RGB(normals):
    mask = np.sum(np.abs(normals), axis=2) != 0
    # convert normal to RGB color
    h, w, c = normals.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                normals[i, j] = normals[i, j] * 0.5 + 0.5
                normals[i, j, 2] = 1 - normals[i, j, 2]
                normals[i, j] = (normals[i, j] * 255)

    return normals.astype(np.uint8)


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


def vertex2normal(vertex, k_idx):
    mask = np.sum(np.abs(vertex), axis=2) != 0
    normals = compute_normal(vertex, mask, k_idx)
    normals_rgb = normal2RGB(normals).astype(np.uint8)
    return normals, normals_rgb


def depth2normal(depth, k_idx, K, R, t):
    if depth.ndim == 2:
        depth = np.expand_dims(depth, axis=2)
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(K),
                          torch.tensor(R).float(),
                          torch.tensor(t).float())
    return vertex2normal(vertex, k_idx)


# -------------------------------------- openCV Utils ------------------------------------------
def addText(img, text):
    cv.putText(img, text=text, org=(10, 50),
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)


def concat_vh(list_2d):
    """
    show image in a 2d array
    :param list_2d: 2d array with image element
    :return: concatenated 2d image array
    """
    # return final image
    return cv.vconcat([cv.hconcat(list_h)
                       for list_h in list_2d])


def show_numpy(numpy_array, title):
    if numpy_array.shape[2] == 3:
        # rgb image
        cv2.imshow(f"numpy_{title}", numpy_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Unsupported input array shape.")


def tenor2numpy(tensor):
    if tensor.size() == (1, 3, 512, 512):
        return tensor.permute(2, 3, 1, 0).sum(dim=3).detach().numpy()
    else:
        print("Unsupported input tensor size.")


def show_tensor(tensor, title):
    numpy_array = tenor2numpy(tensor)
    cv2.imshow(f"tensor_{title}", numpy_array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_images(array, title):
    cv2.imshow(title, array)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_grid(grid, title):
    cv2.imshow(f"grid_{title}", concat_vh(grid))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def scale16bitImage(img, minVal, maxVal):
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0
    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img)

    return img.astype(np.float32)
