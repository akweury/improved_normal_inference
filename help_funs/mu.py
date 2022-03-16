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
    """ Returns the angle in radians between vectors 'v1' and 'v2'::"""
    v1 = v1.reshape(-1, 3)
    v2 = v2.reshape(-1, 3)
    inner = np.sum(v1.reshape(-1, 3) * v2.reshape(-1, 3), axis=1)
    norms = np.linalg.norm(v1, axis=1, ord=2) * np.linalg.norm(v2, axis=1, ord=2)

    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    deg[deg > 90] = 180 - deg[deg > 90]
    return deg


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
    img_permuted = img.permute(0, 2, 3, 1)

    mask = img_permuted.sum(dim=3) == 0
    c_permute = torch.zeros(size=img_permuted.shape)

    c_permute[~mask] = 1

    c = c_permute.permute(0, 3, 1, 2)
    return c


def bi_interpolation(lower_left, lower_right, upper_left, upper_right, x, y):
    return lower_left * (1 - x) * (1 - y) + lower_right * x * (1 - y) + upper_left * (1 - x) * y + upper_right * x * y


def normalize3channel(numpy_array):
    mins, maxs = [], []
    if numpy_array.ndim != 3:
        raise ValueError
    h, w, c = numpy_array.shape
    for i in range(c):
        numpy_array[:, :, i], min, max = normalize(numpy_array[:, :, i], data=None)
        mins.append(min)
        maxs.append(max)
    return numpy_array, mins, maxs


def normalize(numpy_array, data):
    if numpy_array.ndim != 2:
        raise ValueError

    mask = numpy_array == 0
    if data is not None:
        min, max = data["minDepth"], data["maxDepth"]
    else:
        min, max = numpy_array.min(), numpy_array.max()
        if min == max:
            return numpy_array, 0, 1

    numpy_array[~mask] = (numpy_array[~mask] - min) / (max - min)
    return numpy_array, min, max


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
    if img_scaled.ndim == 2:
        raise ValueError

    if img_scaled.shape[2] == 1:
        normalized_img = normalize(img_scaled, data)
    elif img_scaled.shape[2] == 3:
        normalized_img, mins, maxs = normalize3channel(img_scaled)
    else:
        raise ValueError

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
                # TODO: fix here
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
    rgb = np.zeros(shape=normals.shape)
    # convert normal to RGB color
    h, w, c = normals.shape
    for i in range(h):
        for j in range(w):
            if mask[i, j]:
                rgb[i, j] = normals[i, j] * 0.5 + 0.5
                rgb[i, j, 2] = 1 - rgb[i, j, 2]
                rgb[i, j] = (rgb[i, j] * 255)
    rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return rgb


def normal2RGB_single(normal):
    normal = normal * 0.5 + 0.5
    normal[2] = 1 - normal[2]
    normal = normal * 255
    rgb = cv.normalize(normal, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return rgb


def rgb2normal(color):
    color_norm = color / np.linalg.norm(color)
    color_norm[2] = 1 - color_norm[2]
    return color_norm * 2 - 1


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
    normals_rgb = normal2RGB(normals)
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
def addText(img, text, pos='upper_left', font_size=1.6):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=(255, 255, 255),
               thickness=1, lineType=cv.LINE_AA)


def pure_color_img(color, size):
    img = np.zeros(shape=size).astype(np.uint8)
    img[:] = color
    return img


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


# https://www.geeksforgeeks.org/concatenate-images-using-opencv-in-python/
def concat_vh(list_2d):
    """
    show image in a 2d array
    :param list_2d: 2d array with image element
    :return: concatenated 2d image array
    """
    # return final image
    return cv.vconcat([cv.hconcat(list_h)
                       for list_h in list_2d])


def vconcat_resize(img_list, interpolation):
    w_min = min(img.shape[1] for img in img_list)
    im_list_resize = [cv2.resize(img,
                                 (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
                      for img in img_list]
    return cv2.vconcat(im_list_resize)


def hconcat_resize(img_list, interpolation):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv2.resize(img,
                                 (int(img.shape[1] * h_min / img.shape[0]),
                                  h_min), interpolation)
                      for img in img_list]

    return cv2.hconcat(im_list_resize)


def concat_tile_resize(list_2d):
    img_list_v = [hconcat_resize(list_h, cv2.INTER_CUBIC) for list_h in list_2d]
    return vconcat_resize(img_list_v, cv2.INTER_CUBIC)


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
    elif tensor.size() == (1, 3, 10, 10):
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


def median_filter(depth):
    padding = 2  # 2 optimal

    # add padding
    depth_padded = np.expand_dims(copy_make_border(depth, padding * 2), axis=2)
    h, w, c = depth_padded.shape

    mask = (depth_padded == 0)

    # predict
    for i in range(padding, h - padding):
        for j in range(padding, w - padding):
            if mask[i, j]:
                neighbor = depth_padded[i - padding:i + padding + 1, j - padding:j + padding + 1].flatten()
                neighbor = np.delete(neighbor, np.floor((padding * 2 + 1) ** 2 / 2).astype(np.int32))
                # depth_padded[i, j] = mu.bi_interpolation(lower_left, lower_right, upper_left, upper_right, 0.5, 0.5)
                depth_padded[i, j] = np.median(neighbor)

    # remove the padding
    pred_depth = depth_padded[padding:-padding, padding:-padding]

    return pred_depth.reshape(512, 512)


def pred_filter(img, pred_img):
    h, w = img.shape[:2]
    mask = ~(img == 0)
    img[mask] = pred_img[mask]

    return img


def choose_best(candidate_array, target):
    candidate_array = candidate_array.reshape(-1, 3)
    ele_num = candidate_array.shape[0]
    target = target.reshape(1, 3)
    target = np.repeat(target, ele_num, axis=0)

    diff = angle_between(candidate_array, target)
    min_index = np.argmin(diff)
    return candidate_array[min_index], diff
