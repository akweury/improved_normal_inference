import cv2 as cv
import numpy as np
import torch
import json
import matplotlib.pyplot as plt


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
    # inner = np.sum(v1.reshape(-1, 3) * v2.reshape(-1, 3), axis=1)
    # norms = np.linalg.norm(v1, axis=1, ord=2) * np.linalg.norm(v2, axis=1, ord=2)
    v1_u = v1 / (np.linalg.norm(v1, axis=1, ord=2, keepdims=True) + 1e-9)
    v2_u = v2 / (np.linalg.norm(v2, axis=1, ord=2, keepdims=True) + 1e-9)

    rad = np.arccos(np.clip(np.sum(v1_u * v2_u, axis=1), -1.0, 1.0))
    deg = np.rad2deg(rad)
    deg[deg > 90] = 180 - deg[deg > 90]
    return deg


def vertex2light_direction(vertex_map, light_sorce):
    light_direction = light_sorce - vertex_map
    light_direction_map = light_direction / np.linalg.norm(light_direction, ord=2, axis=2, keepdims=True)

    return light_direction_map


def vertex2light_direction_tensor(vertex_map, light_sorce):
    light_direction = light_sorce - vertex_map
    light_direction_map = light_direction / (torch.norm(light_direction, p=2, dim=1, keepdim=True) + 1e-20)

    return light_direction_map


def albedo_tensor(I, N, L):
    rho = I.reshape(1, 1, 512, 512) / (torch.sum(N * L, dim=1, keepdim=True) + 1e-20)
    return rho


def albedo(I, N, L):
    rho = I.reshape(512, 512) / (np.sum(N * L, axis=-1) + 1e-20)
    return rho


def angle_between_2d(m1, m2):
    """ Returns the angle in radians between matrix 'm1' and 'm2'::"""
    m1_u = m1 / (np.linalg.norm(m1, axis=2, ord=2, keepdims=True) + 1e-9)
    m2_u = m2 / (np.linalg.norm(m2, axis=2, ord=2, keepdims=True) + 1e-9)

    rad = np.arccos(np.clip(np.sum(m1_u * m2_u, axis=2), -1.0, 1.0))
    deg = np.rad2deg(rad)
    deg[deg > 90] = 180 - deg[deg > 90]
    return deg


def radians_between_2d_tensor(t1, t2, mask=None):
    """ Returns the angle in radians between matrix 'm1' and 'm2'::"""
    # t1 = t1.permute(0, 2, 3, 1).to("cpu").detach().numpy()
    # t2 = t2.permute(0, 2, 3, 1).to("cpu").detach().numpy()
    # mask = mask.to("cpu").permute(0, 2, 3, 1).squeeze(-1)
    t1 = t1.permute(0, 2, 3, 1)
    t2 = t2.permute(0, 2, 3, 1)

    mask = mask.permute(0, 2, 3, 1).squeeze(-1)
    if mask is not None:
        t1 = t1[mask]
        t2 = t2[mask]
    t1_u = t1 / (torch.norm(t1, dim=-1, keepdim=True) + 1e-9)
    t2_u = t2 / (torch.norm(t2, dim=-1, keepdim=True) + 1e-9)
    # rad = torch.arccos(torch.clip(torch.sum(t1_u * t2_u, dim=-1), -1.0, 1.0))
    rad = torch.arccos(torch.clip(torch.sum(t1_u * t2_u, dim=-1), -1.0, 1.0))
    assert torch.sum(rad != rad) == 0
    # print(f"\t output normal: ({t1[0, 0].item():.2f},{t1[0, 1].item():.2f}, {t1[0, 2].item():.2f})")
    # print(f"\t target normal: ({t2[0, 0].item():.2f},{t2[0, 1].item():.2f},{t2[0, 2].item():.2f}\n"
    #       f"\t rad:{rad[0].item():.2f}\n"
    #       f"\t mse:{F.mse_loss(t1[0, :], t2[0, :]):.2f}\n")

    # deg = torch.rad2deg(rad)
    # deg[deg > 90] = 180 - deg[deg > 90]

    return rad


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

# https: // github.com / bnsreenu / python_for_image_processing_APEER / blob / master / tutorial41_image_filters_using_fourier_transform_DFT.py
def fft_filter(input_array):
    if input_array.shape != (512, 512):
        raise ValueError

    # file_io.save_16bitImage(input_array, str(Path(config.ws_path) / f"depth.png"))

    rows, cols = input_array.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    mask = np.ones((rows, cols, 2), np.uint8)
    r = 50
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0

    input_array = np.float32(input_array)
    dft = cv.dft(input_array, flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    fshift = dft_shift * mask
    fshift_mask_mag = 20 * np.log(cv.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv.idft(f_ishift)
    img_back = cv.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(input_array, cmap='gray')
    ax1.title.set_text('Input Image')
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(magnitude_spectrum, cmap='gray')
    ax2.title.set_text('FFT of image')
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(fshift_mask_mag, cmap='gray')
    ax3.title.set_text('FFT + Mask')
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(img_back, cmap='gray')
    ax4.title.set_text('After inverse FFT')

    mask_valid_area = input_array != 0
    img_back[~mask_valid_area] = 0

    # noramlize hpf image to 16 bit
    img_back = normalise216bitImage(img_back)

    # get the mask of detail pixels
    threshold = 0
    mask_hp = img_back > threshold
    input_array[~mask_hp] = 0

    # save filtered img
    # file_io.save_16bitImage(img_back, str(Path(config.ws_path) / f"fft.png"))
    # plt.savefig(str(Path(config.ws_path) / f"fft_info.png"), dpi=1000)

    # check if save and load function has loss


def canny_edge_filter(img_path):
    img = cv.imread(img_path, 0)

    plt.subplot(131), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])

    edges = cv.Canny(img, 1000, 0, apertureSize=7, L2gradient=True)
    plt.subplot(132), plt.imshow(edges, cmap='gray')
    plt.title('Edge '), plt.xticks([]), plt.yticks([])

    edges = cv.Canny(img, 1000, 0, apertureSize=7, L2gradient=False)
    plt.subplot(133), plt.imshow(edges, cmap='gray')
    plt.title('Edge'), plt.xticks([]), plt.yticks([])

    # plt.savefig(str(Path(config.ws_path) / f"canny_comparison.png"), dpi=1000)


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


def filter_noise(numpy_array, threshold):
    if len(threshold) != 2:
        raise ValueError
    threshold_min, threshold_max = threshold[0], threshold[1]
    numpy_array[numpy_array < threshold_min] = threshold_min
    numpy_array[numpy_array > threshold_max] = threshold_max
    return numpy_array


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


def normalize(numpy_array, data=None):
    if numpy_array.ndim != 2:
        if numpy_array.shape == (512, 512):
            numpy_array = numpy_array.reshape(512, 512, 1)

    mask = numpy_array == 0
    if data is not None:
        min, max = data["minDepth"], data["maxDepth"]
    else:
        min, max = numpy_array.min(), numpy_array.max()
        if min == max:
            return numpy_array, 0, 1

    numpy_array[~mask] = (numpy_array[~mask] - min).astype(np.float32) / (max - min)
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


def normalize2_32bit(img_scaled, data=None):
    if img_scaled.ndim == 2:
        raise ValueError

    if img_scaled.shape[2] == 1:
        normalized_img, mins, maxs = normalize(img_scaled.sum(axis=-1), data)
    elif img_scaled.shape[2] == 3:
        normalized_img, mins, maxs = normalize3channel(img_scaled)
    else:
        raise ValueError

    img_32bit = cv.normalize(normalized_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_32bit


def normalize2_16bit(img):
    newImg = img.reshape(512, 512)
    normalized_img, _, _ = normalize(newImg)
    img_16bit = cv.normalize(normalized_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return img_16bit


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
                neighbors = vertex[max(i - k, 0): min(i + k + 1, vertex.shape[1] - 1),
                            max(0, j - k): min(j + k + 1, vertex.shape[0] - 1)]  # get its k neighbors
                # neighbors = vertex[i - k:i + k, j - k:j + k]
                neighbors = neighbors.reshape(neighbors.shape[0] * neighbors.shape[1], 3)
                neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)  # delete center vertex
                # delete background vertex
                neighbors = np.delete(neighbors, np.where(neighbors == np.zeros(3)), axis=0)

                plane_vectors = neighbors - vertex[i, j]

                u, s, vh = np.linalg.svd(plane_vectors)
                normal = vh.T[:, -1]
                normal = normal_point2view_point(normal, vertex[i][j], np.array([0, 0, 0]))
                if np.linalg.norm(normal) != 1:
                    normal = normal / np.linalg.norm(normal)
                normals[i, j] = normal

    return normals


def generate_normals_all(vectors, vertex, normal_gt, svd_rank=3, MAX_ATTEMPT=1000):
    point_num, neighbors_num = vectors.shape[:2]
    random_idx = np.random.choice(neighbors_num, size=(point_num, MAX_ATTEMPT, svd_rank))
    candidate_neighbors = np.zeros(shape=(point_num, MAX_ATTEMPT, svd_rank, 3))
    # candidate_neighbors = vectors[random_idx]
    for i in range(point_num):
        for j in range(MAX_ATTEMPT):
            candidate_neighbors[i, j] = vectors[i][random_idx[i, j]]

    u, s, vh = np.linalg.svd(candidate_neighbors)
    candidate_normal = np.swapaxes(vh, -2, -1)[:, :, :, -1]
    vertex = np.repeat(np.expand_dims(vertex, axis=1), MAX_ATTEMPT, axis=1)
    normal = normal_point2view_point(candidate_normal, vertex, np.zeros(shape=vertex.shape))
    # if np.linalg.norm(normal, axis=1, ord=2) != 1:
    #     normal = normal / np.linalg.norm(normal)
    normal_gt_expended = np.repeat(np.expand_dims(normal_gt, axis=1), repeats=MAX_ATTEMPT, axis=1)
    error = angle_between_2d(normal, normal_gt_expended)
    best_error = np.min(error, axis=1)
    best_error_idx = np.argmin(error, axis=1)
    best_normals_idx = np.zeros(shape=(point_num, svd_rank))
    best_normals = np.zeros(shape=(point_num, 3))
    for i in range(point_num):
        best_normals_idx[i] = random_idx[i, best_error_idx[i]]
        best_normals[i] = normal[i, best_error_idx[i]]

    return best_normals, best_normals_idx, best_error


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
    rgb[mask] = normals[mask] * 0.5 + 0.5
    rgb[:, :, 2][mask] = 1 - rgb[:, :, 2][mask]
    rgb[mask] = rgb[mask] * 255

    # rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = np.rint(rgb)
    return rgb.astype(np.uint8)


def normal2RGB_torch(normals):
    if normals.size() != (3, 512, 512):
        raise ValueError

    normals = normals.permute(1, 2, 0)
    mask = torch.sum(torch.abs(normals), dim=2) != 0
    rgb = torch.zeros(size=normals.shape).to(normals.device)

    # convert normal to RGB color
    rgb[mask] = normals[mask] * 0.5 + 0.5
    rgb[:, :, 2][mask] = 1 - rgb[:, :, 2][mask]
    rgb[mask] = rgb[mask] * 255

    # rgb = cv.normalize(rgb, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    rgb = torch.round(rgb)
    rgb = rgb.permute(2, 0, 1)
    return rgb.byte()


def normal2RGB_single(normal):
    normal = normal * 0.5 + 0.5
    normal[2] = 1 - normal[2]
    normal = normal * 255
    rgb = cv.normalize(normal, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return rgb


def rgb2normal(color):
    mask = color.sum(axis=2) == 0
    color_norm = np.zeros(shape=color.shape)
    h, w, c = color.shape
    for i in range(h):
        for j in range(w):
            if not mask[i, j]:
                color_norm[i, j] = color[i, j] / 255.0
                color_norm[i, j, 2] = 1 - color_norm[i, j, 2]
                color_norm[i, j] = (color_norm[i, j] - 0.5) / 0.5
    # color_norm = color_norm / (np.linalg.norm(color_norm, axis=2, ord=2, keepdims=True)+1e-8)
    return color_norm


def rgb2normal_tensor(color):
    color = color.permute(0, 2, 3, 1)
    mask = color.sum(dim=-1) == 0
    color_norm = torch.zeros(color.shape).to(color.device)
    color_norm[~mask] = color[~mask] / 255.0
    color_norm[~mask][:, 2] = 1 - color_norm[~mask][:, 2]
    color_norm[~mask] = (color_norm[~mask] - 0.5) / 0.5

    return color_norm


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
def addText(img, text, pos='upper_left', font_size=1.6, color=(255, 255, 255)):
    h, w = img.shape[:2]
    if pos == 'upper_left':
        position = (10, 50)
    elif pos == 'upper_right':
        position = (w - 250, 80)
    elif pos == 'lower_right':
        position = (h - 200, w - 20)
    elif pos == 'lower_left':
        position = (10, w - 20)
    else:
        raise ValueError('unsupported position to put text in the image.')

    cv.putText(img, text=text, org=position,
               fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=font_size, color=color,
               thickness=1, lineType=cv.LINE_AA)


# https://docs.opencv.org/4.x/d1/db7/tutorial_py_histogram_begins.html
def addHist(img):
    h, w = img.shape[:2]
    fig = plt.figure()
    color = ([255, 0, 0], [0, 255, 0], [0, 0, 255])
    color_ranges = []
    for i, col in enumerate(color):
        hist_min, hist_max = img[:, :, i].min().astype(np.int), img[:, :, i].max().astype(np.int)
        color_ranges.append([int(hist_min), int(hist_max)])

        if hist_max - hist_min < 2:
            return "..."
        histr, histr_x = np.histogram(img[:, :, i], bins=np.arange(hist_min, hist_max + 1))
        histr = np.delete(histr, np.where(histr == histr.max()), axis=0)

        thick = 2
        histr = histr / max(histr.max(), 100)
        for i in range(histr.shape[0]):
            height = int(histr[i] * 50)
            width = int(w / histr.shape[0])
            img[max(h - 1 - height - thick, 0):min(h - 1, h - height + thick),
            max(0, i * width - thick):min(w - 1, i * width + thick)] = col
    plt.close('all')
    return color_ranges


def pure_color_img(color, size):
    img = np.zeros(shape=size).astype(np.uint8)
    img[:] = color
    return img


# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
def image_resize(image, width=None, height=None, inter=cv.INTER_AREA):
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
    resized = cv.resize(image, dim, interpolation=inter)

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
    im_list_resize = [cv.resize(img,
                                (w_min, int(img.shape[0] * w_min / img.shape[1])), interpolation=interpolation)
                      for img in img_list]
    return cv.vconcat(im_list_resize)


def hconcat_resize(img_list, interpolation):
    h_min = min(img.shape[0] for img in img_list)
    im_list_resize = [cv.resize(img,
                                (int(img.shape[1] * h_min / img.shape[0]),
                                 h_min), interpolation)
                      for img in img_list]

    return cv.hconcat(im_list_resize)


def concat_tile_resize(list_2d):
    img_list_v = [hconcat_resize(list_h, cv.INTER_CUBIC) for list_h in list_2d]
    return vconcat_resize(img_list_v, cv.INTER_CUBIC)


def show_numpy(numpy_array, title):
    if numpy_array.shape[2] == 3:
        # rgb image
        cv.imshow(f"numpy_{title}", numpy_array)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print(f"Unsupported input array shape.")


def tenor2numpy(tensor):
    if tensor.size() == (1, 3, 512, 512):
        return tensor.permute(2, 3, 1, 0).sum(dim=3).detach().numpy()
    elif tensor.size() == (1, 4, 512, 512):
        return tensor.permute(2, 3, 1, 0).sum(dim=3).detach().numpy()
    elif tensor.size() == (1, 1, 512, 512):
        return tensor.permute(2, 3, 1, 0).sum(dim=3).detach().numpy()
    else:
        print("Unsupported input tensor size.")


def show_tensor(tensor, title):
    numpy_array = tenor2numpy(tensor)
    cv.imshow(f"tensor_{title}", numpy_array)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_images(array, title):
    cv.imshow(title, array)
    cv.waitKey(0)
    cv.destroyAllWindows()


def show_grid(grid, title):
    cv.imshow(f"grid_{title}", concat_vh(grid))
    cv.waitKey(0)
    cv.destroyAllWindows()


def scale16bitImage(img, minVal, maxVal):
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0
    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img)

    return img.astype(np.float32)


def normalise216bitImage(img):
    img_16bit = np.uint16(img / img.max() * 65535)
    return img_16bit


def hpf_torch(data_normal):
    data_normal = data_normal.detach().to("cpu")
    data_img = normal2RGB_torch(data_normal).permute(1, 2, 0).numpy()
    edges = cv.Canny(data_img, 100, 150)

    shifts_extended = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
    shifts_strict = [(0, 1), (0, 0), (1, 0)]

    # shifts = [(0, 5), (0, 4), (0, 3), (0, 2), (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]
    sharp_area = np.zeros(shape=edges.shape)
    strict_sharp_area = np.zeros(shape=edges.shape)
    for (f, b) in shifts_extended:
        for (l, r) in shifts_extended:
            nf = 100000 if f == 0 else -f
            nl = 100000 if l == 0 else -l
            sharp_area += np.pad(edges, ((f, b), (l, r)), mode='constant')[b:nf, r:nl]
            if (f, b) in shifts_strict and (l, r) in shifts_strict:
                strict_sharp_area += np.pad(edges, ((f, b), (l, r)), mode='constant')[b:nf, r:nl]

    mask_sharp_part_extended = np.zeros(data_img.shape[:2])
    mask_sharp_part_extended[sharp_area > 0] = 255
    # mask the non object pixels
    mask_sharp_part_extended[np.sum(data_img, axis=2) == 0] = 0

    mask_sharp_part_strict = np.zeros(data_img.shape[:2])
    mask_sharp_part_strict[strict_sharp_area > 0] = 255
    # mask the non object pixels
    mask_sharp_part_strict[np.sum(data_img, axis=2) == 0] = 0

    return torch.from_numpy(mask_sharp_part_strict), torch.from_numpy(mask_sharp_part_extended)


def hpf(img_path, visual=False):
    img = cv.imread(img_path, 1)
    edges = cv.Canny(img, 150, 250, apertureSize=3, L2gradient=True)
    # left shift
    ls = np.pad(edges, ((0, 0), (0, 1)), mode='constant')[:, 1:]
    # right shift
    rs = np.pad(edges, ((0, 0), (1, 0)), mode='constant')[:, :-1]
    # up shift
    us = np.pad(edges, ((0, 1), (0, 0)), mode='constant')[1:, :]
    # down shift
    ds = np.pad(edges, ((1, 0), (0, 0)), mode='constant')[:-1, :]

    detail = np.zeros(img.shape)
    detail[(ls + rs + us + ds) > 0] = 255
    detail[img == 0] = 0
    img[~(detail == 255)] = 0

    if visual:
        # plot
        plt.subplot(131), plt.imshow(img, cmap='gray')
        plt.title('Original'), plt.xticks([]), plt.yticks([])

        plt.subplot(132), plt.imshow(edges, cmap='gray')
        plt.title('Edge '), plt.xticks([]), plt.yticks([])

        plt.subplot(133), plt.imshow(img, cmap='gray')
        plt.title('Detail'), plt.xticks([]), plt.yticks([])
        plt.show()

    return img


def median_filter(depth):
    padding = 1  # 2 optimal

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
                depth_padded[i, j] = np.median(neighbor)

    # remove the padding
    pred_depth = depth_padded[padding:-padding, padding:-padding]

    return pred_depth.reshape(512, 512, 1)


def median_filter_vectorize(depth):
    depth = depth.reshape((512, 512))
    neighbor = np.zeros(shape=(512, 512, 1))

    # neighbor = np.c_[
    #     neighbor, np.expand_dims(np.pad(depth, ((0, 2), (0, 2)), mode='constant')[2:, 2:], axis=2)]  # upper left
    # neighbor = np.c_[
    #     neighbor, np.expand_dims(np.pad(depth, ((0, 2), (0, 1)), mode='constant')[2:, 1:], axis=2)]  # upper left
    # neighbor = np.c_[
    #     neighbor, np.expand_dims(np.pad(depth, ((0, 2), (0, 0)), mode='constant')[2:, 0:], axis=2)]  # upper left

    shifts = [(0, 2), (0, 1), (0, 0), (1, 0), (2, 0)]
    # shifts = [(0, 5), (0, 4), (0, 3), (0, 2),    (0, 1), (0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]

    for (f, b) in shifts:
        for (l, r) in shifts:
            nf = 100000 if f == 0 else -f
            nl = 100000 if l == 0 else -l
            if (f, b) == (0, 0) and (l, r) == (0, 0):
                continue
            shift_depth = np.expand_dims(np.pad(depth, ((f, b), (l, r)), mode='constant')[b:nf, r:nl], axis=2)
            neighbor = np.c_[neighbor, shift_depth]

    # neighbor = np.c_[
    #     neighbor, np.expand_dims(np.pad(depth, ((0, 1), (0, 1)), mode='constant')[1:, 1:], axis=2)]  # upper left
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((0, 1), (0, 0)), mode='constant')[1:, :], axis=2)]  # upper
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((0, 1), (1, 0)), mode='constant')[1:, :-1], axis=2)]  # upper right
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((0, 0), (1, 0)), mode='constant')[:, :-1], axis=2)]  # right
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((1, 0), (1, 0)), mode='constant')[:-1, :-1], axis=2)]  # lower right
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((1, 0), (0, 0)), mode='constant')[:-1, :], axis=2)]  # lower
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((1, 0), (0, 1)), mode='constant')[:-1, 1:], axis=2)]  # lower left
    # neighbor = np.c_[neighbor,
    #                  np.expand_dims(np.pad(depth, ((0, 0), (0, 1)), mode='constant')[:, 1:], axis=2)]  # left

    depth_median = np.median(neighbor, axis=2)

    # if any depth is known already, then use the original one
    depth_mask = (depth != 0)
    depth_median[depth_mask] = 0
    depth_mended = depth + depth_median

    return depth_mended.reshape((512, 512, 1))


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


def eval_img_angle(output, target):
    mask = target.sum(axis=2) == 0
    angle_matrix = np.zeros(target.shape[:2])
    angle_matrix[~mask] = angle_between(target[~mask], output[~mask])

    img = angle2rgb(angle_matrix)
    img[mask] = 0
    return img, angle_matrix


def angle2rgb(angle_matrix):
    angle_matrix_8bit = cv.normalize(angle_matrix, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return cv.applyColorMap(angle_matrix_8bit, cv.COLORMAP_HOT)


def filter_bg(normal_img):
    normal_img = normal_img.astype(np.int64)
    freq = np.bincount((normal_img[:, :, 0] * 1000000 + normal_img[:, :, 1] * 1000 + normal_img[:, :, 2]).reshape(-1))
    freq_idx = np.nonzero(freq)[0]
    freq_list = np.vstack((freq_idx, freq[freq_idx])).T
    most_freq_color_sum = freq_list[freq_list[:, 1].argsort()[::-1][:15]][:, 0]
    most_freq_color_b = most_freq_color_sum % 1000
    most_freq_color_g = (most_freq_color_sum - most_freq_color_b) / 1000 % 1000
    most_freq_color_r = (((most_freq_color_sum - most_freq_color_b) / 1000 - most_freq_color_g) / 1000)
    most_freq_color = np.vstack((most_freq_color_r, most_freq_color_g, most_freq_color_b)).T

    for color in most_freq_color:
        bg_mask = (np.sum(normal_img == color, axis=2) == 3)
        normal_img[bg_mask] = 0

    return normal_img


def filter_gray_color(img):
    normal_img = img.astype(np.int64)
    low_mask = normal_img[:, :, 1] > 170
    diff = (np.abs(normal_img[:, :, 1] - normal_img[:, :, 0]) + np.abs(normal_img[:, :, 2] - normal_img[:, :, 0]))
    gray_idx = diff < 20
    mask = ~low_mask * gray_idx
    img[mask] = 0
    return img


def visual_input(depth, data, output_name):
    data['R'] = np.identity(3)
    data['t'] = np.zeros(3)
    vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                          torch.tensor(data['K']),
                          torch.tensor(data['R']).float(),
                          torch.tensor(data['t']).float())
    mask = vertex.sum(axis=2) == 0
    # move all the vertex as close to original point as possible, and noramlized all the vertex
    range_0 = vertex[:, :, :1][~mask].max() - vertex[:, :, :1][~mask].min()
    range_1 = vertex[:, :, 1:2][~mask].max() - vertex[:, :, 1:2][~mask].min()
    range_2 = vertex[:, :, 2:3][~mask].max() - vertex[:, :, 2:3][~mask].min()

    vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / range_0
    vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / range_1
    vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / range_2

    vectors = vertex
    vectors[mask] = 0

    input_torch = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
    input_torch = input_torch.permute(2, 0, 1)
    input_torch = input_torch.unsqueeze(0)
    input = tenor2numpy(input_torch[:1, :3, :, :])
    x0_normalized_8bit = normalize2_32bit(input)
    x0_normalized_8bit = image_resize(x0_normalized_8bit, width=512, height=512)

    output = cv.cvtColor(x0_normalized_8bit, cv.COLOR_RGB2BGR)
    output_name = str(f"{output_name}.png")
    cv.imwrite(output_name, output)


def normaliseVertex(vertex):
    mask = vertex.sum(axis=2) == 0

    x_range = vertex[:, :, 0][~mask].max() - vertex[:, :, 0][~mask].min()
    y_range = vertex[:, :, 1][~mask].max() - vertex[:, :, 1][~mask].min()
    z_range = vertex[:, :, 2][~mask].max() - vertex[:, :, 2][~mask].min()

    scale_factor = max(x_range, y_range, z_range)

    vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, 0][~mask].min()) / scale_factor
    vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1][~mask].min()) / scale_factor
    vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2][~mask].min()) / scale_factor

    return vertex


def output_radians_loss(output, target):
    mask = (~torch.prod(output == 0, 1).bool()).unsqueeze(1)
    loss = radians_between_2d_tensor(output, target, mask=mask).sum() / mask.sum()

    return loss


def visual_output(xout, mask):
    xout_std = filter_noise(xout, threshold=[-1, 1])
    xout_img = normal2RGB(xout_std)
    xout_img[mask] = 0
    xout_8bit = cv.normalize(xout_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return xout_8bit


def visual_normal(normal, name, histogram=True):
    normal_img = normal2RGB(normal)
    addText(normal_img, name, font_size=0.8)
    # show histogram under the image, show range of RGB color on upper right corner.
    if histogram:
        out_ranges = addHist(normal_img)
        addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)

    return normal_img


def visual_vertex(vertex, name):
    vertex_8bit = normalize2_32bit(vertex)
    vertex_8bit = image_resize(vertex_8bit, width=512, height=512)
    addText(vertex_8bit, name, font_size=0.8)
    return vertex_8bit


def visual_img(img, name, upper_right=None):
    addText(img, f"{name}")
    if upper_right is not None:
        addText(img, f"angle error: {upper_right}", pos="upper_right", font_size=0.65)
    return img


def visual_albedo(rho, name):
    img = cv.normalize(np.uint8(rho), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    img = cv.merge((img, img, img))

    addText(img, f"{name}(albedo)", font_size=0.8)
    return img
