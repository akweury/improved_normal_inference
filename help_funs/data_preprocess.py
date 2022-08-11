import glob
import json
import os
import shutil

import numpy as np
import torch

import config
from help_funs import file_io


def noisy_1channel(img, noise_factor=None):
    if len(img.shape) == 3:
        img = img.sum(axis=-1)
    img_shape = img.shape
    img_1d = img.reshape(img_shape[0] * img_shape[1])
    if noise_factor == None:
        noise_factor = np.random.uniform(0, 0.5)
    indices = np.random.choice(np.arange(img_1d.size), replace=False, size=int(img_1d.size * noise_factor))
    img_1d[indices] = 0
    noise_img = img_1d.reshape(img_shape)
    return noise_img, noise_factor


def noisy(img):
    h, w, c = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise = noise.reshape(h, w, 1)
    noise = np.repeat(noise, c, axis=-1)
    noise_img = img * noise
    return noise_img


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, input_tensor, device):
    model.eval()  # Swith to evaluate mode

    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)
        torch.cuda.synchronize()

        # Forward Pass
        output = model(input_tensor)

        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)[:, :, :3]
        output = output.to('cpu').numpy()

    return output


def noisy_a_folder(folder_path, output_path):
    # get noise model

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # gt_files = np.array(sorted(glob.glob(str(folder_path / "*normal0*.png"), recursive=True)))
    # depth_files = np.array(sorted(glob.glob(str(folder_path / "*depth0*.png"), recursive=True)))
    # data_files = np.array(sorted(glob.glob(str(folder_path / "*data0*.json"), recursive=True)))
    # img_files = np.array(sorted(glob.glob(str(folder_path / "*image*.png"), recursive=True)))
    for idx in range(10000):
        if os.path.exists(str(output_path / (str(idx).zfill(5) + ".image9.png"))):
            continue

        data_file = str(folder_path / (str(idx).zfill(5) + ".data0.json"))
        depth_file = str(folder_path / (str(idx).zfill(5) + ".depth0.png"))
        gt_file = str(folder_path / (str(idx).zfill(5) + ".normal0.png"))
        if os.path.exists(data_file):
            f = open(data_file)
            data = json.load(f)
            depth = file_io.load_scaled16bitImage(depth_file, data['minDepth'], data['maxDepth'])

            # get noise mask
            # img = np.expand_dims(file_io.load_16bitImage(img_files[idx]), axis=2)
            # img[img < 20] = 0
            depth, noise_factor = noisy_1channel(depth, noise_factor=0.5)

            # save files to the new folders
            file_io.save_scaled16bitImage(depth,
                                          str(output_path / (str(idx).zfill(5) + ".depth0_noise.png")),
                                          data['minDepth'], data['maxDepth'])
            # img_noise = mu.normalise216bitImage(img)
            data['noise_factor'] = noise_factor
            with open(str(output_path / (str(idx).zfill(5) + ".data0.json")), 'w') as f:
                json.dump(data, f)

            # file_io.save_16bitImage(img_noise, str(output_path / (str(idx).zfill(5) + ".image0_noise.png")))
            shutil.copyfile(depth_file, str(output_path / (str(idx).zfill(5) + ".depth0.png")))
            shutil.copyfile(gt_file, str(output_path / (str(idx).zfill(5) + ".normal0.png")))
            for i in range(10):
                img_name = (str(idx).zfill(5) + f".image{str(i)}.png")

                source_img_name = str(folder_path / (str(idx).zfill(5) + f".image{str(i)}.png"))
                if os.path.exists(source_img_name):
                    shutil.copyfile(source_img_name, str(output_path / img_name))

            print(f'File {idx} added noise.')


def neighbor_vector(vertex, xu, xd, yl, yr, zf, zb):
    vertex_padded = np.pad(vertex, ((xd, xu), (yr, yl), (zf, zb)))
    h, w, c = vertex_padded.shape
    return vertex_padded[xu:h - xd, yl:w - yr, zf:c - zb] - vertex


def neighbor_vectors_k(vertex, k=2):
    if k == 1:
        # use vertex itself as the input of network
        return vertex
    vectors = np.zeros(shape=vertex.shape)
    for i in range(k):
        for j in range(k):
            if i == j and i == 0:
                continue
            vectors = np.concatenate((vectors, neighbor_vector(vertex, 0, i, 0, j, 0, 0)), axis=2)
            if j != 0:
                vectors = np.concatenate((vectors, neighbor_vector(vertex, 0, i, j, 0, 0, 0)), axis=2)
            if i != 0:
                vectors = np.concatenate((vectors, neighbor_vector(vertex, i, 0, j, 0, 0, 0)), axis=2)
                if j != 0:
                    vectors = np.concatenate((vectors, neighbor_vector(vertex, i, 0, 0, j, 0, 0)), axis=2)

    return vectors[:, :, 3:]


def neighbor_vectors(vertex, i=1):
    delta_right = np.pad(vertex, ((0, 0), (i, 0), (0, 0)))[:, :-i, :] - vertex
    delta_left = np.pad(vertex, ((0, 0), (0, i), (0, 0)))[:, i:, :] - vertex
    delta_down = np.pad(vertex, ((i, 0), (0, 0), (0, 0)))[:-i, :, :] - vertex
    delta_up = np.pad(vertex, ((0, i), (0, 0), (0, 0)))[i:, :, :] - vertex
    delta_down_right = np.pad(vertex, ((i, 0), (i, 0), (0, 0)))[:-i, :-i, :] - vertex
    delta_up_left = np.pad(vertex, ((0, i), (0, i), (0, 0)))[i:, i:, :] - vertex
    delta_up_right = np.pad(vertex, ((0, i), (i, 0), (0, 0)))[i:, :-i, :] - vertex
    delta_down_left = np.pad(vertex, ((i, 0), (0, i), (0, 0)))[:-i, i:, :] - vertex

    vectors = np.concatenate((delta_up_left, delta_left, delta_down_left, delta_down,
                              delta_down_right, delta_right, delta_up_right, delta_up), axis=2)

    return vectors


def vectex_normalization(vertex, mask):
    # move all the vertex as close to original point as possible, and noramlized all the vertex
    x_range = vertex[:, :, 0][~mask].max() - vertex[:, :, 0][~mask].min()
    y_range = vertex[:, :, 1][~mask].max() - vertex[:, :, 1][~mask].min()
    z_range = vertex[:, :, 2][~mask].max() - vertex[:, :, 2][~mask].min()
    zzz = np.argmax(np.array([x_range, y_range, z_range]))
    scale_factors = [x_range, y_range, z_range]
    shift_vector = np.array([vertex[:, :, 0][~mask].min(), vertex[:, :, 1][~mask].min(), vertex[:, :, 2][~mask].min()])
    vertex[:, :, :1][~mask] = vertex[:, :, :1][~mask] - shift_vector[0]
    vertex[:, :, 1:2][~mask] = vertex[:, :, 1:2][~mask] - shift_vector[1]
    vertex[:, :, 2:3][~mask] = vertex[:, :, 2:3][~mask] - shift_vector[2]

    norms = (np.linalg.norm(vertex, ord=2, axis=2, keepdims=True) + 1e-20)
    vertex = vertex / norms.max()

    return vertex, norms.max(), shift_vector


if __name__ == '__main__':
    # # noisy a folder test code
    # noisy_a_folder(config.synthetic_captured_data, config.synthetic_captured_data_noise)
    for folder in ["selval", "test", "train"]:
        original_folder = config.synthetic_data / folder
        noisy_folder = config.synthetic_data_noise / folder
        noisy_a_folder(original_folder, noisy_folder)

    # # noisy test code
    # f = open(config.synthetic_captured_data / "00000.data0.json")
    # data = json.load(f)
    # f.close()
    #
    # img = file_io.load_scaled16bitImage(str(config.synthetic_captured_data / "00000.depth0.png"),
    #                                     data['minDepth'], data['maxDepth'])
    # noisy_img = noisy(img)
    #
    # cv.imshow('noise_depth', noisy_img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
