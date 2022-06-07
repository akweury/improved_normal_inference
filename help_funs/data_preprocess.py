import glob
import json
import os
import shutil

import numpy as np
import torch

import config
from help_funs import file_io, mu


def noisy_1channel(img):
    img = img.reshape(512, 512)
    h, w = img.shape
    noise = np.random.randint(2, size=(h, w))
    noise_img = img * noise
    return noise_img


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
    for idx in range(1000):
        if os.path.exists(str(output_path / (str(idx).zfill(5) + ".depth0_noise.png"))):
            continue
        image_file, ply_file, json_file, depth_gt_file, _, normal_file = file_io.get_file_name(idx, folder_path)
        if os.path.exists(image_file):
            f = open(json_file)
            data = json.load(f)
            depth = file_io.load_scaled16bitImage(depth_gt_file, data['minDepth'], data['maxDepth'])

            # get noise mask
            img = np.expand_dims(file_io.load_16bitImage(image_file), axis=2)
            # img[img < 20] = 0
            depth = noisy_1channel(depth)
            # noise_mask = noise_mask == 0
            # input_tensor = torch.from_numpy(img.astype(np.float32))  # (depth, dtype=torch.float)
            # input_tensor = input_tensor.permute(2, 0, 1)
            #
            # img_noise = evaluate_epoch(model, input_tensor, device)
            # noise_mask = img_noise.sum(axis=2) == 0
            # add noise
            # depth[noise_mask] = 0

            # save files to the new folders
            file_io.save_scaled16bitImage(depth,
                                          str(output_path / (str(idx).zfill(5) + ".depth0_noise.png")),
                                          data['minDepth'], data['maxDepth'])
            # img_noise = mu.normalise216bitImage(img)
            # file_io.save_16bitImage(img_noise, str(output_path / (str(idx).zfill(5) + ".image0_noise.png")))
            shutil.copyfile(depth_gt_file, str(output_path / (str(idx).zfill(5) + ".depth0.png")))
            shutil.copyfile(image_file, str(output_path / (str(idx).zfill(5) + ".image0.png")))
            shutil.copyfile(json_file, str(output_path / (str(idx).zfill(5) + ".data0.json")))
            shutil.copyfile(normal_file, str(output_path / (str(idx).zfill(5) + ".normal0.png")))

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
    vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, 0][~mask].min()) / scale_factors[0]
    vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1][~mask].min()) / scale_factors[0]
    vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2][~mask].min()) / scale_factors[0]

    return vertex, scale_factors, shift_vector


def convert2training_tensor(path, k, output_type='normal'):
    if not os.path.exists(str(path)):
        raise FileNotFoundError
    if not os.path.exists(str(path / "tensor")):
        os.makedirs(str(path / "tensor"))
    if output_type == "normal_noise":
        if path == config.real_data:
            depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
        else:
            depth_files = np.array(sorted(glob.glob(str(path / "*depth0_noise.png"), recursive=True)))
            depth_gt_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
    else:
        raise ValueError("output_file is not supported. change it in args.json")

    gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
    img_files = np.array(sorted(glob.glob(str(path / "*image0.png"), recursive=True)))

    for item in range(len(data_files)):
        if os.path.exists(str(path / "tensor" / f"{str(item).zfill(5)}_{k}_{output_type}.pth.tar")):
            continue

        # input vertex
        f = open(data_files[item])
        data = json.load(f)
        f.close()

        depth = file_io.load_scaled16bitImage(depth_files[item],
                                              data['minDepth'],
                                              data['maxDepth'])
        if path == config.real_data:
            depth_gt = depth.copy()
        else:
            depth_gt = file_io.load_scaled16bitImage(depth_gt_files[item],
                                                     data['minDepth'],
                                                     data['maxDepth'])
        mask = depth.sum(axis=2) == 0
        mask_gt = depth_gt.sum(axis=2) == 0

        if k == 2:
            # depth_filtered = mu.median_filter(depth)
            depth_filtered_vectorize = mu.median_filter_vectorize(depth)
            file_io.save_scaled16bitImage(depth_filtered_vectorize,
                                          str(config.ws_path / "test.png"),
                                          data['minDepth'], data['maxDepth'])

        img = file_io.load_16bitImage(img_files[item])
        img[mask_gt] = 0
        data['R'], data['t'] = np.identity(3), np.zeros(3)
        vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())
        vertex_gt = mu.depth2vertex(torch.tensor(depth_gt).permute(2, 0, 1),
                                    torch.tensor(data['K']),
                                    torch.tensor(data['R']).float(),
                                    torch.tensor(data['t']).float())

        vertex, scale_factors, shift_vector = vectex_normalization(vertex, mask)
        vertex_gt, scale_factors, shift_vector = vectex_normalization(vertex_gt, mask_gt)

        # gt normal
        gt_normal = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
        gt_normal = gt_normal / (np.linalg.norm(gt_normal, ord=2, axis=2, keepdims=True) + 1e-20)

        # light
        light_pos = (data['lightPos'] - shift_vector) / scale_factors[0]
        light_direction = mu.vertex2light_direction(vertex, light_pos)
        light_direction_gt = mu.vertex2light_direction(vertex_gt, light_pos)
        light_direction_gt[mask_gt] = 0
        light_direction[mask] = 0

        # albedo
        G = np.sum(gt_normal * light_direction_gt, axis=-1)
        G[mask_gt] = 0
        albedo_gt = img / (G + 1e-20)

        # mu.show_images(np.uint8((albedo_gt* G)/(albedo_gt* G).max() * 255), "img")
        # albedo_gt[~mask_gt] = (albedo_gt[~mask_gt] - albedo_gt.min()) / (albedo_gt.max() - albedo_gt.min()) * 255
        # mu.show_images(albedo_gt, "a")

        target = np.c_[
            gt_normal,  # 0,1,2
            np.sum(gt_normal * light_direction_gt, axis=-1, keepdims=True),  # 3
            np.expand_dims(img, axis=2),  # 4
            light_direction_gt  # 5,6,7
        ]

        # case of resng, ng
        if k == 0:
            vertex[mask] = 0
            vectors = np.c_[vertex, np.expand_dims(img, axis=2), light_direction]
        elif k == 1:
            vectors = vertex
        elif k == 2:
            # calculate delta x, y, z of between each point and its neighbors
            vectors = neighbor_vectors_k(vertex, k)
        # detail enhanced
        elif k == 5:
            hp_mask = mu.hpf(depth_files[item], visual=True) == 0
            hp_vertex = vertex.copy()
            hp_vertex[hp_mask] = 0

            vectors = np.c_[vertex, np.expand_dims(img, axis=2), hp_vertex]
        else:
            raise ValueError

        # convert to tensor
        input_tensor = torch.from_numpy(vectors.astype(np.float32)).permute(2, 0, 1)
        gt_tensor = torch.from_numpy(target).permute(2, 0, 1)

        # save tensors
        training_case = {'input_tensor': input_tensor,
                         'gt_tensor': gt_tensor,
                         'scale_factors': scale_factors,
                         'light_source': data['lightPos'],
                         'K': data['K'],
                         'R': data['R'],
                         't': data['t'],
                         'minDepth': data['minDepth'],
                         'maxDepth': data['maxDepth'],
                         }
        torch.save(training_case, str(path / "tensor" / f"{str(item).zfill(5)}_{k}_{output_type}.pth.tar"))
        print(f'File {item + 1}/{len(data_files)} converted to tensor. K = {k}')


# def convert2training_tensor2(path, k, input_size=1000):
#     if not os.path.exists(str(path)):
#         raise FileNotFoundError
#     if not os.path.exists(str(path / "tensor")):
#         os.makedirs(str(path / "tensor"))
#
#     depth_files = np.array(sorted(glob.glob(str(path / "*depth0.png"), recursive=True)))
#     gt_files = np.array(sorted(glob.glob(str(path / "*normal0.png"), recursive=True)))
#     data_files = np.array(sorted(glob.glob(str(path / "*data0.json"), recursive=True)))
#     for item in range(len(data_files)):
#         f = open(data_files[item])
#         data = json.load(f)
#         f.close()
#
#         depth = file_io.load_scaled16bitImage(depth_files[item],
#                                               data['minDepth'],
#                                               data['maxDepth'])
#         data['R'] = np.identity(3)
#         data['t'] = np.zeros(3)
#         vertex = mu.depth2vertex(torch.tensor(depth).permute(2, 0, 1),
#                                  torch.tensor(data['K']),
#                                  torch.tensor(data['R']).float(),
#                                  torch.tensor(data['t']).float())
#         mask = vertex.sum(axis=2) == 0
#         # move all the vertex as close to original point as possible,
#         vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / vertex[:, :, :1][
#             ~mask].max()
#         vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / vertex[:, :, 1:2][
#             ~mask].max()
#         vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / vertex[:, :, 2:3][
#             ~mask].max()
#
#         # gt
#         gt = file_io.load_24bitNormal(gt_files[item]).astype(np.float32)
#         gt[mask] = 0
#         gt = gt.reshape(-1, 3)
#         gt = gt[gt.sum(axis=1) != 0]
#         # calculate delta x, y, z of between each point and its neighbors
#         vectors = neighbor_vectors_k(vertex, k)
#         vectors[mask] = 0
#         vectors = vectors[vectors.sum(axis=2) != 0]
#         vectors = vectors.reshape(-1, (((k - 1) * 2 + 1) ** 2 - 1), 3)
#         vertex = vertex.reshape(-1, 3)
#         vertex = vertex[vertex.sum(axis=1) != 0]
#         assert gt.shape[0] == vectors.shape[0] == vertex.shape[0]
#
#         # align the input to the standard shape
#         random_idx = np.random.choice(vectors.shape[0], input_size)
#         vectors = vectors[random_idx]
#         vertex = vertex[random_idx]
#         gt = gt[random_idx]
#
#         normals, idx_gt, error = mu.generate_normals_all(vectors, vertex, gt)
#
#         # visualise to check if it is correct
#         best_error_avg = np.average(error)
#         error_validition = mu.angle_between(normals, gt)
#         error_avg = np.average(error_validition)
#
#         # save vectors and idx_gt
#         input_torch = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
#         input_torch = input_torch.permute(0, 2, 1)
#
#         idx_gt = (idx_gt).astype(np.float32)
#         gt_torch = torch.from_numpy(idx_gt)  # tensor(gt, dtype=torch.float)
#         gt_torch = gt_torch.permute(1, 0)
#
#         # save tensors
#         torch.save(input_torch, str(path / "tensor" / f"{str(item).zfill(5)}_input_x.pt"))
#         torch.save(gt_torch, str(path / "tensor" / f"{str(item).zfill(5)}_gt_x.pt"))
#         print(f'File {item} converted to tensor.')


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
