"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import os
import argparse
import time
import glob
import cv2 as cv
import torch
import datetime
import numpy as np
from help_funs import file_io, mu
import config
from workspace.svd import eval as svd
from help_funs import data_preprocess
from help_funs import chart
from help_funs.data_preprocess import noisy_a_folder


def preprocessing():
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic', help="choose evaluate dataset")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic":
        path = config.synthetic_data_noise / "test"
    elif args.data == "real":
        path = config.real_data  # key tests 103, 166, 189,9
    else:
        raise ValueError
    # test dataset indices
    all_names = [os.path.basename(path) for path in sorted(glob.glob(str(path / f"*.image?.png"), recursive=True))]
    all_names = np.array(all_names)
    eval_indices = [int(name.split(".")[0]) for name in all_names]
    eval_indices = eval_indices[:5]

    # load test model names
    models = {
        "SVD": None,
        "Neigh_9999": config.ws_path / "nnn24" / "trained_model" / "full_normal_2999" / "checkpoint-9999.pth.tar",
        "NNNN_10082": config.ws_path / "nnnn" / "trained_model" / "output_2022-05-02_08_09_16" / "checkpoint-10082.pth.tar",
        "NG_2994": config.ws_path / "ng" / "trained_model" / "output_2022-05-08_08_17_02" / "checkpoint-2994.pth.tar",
        "NG+_5516": config.ws_path / "ng" / "trained_model" / "output_2022-05-09_21_40_33" / "checkpoint-5516.pth.tar",
    }
    eval_res = np.zeros((len(models), len(eval_indices)))

    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    folder_path = config.ws_path / "eval_output" / f"{eval_date}_{eval_time}"
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    print(f"\n\n==================== Evaluation Start =============================\n"
          f"Eval Type: Visualisation"
          f"Eval Date: {eval_date}\n"
          f"Eval Time: {eval_time}\n"
          f"Eval Objects: {len(eval_indices)}\n"
          f"Eval Models: {models.keys()}\n")
    return models, eval_indices, path, folder_path, eval_res, eval_date, eval_time


def eval_post_processing(normal, normal_img, normal_gt, name):
    out_ranges = mu.addHist(normal_img)
    mu.addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)
    mu.addText(normal_img, name, font_size=0.8)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img, diff


def evaluate(v, model_path, img):
    vertex = v.copy()
    # load model
    if model_path is None:
        normal, normal_img, eval_point_counter, total_time = svd.eval_single(v, farthest_neighbour=2)
        return normal, normal_img, eval_point_counter, total_time

    checkpoint = torch.load(model_path)
    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print('- Checkpoint was loaded successfully.')

    # load model
    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(args.gpu))
    model = checkpoint['model'].to(device)
    k = args.neighbor

    mask = vertex.sum(axis=2) == 0
    # move all the vertex as close to original point as possible, and noramlized all the vertex
    v_min, v_max = vertex[~mask].min(), vertex[~mask].max()
    v_range = v_max - v_min

    vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - v_min) / v_range
    vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - v_min) / v_range
    vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - v_min) / v_range

    # range_0 = vertex[:, :, :1][~mask].max() - vertex[:, :, :1][~mask].min()
    # range_1 = vertex[:, :, 1:2][~mask].max() - vertex[:, :, 1:2][~mask].min()
    # range_2 = vertex[:, :, 2:3][~mask].max() - vertex[:, :, 2:3][~mask].min()
    # vertex[:, :, :1][~mask] = (vertex[:, :, :1][~mask] - vertex[:, :, :1][~mask].min()) / range_0
    # vertex[:, :, 1:2][~mask] = (vertex[:, :, 1:2][~mask] - vertex[:, :, 1:2][~mask].min()) / range_1
    # vertex[:, :, 2:3][~mask] = (vertex[:, :, 2:3][~mask] - vertex[:, :, 2:3][~mask].min()) / range_2
    # calculate delta x, y, z of between each point and its neighbors
    if k >= 2:
        vectors = data_preprocess.neighbor_vectors_k(vertex, k)
    # case of ng
    elif k == 0:
        vectors = np.c_[vertex, np.expand_dims(img, axis=2)]
    elif k == 1:
        vectors = vertex
    else:
        raise ValueError

    vectors[mask] = 0

    input_tensor = torch.from_numpy(vectors.astype(np.float32))  # (depth, dtype=torch.float)
    input_tensor = input_tensor.permute(2, 0, 1)
    normal, normal_img, eval_point_counter, total_time = evaluate_epoch(model, input_tensor, start_epoch, device)

    normal_img = normal_img.astype(np.float32)
    normal_img = mu.normalize2_8bit(normal_img)

    return normal, normal_img, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, input_tensor, epoch, device):
    model.eval()  # Swith to evaluate mode
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.unsqueeze(0)
        torch.cuda.synchronize()
        # Forward Pass
        start = time.time()
        output = model(input_tensor)
        gpu_time = time.time() - start
        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)[:, :, :3]
        output = output.to('cpu').numpy()
    mask = input_tensor.sum(axis=1) == 0
    mask = mask.to('cpu').numpy().reshape(512, 512)
    eval_point_counter = np.sum(mask)

    normal = mu.filter_noise(output, threshold=[-1, 1])

    normal_8bit = mu.normal2RGB(normal)
    normal_8bit = np.ascontiguousarray(normal_8bit, dtype=np.uint8)

    return normal, normal_8bit, eval_point_counter, gpu_time


def model_eval(model_path, input, gt, name, image):
    mask = gt.sum(axis=2) == 0
    normal, img, _, _ = evaluate(input, model_path, image)
    normal[mask] = 0
    img[mask] = 0

    normal_img, angle_err_img, err = eval_post_processing(normal, img, gt, name)
    return normal_img, angle_err_img, err


def main():
    models, eval_idx, dataset_path, folder_path, eval_res, eval_date, eval_time = preprocessing()

    for i, data_idx in enumerate(eval_idx):
        # read data
        data, depth, depth_noise, normal_gt, image = file_io.load_single_data(dataset_path, idx=data_idx)

        depth_filtered = mu.median_filter(depth)
        vertex_filted_gt = mu.depth2vertex(torch.tensor(depth_filtered).permute(2, 0, 1),
                                           torch.tensor(data['K']),
                                           torch.tensor(data['R']).float(),
                                           torch.tensor(data['t']).float())
        vertex = mu.depth2vertex(torch.tensor(depth_noise).permute(2, 0, 1),
                                 torch.tensor(data['K']),
                                 torch.tensor(data['R']).float(),
                                 torch.tensor(data['t']).float())

        img_list = []
        diff_list = []

        # add ground truth
        normal_gt_img = mu.normal2RGB(normal_gt)
        mu.addText(normal_gt_img, "GT", font_size=0.8)
        img_list.append(normal_gt_img)

        # add input
        x0_normalized_8bit = mu.normalize2_8bit(vertex)
        x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
        mu.addText(x0_normalized_8bit, "Input(Vertex)")
        img_list.append(x0_normalized_8bit)

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):
            if name == "Neigh_9999" or "SVD":
                normal_img, angle_err_img, err = model_eval(model, vertex_filted_gt, normal_gt, name, image)
            else:
                normal_img, angle_err_img, err = model_eval(model, vertex, normal_gt, name, image)
            img_list.append(normal_img)
            diff_list.append(angle_err_img)
            eval_res[model_idx, i] = err

        # save the results
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        output_diff = cv.hconcat(diff_list)
        time_now = datetime.datetime.now().strftime("%H_%M_%S")
        date_now = datetime.datetime.today().date()
        cv.imwrite(str(folder_path / f"evaluation_{date_now}_{time_now}.png"), output)
        cv.imwrite(str(folder_path / f"diff{date_now}_{time_now}.png"), output_diff)
        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    main()
