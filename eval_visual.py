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



def eval_post_processing(normal, normal_img, normal_gt, name):
    out_ranges = mu.addHist(normal_img)
    mu.addText(normal_img, str(out_ranges), pos="upper_right", font_size=0.5)
    mu.addText(normal_img, name, font_size=0.8)

    diff_img, diff_angle = mu.eval_img_angle(normal, normal_gt)
    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

    mu.addText(diff_img, f"{name}")
    mu.addText(diff_img, f"angle error: {int(diff)}", pos="upper_right", font_size=0.65)

    return normal_img, diff_img, diff


def evaluate(test_0_tensor, test_1_tensor, model_path, gt_mask):
    # load model
    if model_path is None:
        normal, normal_img, eval_point_counter, total_time = svd.eval_single(test_1_tensor, farthest_neighbour=2)
        return normal, normal_img, eval_point_counter, total_time

    checkpoint = torch.load(model_path)
    # Assign some local variables
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print('- Checkpoint was loaded successfully.')

    # load model
    device = torch.device("cuda:0")
    model = checkpoint['model'].to(device)
    k = args.neighbor

    if k == 0:
        normal, normal_img, eval_point_counter, total_time = evaluate_epoch(args, model, test_0_tensor, start_epoch,
                                                                            device, gt_mask)
    elif k == 1:
        normal, normal_img, eval_point_counter, total_time = evaluate_epoch(args, model, test_1_tensor, start_epoch,
                                                                            device, gt_mask)
    else:
        raise ValueError

    normal_img = normal_img.astype(np.float32)
    normal_cnn_8bit = cv.normalize(normal_img, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

    return normal, normal_cnn_8bit, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(args, model, input_tensor, epoch, device, gt_mask):
    model.eval()  # Swith to evaluate mode
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        torch.cuda.synchronize()
        # Forward Pass
        start = time.time()
        output = model(input_tensor)
        gpu_time = time.time() - start
        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)
        output = output.to('cpu').numpy()

    eval_point_counter = np.sum(gt_mask)
    if args.exp == "degares":
        normal = output[:, :, :3] + output[:, :, 3:6]
        normal = mu.filter_noise(normal, threshold=[-1, 1])
        normal_8bit = mu.visual_output(normal, ~gt_mask)
    else:
        normal_8bit = mu.visual_output(output[:, :, :3], ~gt_mask)
        normal = mu.filter_noise(output[:, :, :3], threshold=[-1, 1])

    normal_8bit = np.ascontiguousarray(normal_8bit, dtype=np.uint8)

    return normal, normal_8bit, eval_point_counter, gpu_time


def model_eval(model_path, test_0_tensor, test_1_tensor, gt, name):
    mask = gt.sum(axis=2) == 0
    normal, img, _, _ = evaluate(test_0_tensor, test_1_tensor, model_path, ~mask)
    normal[mask] = 0
    img[mask] = 0

    normal_img, angle_err_img, err = eval_post_processing(normal, img, gt, name)
    return normal_img, angle_err_img, err


def preprocessing():
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic', help="choose evaluate dataset")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic":
        path = config.synthetic_data_noise / "train"
    elif args.data == "real":
        path = config.real_data  # key tests 103, 166, 189,9
    elif args.data == "paper":
        path = config.paper_pic
    else:
        raise ValueError
    # test dataset indices
    all_names = [os.path.basename(path) for path in sorted(glob.glob(str(path / f"*.image?.png"), recursive=True))]
    all_names = np.array(all_names)
    # eval_indices = [int(name.split(".")[0]) for name in all_names]
    eval_indices = [6]

    # load test model names
    models = {
        # "SVD": None,
        # "Neigh_9999": config.ws_path / "nnn24" / "trained_model" / "full_normal_2999" / "checkpoint-9999.pth.tar",
        # "NNNN": config.ws_path / "nnnn" / "trained_model" / "checkpoint.pth.tar",
        # "NG": config.ws_path / "ng" / "trained_model" / "checkpoint.pth.tar",
        "NG+": config.ws_path / "resng" / "trained_model" / "checkpoint.pth.tar",
        "DeGaRes": config.ws_path / "degares" / "trained_model" / "checkpoint.pth.tar",
    }
    eval_res = np.zeros((len(models), len(eval_indices)))

    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    folder_path = config.paper_pic / f"{eval_date}_{eval_time}"
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    print(f"\n\n==================== Evaluation Start =============================\n"
          f"Eval Type: Visualisation"
          f"Eval Date: {eval_date}\n"
          f"Eval Time: {eval_time}\n"
          f"Eval Objects: {len(eval_indices)}\n"
          f"Eval Models: {models.keys()}\n")
    return models, eval_indices, path, folder_path, eval_res, eval_date, eval_time


def main():
    models, eval_idx, dataset_path, folder_path, eval_res, eval_date, eval_time = preprocessing()

    test_0_data = np.array(
        sorted(glob.glob(str(dataset_path / "tensor" / f"*_0_normal_noise.pth.tar"), recursive=True)))
    test_1_data = np.array(
        sorted(glob.glob(str(dataset_path / "tensor" / f"*_1_normal_noise.pth.tar"), recursive=True)))

    for i, data_idx in enumerate(eval_idx):
        # read data
        test_0 = torch.load(test_0_data[data_idx])
        test_1 = torch.load(test_1_data[data_idx])

        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        test_1_tensor = test_1['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)

        img_list = []
        diff_list = []

        gt = mu.tenor2numpy(gt_tensor)
        vertex_0 = mu.tenor2numpy(test_0_tensor)
        vertex_1 = mu.tenor2numpy(test_1_tensor)

        # add ground truth
        normal_gt_img = mu.normal2RGB(gt)
        out_ranges = mu.addHist(normal_gt_img)
        mu.addText(normal_gt_img, "GT", font_size=0.8)
        img_list.append(normal_gt_img)

        # add input
        # x0_normalized_8bit = mu.normalize2_8bit(vertex_0)
        # x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
        # mu.addText(x0_normalized_8bit, "Input(Vertex0)")
        # img_list.append(x0_normalized_8bit)

        # # add input
        # x0_normalized_8bit = mu.normalize2_8bit(vertex_1[:, :, :3])
        # x0_normalized_8bit = mu.image_resize(x0_normalized_8bit, width=512, height=512)
        # mu.addText(x0_normalized_8bit, "Input(Vertex1)")
        # img_list.append(x0_normalized_8bit)

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):
            normal_img, angle_err_img, err = model_eval(model, test_0_tensor, test_1_tensor, gt, name)
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
