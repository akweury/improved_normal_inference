"""
input: an image, the incomplete depth map of the image
output: predicted normal map
"""
import argparse
import datetime
import glob
import os

import cv2 as cv
import numpy as np
import torch

import config
from help_funs import mu
from workspace.svd import eval as svd


############ EVALUATION FUNCTION ############
def evaluate_epoch(args, model, input_tensor, device):
    model.eval()  # Swith to evaluate mode
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        torch.cuda.synchronize()
        # Forward Pass
        output = model(input_tensor)
        # store the predicted normal
        output = output[0, :].permute(1, 2, 0)
        output = output.to('cpu').numpy()
        output[output > 1] = 1
        output[output < -1] = -1
    return output[:, :, :3]


def preprocessing(models):
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic_noise', help="choose evaluate dataset")
    parser.add_argument('--datasize', type=str, default='synthetic128', help="choose evaluate dataset size")

    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic_noise":
        path = config.synthetic_data_noise_local / args.datasize / "selval" / "tensor"
    elif args.data == "synthetic_noise_dfki":
        path = config.synthetic_data_noise_dfki / args.datasize / "test" / "tensor"
    elif args.data == "synthetic":
        path = config.synthetic_data / "test" / "tensor"  # key tests 103, 166, 189,9
    elif args.data == "real":
        path = config.real_data / "tensor"  # key tests 103, 166, 189,9
    elif args.data == "paper":
        path = config.paper_pic
    else:
        raise ValueError
    # test dataset indices
    all_names = [os.path.basename(path) for path in sorted(glob.glob(str(path / f"*_0_*"), recursive=True))]
    all_names = np.array(all_names)
    # eval_indices = [6]

    eval_res = np.zeros((len(models), len(all_names)))

    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    output_path = config.ws_path / "eval_output" / f"{eval_date}_{eval_time}"
    if not os.path.exists(str(output_path)):
        os.mkdir(str(output_path))

    print(f"\n\n==================== Evaluation Start =============================\n"
          f"Eval Type: Visualisation"
          f"Eval Date: {eval_date}\n"
          f"Eval Time: {eval_time}\n"
          f"Eval Models: {models.keys()}\n")

    return models, path, output_path, eval_res, eval_date, eval_time, all_names, args.datasize


def get_s_image(test_0_tensor, s_window_length, img, x_shift, y_shift):
    w, h = test_0_tensor.size(2), test_0_tensor.size(3)

    left = int(w / 2 - s_window_length)
    right = int(w / 2 + s_window_length)
    below = int(h / 2 - s_window_length)
    top = int(h / 2 + s_window_length)
    s_img = img[left + x_shift: right + x_shift, below + y_shift:top + y_shift, :]
    return s_img


def start(models_path_dict, infos, s_window_length=16):
    models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, data_size = preprocessing(
        models_path_dict)
    test_0_data = np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))
    # iterate evaluate images
    for i, data_idx in enumerate(all_names):
        # read data
        test_0 = torch.load(test_0_data[i])

        # unpack model
        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)
        gt = gt_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        img = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        vertex = test_0_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

        w, h = test_0_tensor.size(2), test_0_tensor.size(3)

        left = int(w / 2 - s_window_length)
        right = int(w / 2 + s_window_length)
        below = int(h / 2 - s_window_length)
        top = int(h / 2 + s_window_length)

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):

            checkpoint = torch.load(model)
            args = checkpoint['args']
            device = torch.device("cuda:0")
            model = checkpoint['model'].to(device)
            print(f'- model {name} evaluation...')

            normal_full = evaluate_epoch(args, model, test_0_tensor[:, :, :, :], device)

            for s_i in range(-48, 48, 10):
                for s_j in range(-48, 48, 10):
                    normal = normal_full[left + s_i: right + s_i, below + s_j:top + s_j, :]
                    s_gt = gt[left + s_i: right + s_i, below + s_j:top + s_j, :]
                    s_mask = s_gt.sum(axis=2) == 0
                    normal[s_mask] = 0

                    cv.imwrite(str(folder_path / f"eval_{i}_{s_i}_{s_j}_normal.png"),
                               cv.cvtColor(mu.visual_normal(normal, "", histogram=False),
                                           cv.COLOR_RGB2BGR))
                    cv.imwrite(str(folder_path / f"eval_{i}_{s_i}_{s_j}_gt.png"),
                               cv.cvtColor(mu.visual_normal(s_gt, "", histogram=False),
                                           cv.COLOR_RGB2BGR))

                    # visual error

                    diff_img, diff_angle = mu.eval_img_angle(normal, s_gt)
                    diff = np.sum(diff_angle) / (np.count_nonzero(diff_angle) + 1e-20)
                    cv.imwrite(str(folder_path / f"eval_{i}_{s_i}_{s_j}_error.png"),
                               cv.cvtColor(mu.visual_diff(normal, s_gt, "angle"), cv.COLOR_RGB2BGR))

            # eval_res[model_idx, i] = diff

        # save the results

        s_gt = get_s_image(test_0_tensor, s_window_length, gt, 0, 0)
        s_img = get_s_image(test_0_tensor, s_window_length, img, 0, 0)
        s_input_numpy_3ch = get_s_image(test_0_tensor, s_window_length, vertex, 0, 0)
        s_mask = s_gt.sum(axis=2) == 0

        cv.imwrite(str(folder_path / f"eval_{i}_s_normal_GT.png"),
                   cv.cvtColor(mu.visual_normal(s_gt, "", histogram=False),
                               cv.COLOR_RGB2BGR))
        cv.imwrite(str(folder_path / f"eval_{i}_normal_GT.png"),
                   cv.cvtColor(mu.visual_normal(gt, "", histogram=False),
                               cv.COLOR_RGB2BGR))
        if infos["title"]:
            img_list = [mu.visual_normal(s_gt, "GT", histogram=infos["histo"])]
        else:
            img_list = [mu.visual_normal(s_gt, "", histogram=infos["histo"])]

        cv.imwrite(str(folder_path / f"eval_{i}_input.png"),
                   cv.cvtColor(mu.visual_vertex(s_input_numpy_3ch, ""), cv.COLOR_RGB2BGR))

        cv.imwrite(str(folder_path / f"eval_{i}_img.png"), mu.visual_img(s_img, ""))
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        cv.imwrite(str(folder_path / f"eval_{i}_normal.png"), output)

        # cv.imwrite(str(folder_path / f"eval_{i}_error.png"),  mu.hconcat_resize(diff_list))
        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    # load test model names

    models = {
        # "light-gcnn": config.paper_exp / "light" / "checkpoint-640.pth.tar",
        # "light-noc": config.paper_exp / "light" / "checkpoint-noc-499.pth.tar",
        # "light-cnn": config.paper_exp / "light" / "checkpoint-cnn-599.pth.tar",
        #
        # "SVD": None,
        #
        # "GCNN-GCNN": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",  # GCNN
        # "GCNN-NOC": config.paper_exp / "gcnn" / "checkpoint-noc-807.pth.tar",
        # "GCNN-CNN": config.paper_exp / "gcnn" / "checkpoint-cnn-695.pth.tar",

        "an2": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
        # "an-8-1000": config.paper_exp / "an" / "checkpoint-818.pth.tar",

    }
    infos = {
        "title": False,
        "histo": False,
        "error": False
    }
    start(models, infos=infos)
