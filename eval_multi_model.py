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


def start(models_path_dict, infos, s_window_length=32):
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

        w, h = test_0_tensor.size(2), test_0_tensor.size(3)

        left = int(w / 2 - s_window_length)
        right = int(w / 2 + s_window_length)
        below = int(h / 2 - s_window_length)
        top = int(h / 2 + s_window_length)
        s_gt = gt[left: right, below:top, :]
        s_input = test_0_tensor[:, :, left: right, below:top]
        s_input_numpy = s_input.clone()
        s_input_numpy = s_input_numpy[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

        s_mask = s_gt.sum(axis=2) == 0

        cv.imwrite(str(folder_path / f"eval_{i}_input.png"),
                   cv.cvtColor(mu.visual_vertex(s_input_numpy,
                                                ""),
                               cv.COLOR_RGB2BGR))

        cv.imwrite(str(folder_path / f"eval_{i}_normal_GT.png"),
                   cv.cvtColor(mu.visual_normal(s_gt, "", histogram=False),
                               cv.COLOR_RGB2BGR))

        if infos["title"]:
            img_list = [mu.visual_normal(s_gt, "GT", histogram=infos["histo"])]
        else:
            img_list = [mu.visual_normal(s_gt, "", histogram=infos["histo"])]

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):
            # load model
            # diff_list = []
            if name == "SVD":
                print(f'- model {name} evaluation...')
                normal, gpu_time = svd.eval_single(s_input_numpy, ~s_mask, np.array([0, 0, -7]), farthest_neighbour=2)
            else:
                checkpoint = torch.load(model)
                args = checkpoint['args']
                start_epoch = checkpoint['epoch']
                device = torch.device("cuda:0")
                model = checkpoint['model'].to(device)
                print(f'- model {name} evaluation...')

                if name == "s-window":
                    normal = evaluate_epoch(args, model, s_input[:, :3, :, :], device)
                else:
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :3, :, :], device)
                    normal = normal[left: right, below:top, :]

            normal[s_mask] = 0

            cv.imwrite(str(folder_path / f"eval_{i}_normal_{name}.png"),
                       cv.cvtColor(mu.visual_normal(normal, "", histogram=False),
                                   cv.COLOR_RGB2BGR))

            if infos["title"]:
                img_list.append(mu.visual_normal(normal, name, histogram=infos["histo"]))
            else:
                img_list.append(mu.visual_normal(normal, "", histogram=infos["histo"]))
            # visual error

            diff_img, diff_angle = mu.eval_img_angle(normal, s_gt)
            diff = np.sum(diff_angle) / (np.count_nonzero(diff_angle) + 1e-20)

            cv.imwrite(str(folder_path / f"eval_{i}_error_{name}.png"),
                       cv.cvtColor(mu.visual_img(diff_img, "", upper_right=int(diff), font_scale=1),
                                   cv.COLOR_RGB2BGR))

            if infos["title"]:
                img_list.append(mu.visual_img(diff_img, name, upper_right=int(diff), font_scale=1))
            else:
                img_list.append(mu.visual_img(diff_img, "", upper_right=int(diff), font_scale=1))

            # diff_angle = np.uint8(diff_angle)
            # if diff_list is None:
            #     diff_list = diff_angle
            # else:
            #     diff_list = np.c_[diff_list, diff_angle]
            eval_res[model_idx, i] = diff

        # save the results
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        # diff_list = cv.normalize(diff_list, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)

        # diff_img = cv.applyColorMap(diff_list, cv.COLORMAP_JET)

        # dmask = np.c_[mask]
        # diff_img[dmask] = 0
        # mu.show_images(diff_img,"ss")
        cv.imwrite(str(folder_path / f"eval_{i}_normal.png"), output)
        # cv.imwrite(str(folder_path / f"eval_{i}_error.png"),  mu.hconcat_resize(diff_list))
        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    # load test model names

    models = {
        "SVD": None,
        # "GCNN": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint.pth.tar",  # image guided
        # "AG": config.ws_path / "ag" / "trained_model" / "128" / "checkpoint.pth.tar",  # with light direction
        # "s-window": config.ws_path / "resng" / "trained_model" / "64" / "checkpoint.pth.tar",
        "GCNN": config.ws_path / "resng" / "trained_model" / "512" / "checkpoint-3-32.pth.tar",
        # "FUGRC": config.ws_path / "fugrc" / "trained_model" / "128" / "checkpoint-608.pth.tar",

    }
    infos = {
        "title": False,
        "histo": False,
        "error": False
    }
    start(models, infos=infos)
