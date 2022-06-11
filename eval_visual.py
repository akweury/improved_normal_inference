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

    if args.exp == "degares":
        normal = output[:, :, :3] + output[:, :, 3:6]
        normal = mu.filter_noise(normal, threshold=[-1, 1])
    else:
        normal = mu.filter_noise(output[:, :, :3], threshold=[-1, 1])

    # normal_8bit = np.ascontiguousarray(normal_8bit, dtype=np.uint8)

    return normal


def preprocessing(models):
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic', help="choose evaluate dataset")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic":
        path = config.synthetic_data_noise / "test" / "tensor"
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

    return models, path, output_path, eval_res, eval_date, eval_time, all_names


def start(models_path_dict):
    models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names = preprocessing(models_path_dict)
    test_0_data = np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))

    # iterate evaluate images
    for i, data_idx in enumerate(all_names):
        # read data
        test_0 = torch.load(test_0_data[i])

        # unpack model
        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)
        gt = mu.tenor2numpy(gt_tensor[:, :3, :, :])
        vertex_0 = mu.tenor2numpy(test_0_tensor[:, :3, :, :])[:, :, :3]
        img_0 = mu.tenor2numpy(test_0_tensor[:, 3:4, :, :])
        mask = gt.sum(axis=2) == 0
        mask_input = vertex_0.sum(axis=2) == 0

        img_list, diff_list, albedo_list = [mu.visual_vertex(vertex_0, "Input(Vertex)"),
                                            mu.visual_normal(gt, "GT")], [], []
        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):

            # normal, img, _, _ = evaluate(test_0_tensor, test_1_tensor, model, ~mask)

            # load model
            if name == "SVD":
                normal = svd.eval_single(vertex_0, farthest_neighbour=2)

            else:
                checkpoint = torch.load(model)
                args = checkpoint['args']
                start_epoch = checkpoint['epoch']
                print(f'- model {name} evaluation...')

                # load model
                device = torch.device("cuda:0")
                model = checkpoint['model'].to(device)
                k = args.neighbor

                if k == 0:
                    if args.exp == "ag":
                        normal = evaluate_epoch(args, model, test_0_tensor, device)
                    else:
                        normal = evaluate_epoch(args, model, test_0_tensor[:, :4, :, :], device)
                elif k == 1:
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :3, :, :], device)
                else:
                    raise ValueError
            # visual normal
            # normal[mask_input] = 0
            img_list.append(mu.visual_normal(normal, name))

            # albedo
            L = mu.vertex2light_direction(vertex_0, test_0['light_source'])
            rho = mu.albedo(img_0, normal, L)
            albedo_list.append(mu.visual_albedo(rho, name))

            # visual error
            gt[mask_input] = 0
            diff_img, diff_angle = mu.eval_img_angle(normal, gt)
            diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)
            diff_list.append(mu.visual_img(diff_img, name, upper_right=int(diff)))
            eval_res[model_idx, i] = diff

        # save the results
        output = cv.cvtColor(cv.hconcat(img_list), cv.COLOR_RGB2BGR)
        output_diff = cv.hconcat(diff_list)
        output_albedo = cv.hconcat(albedo_list)

        cv.imwrite(str(folder_path / f"eval_{i}_normal.png"), output)
        cv.imwrite(str(folder_path / f"eval_{i}_error.png"), output_diff)
        cv.imwrite(str(folder_path / f"eval_{i}_albedo.png"), output_albedo)

        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    # load test model names

    models = {
        # "SVD": None,
        "NNNN": config.ws_path / "nnnn" / "trained_model" / "checkpoint.pth.tar",
        # "NG": config.ws_path / "ng" / "trained_model" / "checkpoint.pth.tar",
        "AG": config.ws_path / "ag" / "trained_model" / "model_best.pth.tar",
        # "AG2": config.ws_path / "ag" / "trained_model" / "checkpoint-2600.pth.tar",
        # "NG+": config.ws_path / "resng" / "trained_model" / "checkpoint.pth.tar",
        # "NNNN+ResNet": config.ws_path / "resng" / "trained_model" / "checkpoint-6693.pth.tar"

    }
    start(models)
