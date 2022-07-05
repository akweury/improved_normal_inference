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
    return output[:, :, :]


def preprocessing(models):
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic_noise', help="choose evaluate dataset")
    parser.add_argument('--datasize', type=str, default='synthetic128', help="choose evaluate dataset size")
    parser.add_argument('--combine', type=str, default='true', help="combine the output images in one")

    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic_noise":
        path = config.synthetic_data_noise_local / args.datasize / "selval" / "tensor"
    elif args.data == "synthetic_noise_dfki":
        path = config.synthetic_data_noise_dfki / args.datasize / "test" / "tensor"
    elif args.data == "synthetic":
        path = config.synthetic_data / args.datasize / "selval" / "tensor"  # key tests 103, 166, 189,9
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

    return models, path, output_path, eval_res, eval_date, eval_time, all_names, args


def start(models_path_dict):
    models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, args = preprocessing(
        models_path_dict)

    data_size = args.datasize

    test_0_data = np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))

    # iterate evaluate images
    for i, data_idx in enumerate(all_names):
        # read data
        test_0 = torch.load(test_0_data[i])

        # unpack model
        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)
        gt = mu.tenor2numpy(gt_tensor[:, :3, :, :])
        light_gt = mu.tenor2numpy(gt_tensor[:, 5:8, :, :])
        vertex_0 = mu.tenor2numpy(test_0_tensor[:, :3, :, :])[:, :, :3]
        img_0 = mu.tenor2numpy(test_0_tensor[:, 3:4, :, :])
        mask = gt.sum(axis=2) == 0
        mask_input = vertex_0.sum(axis=2) == 0
        rho_gt = mu.albedo(img_0, gt, light_gt)

        img_list, diff_list, albedo_list, albedo_diff_list = [mu.visual_vertex(vertex_0, "Input(Vertex)"),
                                                              mu.visual_normal(gt, "GT")], [], [
                                                                 mu.visual_albedo(rho_gt, 'GT')], []
        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):

            # normal, img, _, _ = evaluate(test_0_tensor, test_1_tensor, model, ~mask)

            # load model
            if name == "SVD":
                print(f'- model {name} evaluation...')
                normal = svd.eval_single(vertex_0, mask, np.array([0, 0, 7.5]), farthest_neighbour=2)

            else:
                checkpoint = torch.load(model)
                args = checkpoint['args']
                start_epoch = checkpoint['epoch']
                print(f'- model {name} evaluation...')

                # load model
                device = torch.device("cuda:0")
                model = checkpoint['model'].to(device)
                k = args.neighbor
                if args.exp == "ag":
                    xout = evaluate_epoch(args, model, test_0_tensor, device)
                    normal = xout[:, :, :3]
                    light = xout[:, :, 3:6]
                    light[mask] = 0
                    rho = mu.albedo(img_0, normal, light)
                    albedo_list.append(mu.visual_albedo(rho, name))

                    # visual albedo error
                    diff_albedo = np.abs(np.uint8(rho) - np.uint8(rho_gt))
                    diff_albedo_img = cv.applyColorMap(
                        cv.normalize(diff_albedo, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U),
                        cv.COLORMAP_HOT)
                    diff_albedo_avg = np.sum(diff_albedo) / np.count_nonzero(diff_albedo)
                    # mu.addText(diff_albedo_img, name)
                    # mu.addText(diff_albedo_img, f"error: {int(diff_albedo_avg)}", pos="upper_right", font_size=0.65)
                    albedo_diff_list.append(diff_albedo_img)


                elif args.exp == "ng":
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :4, :, :], device)
                elif args.exp == "nnnn":
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :3, :, :], device)
                else:
                    raise ValueError

            normal[mask] = 0

            # visual normal
            img_list.append(mu.visual_normal(normal, name))

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
        output_diff_albedo = cv.hconcat(albedo_diff_list)

        cv.imwrite(str(folder_path / f"eval_{i}_normal.png"), output)
        cv.imwrite(str(folder_path / f"eval_{i}_error.png"), output_diff)
        cv.imwrite(str(folder_path / f"eval_{i}_albedo.png"), output_albedo)
        cv.imwrite(str(folder_path / f"eval_{i}_albedo_error.png"), output_diff_albedo)

        print(f"{data_idx} has been evaluated.")


def start2(models_path_dict):
    models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, args = preprocessing(
        models_path_dict)
    datasize = args.datasize
    if datasize == "synthetic64":
        font_scale = 0.8
    elif datasize == "synthetic128":
        font_scale = 0.8
    elif datasize == "synthetic256":
        font_scale = 0.7
    elif datasize == "synthetic512":
        font_scale = 1
    else:
        raise ValueError

    # read data
    test_0_data = np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))

    # iterate evaluate images
    for i, data_idx in enumerate(all_names):
        # read data
        test_0 = torch.load(test_0_data[i])

        # unpack model
        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)
        gt = gt_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        light_gt = gt_tensor[:, 5:8, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        vertex_0 = test_0_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        img_0 = test_0_tensor[:, 3, :, :].permute(1, 2, 0).squeeze(-1).numpy()

        mask = gt.sum(axis=2) == 0
        # mask_input = vertex_0.sum(axis=2) == 0
        # rho_gt = mu.albedo(img_0, gt, light_gt)

        img_8bit = cv.normalize(np.uint8(img_0), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        img = cv.merge((img_8bit, img_8bit, img_8bit))
        normal_no_mask_img = None
        # black_img = cv.normalize(np.uint8(np.zeros(shape=(512, 512, 3))), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        # gt_img = cv.cvtColor(mu.visual_normal(gt, "GT"), cv.COLOR_RGB2BGR)
        input_list, output_list, error_list, albedo_list, albedo_diff_list = [], [], [], [], []
        # cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_gt.png"), black_img)

        # evaluate CNN models
        for model_idx, (name, model) in enumerate(models.items()):

            # normal, img, _, _ = evaluate(test_0_tensor, test_1_tensor, model, ~mask)

            # load model
            if name == "SVD":
                print(f'- model {name} evaluation...')
                normal, gpu_time = svd.eval_single(vertex_0, ~mask, np.array([0, 0, -7]), farthest_neighbour=1)
            else:
                checkpoint = torch.load(model)
                args = checkpoint['args']
                start_epoch = checkpoint['epoch']
                print(f'- model {name} evaluation...')

                # load model
                device = torch.device("cuda:0")
                model = checkpoint['model'].to(device)
                if args.exp in ["ng", "hfm", "resng"]:
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :4, :, :], device)
                elif args.exp in ["ag"]:
                    xout = evaluate_epoch(args, model, test_0_tensor, device)

                    normal = xout[:, :, :3]
                    xout_albedo = xout[:, :, 3]
                    xout_g = xout[:, :, 5]

                    # visual albedo error
                    # rho_gt = mu.albedo(img_0, gt, light_gt)
                    # xout_albedo = np.uint8(xout_albedo)
                    # output_list.append(mu.visual_img(xout_albedo, "albedo_out"))

                    xout_albedo2 = np.uint8(img_0 / (xout_g + 1e-20))
                    output_list.append(mu.visual_img(xout_albedo2, "albedo_out"))

                    g = np.sum(gt * light_gt, axis=-1)
                    albedo_gt = np.uint8(img_0 / (g + 1e-20))
                    output_list.append(mu.visual_img(albedo_gt, "albedo_gt"))

                    img_out = xout_albedo2 * xout_g
                    img_out[mask] = 0
                    img_out = np.uint8(img_out)
                    img_out = mu.visual_img(img_out, "img_out")
                    output_list.append(img_out)
                    output_list.append(mu.visual_img(img_8bit, "img_gt"))

                    diff_albedo = np.abs(xout_albedo2 - albedo_gt)
                    diff_albedo_img = cv.applyColorMap(cv.normalize(diff_albedo, None, 0, 255,
                                                                    cv.NORM_MINMAX, dtype=cv.CV_8U),
                                                       cv.COLORMAP_HOT)
                    diff_albedo_avg = np.sum(diff_albedo) / np.count_nonzero(diff_albedo)
                    # mu.addText(diff_albedo_img, name)
                    # mu.addText(diff_albedo_img, f"error: {int(diff_albedo_avg)}", pos="upper_right", font_size=0.65)
                    error_list.append(
                        mu.visual_img(diff_albedo_img, name, upper_right=int(diff_albedo_avg), font_scale=font_scale))


                elif args.exp in ["nnnn", "fugrc"]:
                    normal = evaluate_epoch(args, model, test_0_tensor[:, :3, :, :], device)
                    normal_no_mask_img = cv.cvtColor(mu.visual_normal(normal, "", histogram=False), cv.COLOR_RGB2BGR)
                    normal[mask] = 0
                else:
                    raise ValueError

            normal[mask] = 0

            # visual normal
            # output_list.append(mu.visual_normal(normal, name))

            # visual error
            # gt[mask_input] = 0
            diff_img, diff_angle = mu.eval_img_angle(normal, gt)
            diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

            # mu.save_array(normal, str(folder_path / f"fancy_eval_{i}_normal_{name}"))
            # if normal_no_mask_img is not None:
            #     cv.imwrite(str(folder_path / f"fancy_eval_{i}_normal_{name}_no_mask.png"), normal_no_mask_img)
            output_list.append(cv.cvtColor(mu.visual_normal(normal, name), cv.COLOR_RGB2BGR))
            error_list.append(mu.visual_img(diff_img, name, upper_right=int(diff), font_scale=font_scale))

            cv.imwrite(str(folder_path / f"fancy_eval_{i}_normal_{name}.png"),
                       cv.cvtColor(mu.visual_normal(normal, "", histogram=False), cv.COLOR_RGB2BGR))
            cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_{name}.png"),
                       mu.visual_img(diff_img, "", upper_right=int(diff), font_scale=font_scale))

            eval_res[model_idx, i] = diff

        # save files
        # mu.save_array(gt, str(folder_path / f"fancy_eval_{i}_normal_gt"))

        output_list.append(cv.cvtColor(mu.visual_normal(gt, "GT"), cv.COLOR_RGB2BGR))
        output_list.append(mu.visual_vertex(vertex_0, "Input_Vertex"))
        cv.imwrite(str(folder_path / f"fancy_eval_{i}_img.png"), mu.visual_img(img, "", font_scale=font_scale))
        cv.imwrite(str(folder_path / f"fancy_eval_{i}_groundtruth.png"),
                   cv.cvtColor(mu.visual_normal(gt, "", histogram=False), cv.COLOR_RGB2BGR), )
        cv.imwrite(str(folder_path / f"fancy_eval_{i}_point_cloud_noise.png"),
                   mu.visual_vertex(vertex_0, ""))

        # save the results

        # output = cv.cvtColor(cv.hconcat(output_list), cv.COLOR_RGB2BGR)
        # left = mu.hconcat_resize(input_list)
        right_normal = mu.hconcat_resize([cv.hconcat(output_list)])
        right_error = mu.hconcat_resize(error_list)
        # im_tile_resize = mu.hconcat_resize([left, right])

        # cv.imwrite(str(folder_path / f"fancy_eval_{i}_input.png"), left)
        cv.imwrite(str(folder_path / f"fancy_eval_{i}_output_normal.png"), right_normal)
        cv.imwrite(str(folder_path / f"fancy_eval_{i}_output_error.png"), right_error)

        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    # load test model names

    models = {
        # "SVD": None,
        # "GCNN-64": config.ws_path / "resng" / "trained_model" / "64" / "checkpoint.pth.tar",  # image guided
        "GCNN3-32-512": config.ws_path / "resng" / "trained_model" / "512" / "checkpoint-3-32.pth.tar",
        "GCNN3-32-512-2": config.ws_path / "resng" / "trained_model" / "512" / "checkpoint-3-32-2.pth.tar",
        # "GCNN3-64-512": config.ws_path / "resng" / "trained_model" / "512" / "checkpoint-3-64.pth.tar",

        "AG": config.ws_path / "ag" / "trained_model" / "512" / "checkpoint.pth.tar",  # with light direction
        # "GCNN": config.ws_path / "nnnn" / "trained_model" / "512" / "checkpoint.pth.tar",
        # "FUGRC": config.ws_path / "fugrc" / "trained_model" / "128" / "checkpoint-608.pth.tar",

    }
    start2(models)
