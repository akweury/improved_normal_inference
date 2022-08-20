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
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--data_type', type=str, default="normal_noise", help="choose data type")
    parser.add_argument('--machine', type=str, default="local",
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # load data file names
    if args.data == "synthetic_noise":
        path = config.synthetic_data_noise_local / args.datasize / "selval" / "tensor"
    elif args.data == "synthetic_noise_dfki":
        path = config.synthetic_data_noise_dfki / args.datasize / "test" / "tensor"
    elif args.data == "synthetic_noise_pc2103":
        path = config.synthetic_data_noise_pc2103 / args.datasize / "test" / "tensor" / "tensor"
    elif args.data == "synthetic":
        path = config.synthetic_data / args.datasize / "selval" / "tensor"  # key tests 103, 166, 189,9
    elif args.data == "real":
        path = config.real_data / "test" / "tensor"  # key tests 103, 166, 189,9
    elif args.data == "real_pc2103":
        path = config.real_data_pc2103 / "test" / "tensor"  # key tests 103, 166, 189,9
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

def start2(models_path_dict):
    models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, args = preprocessing(
        models_path_dict)
    datasize = args.datasize
    # if datasize == "synthetic64":
    #     font_scale = 0.8
    # elif datasize == "synthetic128":
    #     font_scale = 0.8
    # elif datasize == "synthetic256":
    #     font_scale = 0.7
    # elif datasize == "synthetic512":
    #     font_scale = 1
    # else:
    #     raise ValueError
    font_scale = 2

    # read data
    test_0_data = np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))

    # iterate evaluate images
    for i, data_idx in enumerate(all_names):
        # read data
        test_0 = torch.load(test_0_data[i])

        # unpack model
        test_0_tensor = test_0['input_tensor'].unsqueeze(0)
        gt_tensor = test_0['gt_tensor'].unsqueeze(0)
        scaleProd_gt = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        gt = gt_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        light_gt = gt_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        light_input = test_0_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
        vertex_0 = test_0_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

        img_0 = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

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
                normal, gpu_time = svd.eval_single(vertex_0, ~mask, np.array([0, 0, -5]), farthest_neighbour=3,
                                                   data_idx=i)

                normal = normal[:, :, :3]
                normal[normal > 1] = 1
                normal[normal < -1] = -1
                normal[mask] = 0
                diff_img, diff_angle = mu.eval_img_angle(gt, normal)
                diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

                cv.imwrite(str(folder_path / f"fancy_eval_{i}_normal_{name}.png"),
                           cv.cvtColor(mu.visual_normal(normal, "", histogram=False), cv.COLOR_RGB2BGR))
                cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_{name}.png"), cv.cvtColor(
                    mu.visual_diff(gt, normal, "angle"), cv.COLOR_RGB2BGR))
                # cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_{name}.png"), cv.cvtColor(
                #     mu.visual_img(diff_img, "", upper_right=int(diff), font_scale=font_scale),
                #     cv.COLOR_RGB2BGR))

            else:
                checkpoint = torch.load(model)
                args = checkpoint['args']
                start_epoch = checkpoint['epoch']
                print(f'- model {name} evaluation... from epoch {start_epoch}')

                # load model
                device = torch.device("cuda:0")
                model = checkpoint['model'].to(device)

                if args.exp in ["light"]:
                    x_out_light = evaluate_epoch(args, model, test_0_tensor, device)
                    light_no_mask_img = cv.cvtColor(mu.visual_normal(x_out_light, "", histogram=False),
                                                    cv.COLOR_RGB2BGR)
                    x_out_light[mask] = 0
                    diff_img, diff = mu.eval_img_angle(x_out_light, light_gt)

                    output_list.append(mu.visual_light(x_out_light, name))
                    error_list.append(mu.visual_diff(x_out_light, light_gt, "angle"))
                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_light_input.png"),
                               mu.visual_light(light_input, "", histogram=False))
                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_light_gt.png"),
                               mu.visual_light(light_gt, "", histogram=False))
                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_light_{name}.png"),
                               mu.visual_light(x_out_light, "", histogram=False))
                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_light_error_{name}.png"),
                               mu.visual_diff(x_out_light, light_gt, "angle"))

                    eval_res[model_idx, i] = diff
                else:
                    xout = evaluate_epoch(args, model, test_0_tensor, device)

                    normal = xout[:, :, :3]
                    normal[normal > 1] = 1
                    normal[normal < -1] = -1
                    normal[mask] = 0
                    diff_img, diff_angle = mu.eval_img_angle(normal, gt)
                    diff = np.sum(np.abs(diff_angle)) / np.count_nonzero(diff_angle)

                    # mu.save_array(normal, str(folder_path / f"fancy_eval_{i}_normal_{name}"))
                    # if normal_no_mask_img is not None:
                    #     cv.imwrite(str(folder_path / f"fancy_eval_{i}_normal_{name}_no_mask.png"), normal_no_mask_img)

                    output_list.append(cv.cvtColor(mu.visual_normal(normal, name), cv.COLOR_RGB2BGR))
                    error_list.append(mu.visual_img(diff_img, name, upper_right=int(diff), font_scale=font_scale))

                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_normal_{name}.png"),
                               cv.cvtColor(mu.visual_normal(normal, "", histogram=False), cv.COLOR_RGB2BGR))

                    cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_{name}.png"), cv.cvtColor(
                        mu.visual_diff(gt, normal, "angle"), cv.COLOR_RGB2BGR))
                    # cv.imwrite(str(folder_path / f"fancy_eval_{i}_error_{name}.png"), cv.cvtColor(
                    #     mu.visual_img(diff_img, "", upper_right=int(diff), font_scale=font_scale),
                    #     cv.COLOR_RGB2BGR))

                    eval_res[model_idx, i] = diff

            # visual normal
            # output_list.append(mu.visual_normal(normal, name))

            # visual error
            # gt[mask_input] = 0

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
        # right_normal = mu.hconcat_resize([cv.hconcat(output_list)])
        # right_error = mu.hconcat_resize(error_list)
        # im_tile_resize = mu.hconcat_resize([left, right])

        # cv.imwrite(str(folder_path / f"fancy_eval_{i}_input.png"), left)
        # cv.imwrite(str(folder_path / f"fancy_eval_{i}_output_normal.png"), right_normal)
        # cv.imwrite(str(folder_path / f"fancy_eval_{i}_output_error.png"), right_error)

        print(f"{data_idx} has been evaluated.")


if __name__ == '__main__':
    # load test model names

    models = {
        # "light-gcnn": config.paper_exp / "light" / "checkpoint-640.pth.tar",
        # "light-noc": config.paper_exp / "light" / "checkpoint-noc-499.pth.tar",
        # "light-cnn": config.paper_exp / "light" / "checkpoint-cnn-599.pth.tar",
        #
        "SVD": None,
        # "GCNN-512": config.ws_path / "nnnn" / "output_2022-07-30_16_43_11" / "checkpoint-289.pth.tar",  # Trip Net

        # "GCNN-512-226": config.ws_path / "nnnn" / "output_2022-07-30_16_43_11" / "checkpoint-226.pth.tar",  # Trip Net
        # "GCNN-GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-03_10_04_18" / "checkpoint-850.pth.tar",  # GCNN
        # "GCNN-NOC": config.ws_path / "nnnn" / "nnnn_gcnn_noc_2022-08-03_10_00_37" / "checkpoint-894.pth.tar",
        # "GCNN-CNN": config.ws_path / "nnnn" / "nnnn_cnn_2022-08-03_09_59_25" / "checkpoint-900.pth.tar",
        #
        # "an2-8-1000": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
        # "f1": config.ws_path / "an2" / "an2_gnet-f1f_2022-07-30_22_33_05" / "checkpoint-655.pth.tar",
        # "f2": config.ws_path / "an2" / "an2_gnet-f2f_2022-08-02_01_06_20" / "model_best.pth.tar",

        # "Trip-Net-512-36": config.ws_path / "an2" / "checkpoint-36.pth.tar",  # GCNN
        # "Trip-Net-512": config.ws_path / "an2" / "checkpoint-41.pth.tar",  # GCNN
        # "An2-real-train": config.ws_path / "an_real" / "checkpoint-train-82.pth.tar",
        # "An2-real-resume": config.ws_path / "an_real" / "checkpoint-resume-444.pth.tar",
        # "GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-03_10_04_18" / "checkpoint-850.pth.tar",
        # "SVD": None,
        # "NNNN-512": config.ws_path / "nnnn" / "checkpoint-226.pth.tar",  # GCNN
        # "f3": config.ws_path / "an2" / "an2_gnet-f3f_2022-07-30_22_34_22" / "checkpoint-168.pth.tar",
        # "f4": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net

        # "an-real": config.paper_exp / "an_real" / "checkpoint-499.pth.tar",

    }

    # models = {
    #     "SVD": None,
    #     "GCNN-512-269": config.model_dfki / "checkpoint-289.pth.tar",  # GCNN
    #     "Trip-Net-512": config.model_dfki / "checkpoint-68.pth.tar",  # GCNN
    # }

    start2(models)
