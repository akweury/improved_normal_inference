import argparse
import datetime
import glob
import os

import cv2 as cv
import numpy as np
import torch
import matplotlib.pyplot as plt

import config
from help_funs import mu
from eval_visual import preprocessing, evaluate_epoch

models = {
    "an3-3-12-1000": config.paper_exp / "an3" / "checkpoint-3-12-1000-629.pth.tar",
    "gcnn-8-1000": config.paper_exp / "gcnn" / "checkpoint-8-1000-819.pth.tar",
}

models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, args = preprocessing(
    models)
datasize = args.datasize
font_scale = 1

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
    light_gt = gt_tensor[:, 5:8, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    light_input = test_0_tensor[:, 4:7, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    vertex_0 = test_0_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

    img_0 = test_0_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

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

        checkpoint = torch.load(model)
        args = checkpoint['args']
        start_epoch = checkpoint['epoch']
        print(f'- model {name} evaluation...')

        # load model
        device = torch.device("cuda:0")
        model = checkpoint['model'].to(device)


        def normalize_output(img):
            img = img - img.min()
            img = img / img.max()
            return img


        # Plot some images

        pred = normalize_output(light_gt)

        # fig, axarr = plt.subplots(1, 2)
        # axarr[0].imshow(img)
        # axarr[1].imshow(pred)

        # Visualize feature maps
        activation = {}


        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output

            return hook


        model.g_net.uconv4.register_forward_hook(get_activation('uconv4'))
        xout = evaluate_epoch(args, model, test_0_tensor, device)

        act = activation['uconv4'].squeeze().to("cpu")
        fig, axarr = plt.subplots(2)
        for idx in range(2):
            axarr[idx].imshow(normalize_output(act[idx]))

        plt.show()

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
