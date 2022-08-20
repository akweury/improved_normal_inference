import glob

import cv2 as cv
import numpy as np
import torch

import config
from help_funs import mu, file_io


def gau_histo(gt_normal, sigma):
    gt_normal = torch.from_numpy(gt_normal)
    mask = gt_normal != 0
    min = -2
    max = 2
    bins = 100
    delta = float(max - min) / float(bins)
    centers = min + delta * (torch.arange(bins).float() + 0.5)

    output_histo = gt_normal[mask].unsqueeze(0) - centers.unsqueeze(1)
    output_histo = torch.exp(-0.5 * (output_histo / sigma) ** 2) / (sigma * np.sqrt(np.pi * 2)) * delta
    output_histo = output_histo.sum(dim=-1)
    output_histo = (output_histo - output_histo.min()) / (output_histo.max() - output_histo.min())  # normalization
    return output_histo


def load_a_training_tensor():
    test_0_data = np.array(
        sorted(glob.glob(str(config.synthetic_data_noise_local / "synthetic128" / "train" / "tensor" /
                             f"*_0_*"), recursive=True)))
    test_0 = torch.load(test_0_data[4])  # 0, 3, 5
    test_0_tensor = test_0['input_tensor'].unsqueeze(0)
    gt_tensor = test_0['gt_tensor'].unsqueeze(0)
    return test_0_tensor, gt_tensor


def load_a_training_case():
    path = config.synthetic_data_noise_local / "synthetic512" / "selval"
    image_file, ply_file, json_file, depth_gt_file, depth_noise_file, normal_file = file_io.get_file_name(0, path)

    img = file_io.load_16bitImage(image_file)
    return img


def show_img(albedo_gt, g_gt, name):
    img_recon = albedo_gt * g_gt * 255
    img_8bit = np.uint8(img_recon)
    mu.show_images(img_8bit, name)


def visual_albedo_histo(albedo_gt):
    albedo_gt_array = albedo_gt.reshape(512 * 512)
    import matplotlib.pyplot as plt

    plt.cla()
    # histr, histr_x = np.histogram(albedo_gt_array,
    #                               bins=np.arange(albedo_gt_array.min()-1, albedo_gt_array.max() + 1))

    # deterministic random data

    _ = plt.hist(albedo_gt_array, bins=10)  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    # plt.xscale('log')
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    gt_file = str(config.paper_pic / "comparison_real" / "fancy_eval_20_groundtruth.png")

    gt = file_io.load_24bitNormal(gt_file)

    training_tensor, gt_tensor = load_a_training_tensor()
    vertex = training_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    light_input = training_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    # light_input = training_tensor[:, 4:7, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    light_gt = -gt_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    # light_gt = gt_tensor[:, 4:7, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    normal_gt = gt_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
    img_gt = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).squeeze(-1).numpy()
    # img_gt = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).squeeze(-1).numpy()
    # g_gt = gt_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).squeeze(-1).numpy()

    # img_gt_norm = img_gt / 255
    # normal_gt = normal_gt / (np.linalg.norm(normal_gt, axis=-1, ord=2, keepdims=True) + 1e-20)
    # light_gt = light_gt / (np.linalg.norm(light_gt, axis=-1, ord=2, keepdims=True) + 1e-20)
    g_gt = np.sum(normal_gt * light_gt, axis=-1)
    albedo_gt = img_gt / (g_gt + 1e-20)

    # tranculation
    tranculate_threshold = 255

    albedo_gt[albedo_gt > tranculate_threshold] = tranculate_threshold
    albedo_gt[albedo_gt < -tranculate_threshold] = -tranculate_threshold

    # visual_albedo_histo(albedo_gt)
    # albedo_gt_aligned = (albedo_gt + tranculate_threshold) / (2 * tranculate_threshold)

    # convert back
    # albedo_gt_recon = albedo_gt_aligned * (2 * tranculate_threshold) - tranculate_threshold

    # show_img(albedo_gt_recon, g_gt, "aligned")
    # show_img(albedo_gt, g_gt, "gt")

    # show diff
    # albedo_gt_recon = np.expand_dims(albedo_gt_recon, axis=2)
    albedo_gt = mu.image_resize(albedo_gt, 512, 512)
    albedo_gt = np.expand_dims(albedo_gt, axis=2)
    # diff_img, diff_avg = mu.eval_albedo_diff(albedo_gt_recon, albedo_gt)
    g_gt = mu.image_resize(g_gt, 512, 512)
    mask = np.sum(g_gt, axis=-1) == 0
    g_gt_img = mu.normalize2_16bit(g_gt)

    cv.imwrite(str(config.ws_path / f"intrinsic_image.png"),
               cv.cvtColor(mu.visual_img(img_gt, ""), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_reflectance.png"),
               cv.cvtColor(mu.visual_albedo(albedo_gt, mask, "", histo=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_shading.png"),
               cv.cvtColor(mu.visual_albedo(g_gt_img, mask, "", histo=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_normal.png"),
               cv.cvtColor(mu.visual_normal(normal_gt, "", histogram=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_light.png"),
               cv.cvtColor(mu.visual_light(light_gt, "", histogram=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_vertex_input.png"),
               cv.cvtColor(mu.visual_vertex(vertex, ""), cv.COLOR_RGB2BGR))
    cv.imwrite(str(config.ws_path / f"intrinsic_image_light_input.png"),
               cv.cvtColor(mu.visual_light(light_input, "", histogram=False), cv.COLOR_RGB2BGR))
    print("ssss")
