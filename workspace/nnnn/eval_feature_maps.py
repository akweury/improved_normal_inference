import glob

import cv2 as cv
import numpy as np
import torch

import config
from eval_visual import preprocessing, evaluate_epoch
from help_funs import mu

models = {
    # "gcnn-cnn": config.gcnn_cnn,
    # "gcnn-noc": config.gcnn_noc,
    "gcnn-gcnn": config.gcnn_gcnn,
}

models, dataset_path, folder_path, eval_res, eval_date, eval_time, all_names, args = preprocessing(
    models)
datasize = args.datasize
font_scale = 1

# read data
test_0 = torch.load(np.array(sorted(glob.glob(str(dataset_path / f"*_0_*"), recursive=True)))[3])

# unpack model
test_0_tensor = test_0['input_tensor'].unsqueeze(0)
gt_tensor = test_0['gt_tensor'].unsqueeze(0)

gt = gt_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
light_gt = gt_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
light_input = test_0_tensor[:, 13:16, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
vertex_input = test_0_tensor[:, :3, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()
img_gt = test_0_tensor[:, 3:4, :, :].permute(2, 3, 1, 0).squeeze(-1).numpy()

mask = gt.sum(axis=2) == 0

img_8bit = cv.normalize(np.uint8(img_gt), None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
img = cv.merge((img_8bit, img_8bit, img_8bit))
normal_no_mask_img = None
FEATURE_MAP_NUM = 128
CV_COLOR = cv.COLORMAP_RAINBOW

# evaluate CNN models
for model_idx, (name, model) in enumerate(models.items()):

    checkpoint = torch.load(model)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    print(f'- model {name} evaluation...')

    # load model
    device = torch.device("cuda:0")
    model = checkpoint['model'].to(device)


    # https://discuss.pytorch.org/t/visualize-feature-map/29597/2
    def normalize_output(img):
        img = img - img.min()
        img = img / img.max()
        return img


    # Plot some images
    pred = normalize_output(light_gt)
    # Visualize feature maps
    activation = {}


    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output

        return hook


    feature_map_dconv1 = []
    feature_map_dconv2 = []
    feature_map_dconv3 = []
    feature_map_dconv4 = []

    feature_map_uconv1 = []
    feature_map_uconv2 = []
    feature_map_uconv3 = []

    feature_map_conv1 = []
    feature_map_conv2 = []

    model.normal_net.dconv1.register_forward_hook(get_activation('dconv1'))
    model.normal_net.dconv2.register_forward_hook(get_activation('dconv2'))
    model.normal_net.dconv3.register_forward_hook(get_activation('dconv3'))
    model.normal_net.dconv4.register_forward_hook(get_activation('dconv4'))

    model.normal_net.uconv1.register_forward_hook(get_activation('uconv1'))
    model.normal_net.uconv2.register_forward_hook(get_activation('uconv2'))
    model.normal_net.uconv3.register_forward_hook(get_activation('uconv3'))

    model.normal_net.conv1.register_forward_hook(get_activation('conv1'))
    model.normal_net.conv2.register_forward_hook(get_activation('conv2'))

    xout = evaluate_epoch(args, model, test_0_tensor, device)

    act = activation['dconv1'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_dconv1.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['dconv2'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_dconv2.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['dconv3'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_dconv3.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['dconv4'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_dconv4.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['uconv1'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_uconv1.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['uconv2'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_uconv2.append(cv.applyColorMap(feature_map, CV_COLOR))

    act = activation['uconv3'].squeeze().to("cpu")
    for idx in range(FEATURE_MAP_NUM):
        feature_map = act[idx].unsqueeze(-1).numpy()
        feature_map = mu.visual_vertex(feature_map, str(idx))
        feature_map_uconv3.append(cv.applyColorMap(feature_map, CV_COLOR))

    feature_map_list = [
        # feature_map_dconv1, feature_map_dconv2, feature_map_dconv3, feature_map_dconv4,
        feature_map_uconv1, feature_map_uconv2, feature_map_uconv3
    ]
    xout[mask] = 0
    xout[xout > 1] = 1
    xout[xout < -1] = -1
    # save the results
    feature_map_img = mu.concat_vh(feature_map_list)
    cv.imwrite(str(folder_path / f"feature_map_{name}.png"), feature_map_img)
    cv.imwrite(str(folder_path / f"feature_map_normal_{name}.png"),
               cv.cvtColor(mu.visual_normal(gt, "GT", histogram=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(folder_path / f"feature_map_out_{name}.png"),
               cv.cvtColor(mu.visual_normal(xout, "Out", histogram=False), cv.COLOR_RGB2BGR))
    cv.imwrite(str(folder_path / f"feature_map_err_{name}.png"),
               mu.visual_diff(xout, gt, "angle"))
    cv.imwrite(str(folder_path / f"feature_map_vertex_{name}.png"),
               cv.cvtColor(mu.visual_vertex(vertex_input, "Vertex"), cv.COLOR_RGB2BGR))
