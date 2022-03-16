# !/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import json
import torch
import numpy as np
import scipy.spatial as spatial
import config

from help_funs import mu
from workspace.deepfit import DeepFit

# compute normal vectors of a single point cloud
import sys

sys.path.insert(0, '../utils')
sys.path.insert(0, '../models')
sys.path.insert(0, '../trained_models')
import torch
import argparse

from workspace.deepfit import tutorial_utils as tu


# load the point cloud
# point_cloud_dataset = tu.SinglePointCloudDataset(args.input, points_per_patch=args.k_neighbors)
# dataloader = torch.utils.data.DataLoader(point_cloud_dataset, batch_size=128, num_workers=8)


def eval(vertex):
    print('\n==> Evaluation mode!')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./0000000000.xyz', help='full path to input point cloud')
    parser.add_argument('--output_path', type=str, default='../log/outputs/', help='full path to input point cloud')
    parser.add_argument('--gpu_idx', type=int, default=0, help='index of gpu to use, -1 for cpu')
    parser.add_argument('--trained_model_path', type=str,
                        default=str(config.ws_path / "deepfit" / "log" / "DeepFit_no_noise" / "trained_models"),
                        help='path to trained model')
    parser.add_argument('--mode', type=str, default='DeepFit', help='how to compute normals. use: DeepFit | classic')
    parser.add_argument('--k_neighbors', type=int, default=25,
                        help='number of neighboring points for each query point')
    parser.add_argument('--jet_order', type=int, default=3,
                        help='order of jet to fit: 1-4. if in DeepFit mode, make sure to match training order')
    parser.add_argument('--compute_curvatures', type=bool, default=True,
                        help='true | false indicator to compute curvatures')
    args = parser.parse_args()

    device = torch.device("cpu" if args.gpu_idx < 0 else "cuda:%d" % 0)

    # load trained model parameters
    params = torch.load(os.path.join(args.trained_model_path, 'DeepFit_params.pth'))
    jet_order = params.jet_order
    print('Using {} order jet for surface fitting'.format(jet_order))
    model = DeepFit.DeepFit(k=1, num_points=args.k_neighbors, use_point_stn=params.use_point_stn,
                            use_feat_stn=params.use_feat_stn, point_tuple=params.point_tuple, sym_op=params.sym_op,
                            arch=params.arch, n_gaussians=params.n_gaussians, jet_order=jet_order,
                            weight_mode=params.weight_mode, use_consistency=False)
    checkpoint = torch.load(os.path.join(args.trained_model_path, 'DeepFit.pth'))
    model.load_state_dict(checkpoint)
    if not (params.points_per_patch == args.k_neighbors):
        print('Warning: You are using a different number of neighbors than trained.')
    model.to(device)

    print('- Checkpoint was loaded successfully.')

    normal_img, eval_point_counter, total_time = evaluate_epoch(model, vertex, device, params)

    return normal_img, eval_point_counter, total_time


############ EVALUATION FUNCTION ############
def evaluate_epoch(model, vertex, device, params):
    # print('\n==> Evaluating Epoch [{}]'.format(epoch))
    model.eval()  # Swith to evaluate mode
    eval_point_counter = 0
    total_time = 0.0
    k = 2
    points_per_patch = 25
    vertex = vertex.reshape(-1, 3)
    mask = vertex.sum(axis=1) == 0

    bbdiag = float(np.linalg.norm(vertex.max(0) - vertex.min(0), 2))
    vertex[~mask] = (vertex[~mask] - vertex.mean(0)) / (0.5 * bbdiag)  # shrink shape to unit sphere
    kdtree = spatial.KDTree(vertex, 10)
    normal_img = np.zeros(shape=vertex.shape)

    num, c = vertex.shape
    for i in range(num):
        if not mask[i]:
            eval_point_counter += 1
            with torch.no_grad():
                # start counting time
                start = time.time()

                point_distances, patch_point_inds = kdtree.query(vertex[i], k=25)
                rad = max(point_distances)
                patch_points = torch.from_numpy(vertex[patch_point_inds, :])

                # center the points around the query point and scale patch to unit sphere
                patch_points = patch_points - torch.from_numpy(vertex[i])
                patch_points = patch_points / rad

                patch_points, data_trans = pca_points(patch_points)
                data = torch.transpose(patch_points, 0, 1)

                points = data.to(device)
                data_trans = data_trans.to(device)

                # forward pass
                torch.cuda.synchronize()
                points = points.unsqueeze(0)
                data_trans = data_trans.unsqueeze(0)
                n_est, beta, weights, trans, trans2, neighbors_n_est = model.forward(points)
                n_est = torch.bmm(n_est.unsqueeze(1), data_trans.transpose(2, 1)).squeeze(dim=1)



                if params.use_point_stn:
                    # transform predictions with inverse transform
                    # since we know the transform to be a rotation (QSTN), the transpose is the inverse
                    n_est[:, :] = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)

                if params.use_pca:
                    # transform predictions with inverse pca rotation (back to world space)
                    n_est[:, :] = torch.bmm(n_est.unsqueeze(1), trans.transpose(2, 1)).squeeze(dim=1)
                n_est = n_est.detach().cpu()
                n_est = mu.normal_point2view_point(n_est, vertex[i], np.array([0, 0, 0]))
                # store the predicted normal
                normal_deepfit = n_est.to('cpu').numpy().reshape(3)
                normal_img[i] = mu.normal2RGB_single(normal_deepfit).reshape(3)

                # candidate normals are the input of the model
                gpu_time = time.time() - start

            total_time += gpu_time

    normal_img = normal_img.reshape(512, 512, 3)

    avg_time = total_time / eval_point_counter
    normal_img = normal_img.astype(np.uint8)
    return normal_img, eval_point_counter, total_time


def pca_points(patch_points):
    '''

    Args:
        patch_points: xyz points

    Returns:
        patch_points: xyz points after aligning using pca
    '''
    # compute pca of points in the patch:
    # center the patch around the mean:
    pts_mean = patch_points.mean(0)
    patch_points = patch_points - pts_mean

    trans, _, _ = torch.svd(torch.t(patch_points))
    patch_points = torch.mm(patch_points, trans)

    cp_new = -pts_mean  # since the patch was originally centered, the original cp was at (0,0,0)
    cp_new = torch.matmul(cp_new, trans)

    # re-center on original center point
    patch_points = patch_points - cp_new
    return patch_points, trans


if __name__ == '__main__':
    pass
