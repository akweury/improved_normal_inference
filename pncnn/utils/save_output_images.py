import numpy as np
import matplotlib.pyplot as plt
import os
import cv2 as cv
import torch
from PIL import Image
from improved_normal_inference.help_funs import file_io

cmap = plt.cm.viridis


def create_out_image_saver(exp_dir, args, epoch):
    if args.dataset == 'kitti_depth' or args.dataset == 'vkitti':
        return KittiOutputImageSaver(exp_dir, args, epoch)
    elif args.dataset == 'nyudepthv2':
        return NyuOutputImageSaver(exp_dir, args, epoch)
    elif args.dataset == 'synthetic':
        return SyntheticOutputImageSaver(exp_dir, args, epoch)


class OutputImageSaver(object):
    def __init__(self, exp_dir, args, epoch):
        self.exp_dir = exp_dir
        self.args = args
        self.epoch = epoch


############ Synthetic SAVER ############
class SyntheticOutputImageSaver(OutputImageSaver):
    def __init__(self, exp_dir, args, epoch):
        super(SyntheticOutputImageSaver, self).__init__(exp_dir, args, epoch)

    # def save_torch2rbgimg(self, torch_array, path):
    #     img = torch.permute(torch_array, (2, 3, 1, 0)).sum(dim=-1).numpy().astype(np.uint32)
    #     img = file_io.normalize_rgb_image(img)
    #     file_io.write_np2rgbimg(img, path)

    # TODO: Evaluate on different sets or different dataset vkitti
    def update(self, i, out_img, input, pred, target):

        d_out = pred[:, :1, :, :] * self.args.norm_factor

        if pred.shape[1] > 1:
            cout = pred[:, 1:2, :, :]
            cin = pred[:, 2:, :, :]
            # cout = pred[:, 3:6, :, :]
            # cin = pred[:, 6:, :, :]
        else:
            cout = None
            cin = None
        outimg_path = self.exp_dir / "output"
        if not os.path.exists(outimg_path):
            os.mkdir(outimg_path)
        d_out_folder = outimg_path / f'epoch_{str(self.epoch)}_{self.args.dataset}_{self.args.val_ds}_output'
        if not os.path.exists(d_out_folder):
            os.mkdir(d_out_folder)
        d_out_path = d_out_folder / f'{str(i * d_out.shape[0] + i).zfill(5)}.png'

        # self.save_torch2rbgimg(d_out, d_out_path)
        d_min, d_max = torch.min(d_out).item(), torch.max(d_out).item()
        file_io.save_scaled16bitImage(d_out, d_out_path, d_min, d_max)

        if cout is not None:
            cout_folder = outimg_path / f'epoch_{str(self.epoch)}_{self.args.dataset}_{self.args.val_ds}_cout'
            if not os.path.exists(cout_folder):
                os.mkdir(cout_folder)

            cout_path = cout_folder / f'{str(i * cout.shape[0] + i).zfill(5)}.png'
            cout_min, cout_max = torch.min(cout).item(), torch.max(cout).item()
            file_io.save_scaled16bitImage(cout, cout_path, cout_min, cout_max)
            # self.save_torch2rbgimg(cout, cout_path)

        if cin is not None:
            cin_folder = outimg_path / f'epoch_{str(self.epoch)}_{self.args.dataset}_{self.args.val_ds}_cin'
            if not os.path.exists(cin_folder):
                os.mkdir(cin_folder)
            cin_path = cin_folder / f'{str(i * cin.shape[0] + i).zfill(5)}.png'
            cin_min, cin_max = torch.min(cin).item(), torch.max(cin).item()

            # self.save_torch2rbgimg(cin, cin_path)
            file_io.save_scaled16bitImage(cin, cin_path, cin_min, cin_max)

        return pred


############ KITTI SAVER ############
class KittiOutputImageSaver(OutputImageSaver):

    def __init__(self, exp_dir, args, epoch):
        super(KittiOutputImageSaver, self).__init__(exp_dir, args, epoch)

    # TODO: Evaluate on different sets or different dataset vkitti
    def update(self, i, out_img, input, pred, target):

        d_out = pred[:, :1, :, :] * self.args.norm_factor

        if pred.shape[1] > 1:
            cout = pred[:, 1:2, :, :]
            cin = pred[:, 2:3, :, :]
        else:
            cout = None
            cin = None

        save_tensor_to_images(colored_depthmap_tensor(d_out), i,
                              os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                           self.args.dataset + '_' + self.args.val_ds + '_output'))
        if cout is not None:
            save_tensor_to_images(colored_depthmap_tensor(cout), i,
                                  os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                               self.args.dataset + '_' + self.args.val_ds + '_cout'))

        if cin is not None:
            save_tensor_to_images(colored_depthmap_tensor(cin), i,
                                  os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                               self.args.dataset + '_' + self.args.val_ds + '_cin'))
        return pred


############ NYU SAVER ############
class NyuOutputImageSaver(OutputImageSaver):

    def __init__(self, exp_dir, args, epoch):
        super(NyuOutputImageSaver, self).__init__(exp_dir, args, epoch)

    def update(self, i, out_image, inpt, pred, target):
        # save 8 images for visualization

        skip = 50
        if self.args.modality == 'd':
            if pred.shape[1] > 1:
                d_out = pred[:, :1, :, :]
                cout = pred[:, 1:2, :, :]
                cin = pred[:, 2:, :, :]
            else:
                d_out = pred
                cout = None
                cin = None

            save_tensor_to_images(target * 65535 / 10, i,
                                  os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                               self.args.dataset + '_' + self.args.val_ds + '_target'))

            save_tensor_to_images(d_out * 65535 / 10, i,
                                  os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                               self.args.dataset + '_' + self.args.val_ds + '_output'))
            if cout is not None:
                # cout[:,:,0:10,:] = 0 # Remove this weird artifact at the top
                save_tensor_to_images(cout * 65535, i,
                                      os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                                   self.args.dataset + '_' + self.args.val_ds + '_cout'))

            if cin is not None:
                cin[:, :, 0:10, :] = 0  # Remove this weird artifact at the top
                save_tensor_to_images(cin * 65535, i, os.path.join(self.exp_dir, 'epoch_' + str(self.epoch) + '_' +
                                                                   self.args.dataset + '_' + self.args.val_ds + '_cin'))

            out_image = None

        else:
            if self.args.modality == 'rgb':
                rgb = inpt
            elif self.args.modality == 'rgbd':
                rgb = inpt[:, :3, :, :]
                depth = inpt[:, 3:, :, :]

            if i == 0:
                if self.args.modality == 'rgbd':
                    out_image = merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    out_image = merge_into_row(rgb, target, pred)
            elif (i < 8 * skip) and (i % skip == 0):
                if self.args.modality == 'rgbd':
                    row = merge_into_row_with_gt(rgb, depth, target, pred)
                else:
                    row = merge_into_row(rgb, target, pred)
                out_image = add_row(out_image, row)
            elif i == 8 * skip:
                filename = os.path.join(self.exp_dir, 'comparison_' + str(self.epoch) + '.png')
                print('Image saved')
                save_image(out_image, filename)

        return out_image


############ HELPER FUNCTIONS ############

def colored_depthmap_tensor(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = torch.min(depth).item()
    if d_max is None:
        d_max = torch.max(depth).item()
    depth_relative = (depth - d_min) / (d_max - d_min)


    if depth_relative.size(1) == 1:
        depth_color = torch.from_numpy(cmap(depth_relative.cpu().numpy())).squeeze(1).permute((0, 3, 1, 2))[:, :3, :, :]
    else:
        depth_color = depth_relative.permute((0,1,2, 3))
    return depth_color  # N, 3, H, W


def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    return 255 * cmap(depth_relative)[:, :, :3]  # H, W, C


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge


def merge_into_row_with_gt(input, depth_input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    depth_input_cpu = np.squeeze(depth_input.cpu().numpy())
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_input_cpu), np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_input_cpu), np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_input_col = colored_depthmap(depth_input_cpu, d_min, d_max)
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)

    img_merge = np.hstack([rgb, depth_input_col, depth_target_col, depth_pred_col])

    return img_merge


def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = Image.fromarray(img_merge.astype('uint8'))
    img_merge.save(filename)


def save_tensor_to_images(t, idx, save_to_path):
    if os.path.exists(save_to_path) == False:
        os.mkdir(save_to_path)
    batch = t.shape[0]
    for i in range(batch):
        im = t[i, :, :, :].detach().data.cpu().numpy()
        im = np.transpose(im, (1, 2, 0))
        im = (im * 256).astype(np.int16)

        path = os.path.join(save_to_path, str(idx * batch + i).zfill(5) + '.png')
        # print('==> Writing {}'.format(path))
        cv.imwrite(path, im,
                   [cv.IMWRITE_PNG_COMPRESSION, 4])
