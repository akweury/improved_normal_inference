import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader

import improved_normal_inference.pncnn.dataloaders as dataloaders
from improved_normal_inference import config
# Kitti_dept
from improved_normal_inference.pncnn.dataloaders.SyntheticDepthDataset import SyntheticDepthDataset
from improved_normal_inference import config


def create_dataloader(args, eval_mode=False):
    print('==> Loading dataset "{}" .. \n'.format(args.dataset))
    if args.dataset_path == 'machine':
        get_data_set_path(args)
    if args.dataset == "synthetic":
        train_loader, val_loader = create_synthetic_depth_dataloader(args, eval_mode)
    else:
        raise ValueError
    val_set = args.val_ds

    if not eval_mode:
        print('- Found {} images in "{}" folder.\n'.format(train_loader.dataset.__len__(), 'train'))

    print('- Found {} images in "{}" folder.\n'.format(val_loader.dataset.__len__(), val_set))
    print('==> Dataset "{}" was loaded successfully!'.format(args.dataset))

    return train_loader, val_loader


################### Synthetic DEPTH ###################
def create_synthetic_depth_dataloader(args, eval_mode=False):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    norm_factor = args.norm_factor
    invert_depth = args.train_disp
    ds_dir = args.dataset_path
    rgb_dir = args.raw_kitti_path
    train_on = args.train_on
    rgb2gray = args.rgb2gray
    val_set = args.val_ds
    data_aug = args.data_aug if hasattr(args, 'data_aug') else False

    if args.modality == 'rgbd':
        load_rgb = True
    else:
        load_rgb = False

    train_loader = []
    val_loader = []

    ds_dir = eval(f"config.{ds_dir}")

    if eval_mode is not True:
        ###### Training Set ######
        # trans_list = [transforms.CenterCrop((352, 1216))]
        # train_transform = transforms.Compose(trans_list)
        train_dataset = SyntheticDepthDataset(ds_dir, setname='train', invert_depth=invert_depth,
                                              load_rgb=load_rgb, rgb2gray=rgb2gray,
                                              hflip=data_aug)

        # Select the desired number of images from the training set
        if train_on != 'full':
            import random
            training_idxs = np.array(random.sample(range(0, len(train_dataset)), int(train_on)))
            train_dataset.depth = train_dataset.depth[training_idxs]
            train_dataset.gt = train_dataset.gt[training_idxs]

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size, num_workers=args.workers)

    # Validation set
    if val_set == 'selval':
        ###### Validation Set ######
        val_dataset = SyntheticDepthDataset(ds_dir, setname='selval', transform=None,
                                            invert_depth=invert_depth,
                                            load_rgb=load_rgb, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'selval':
        ###### Selected Validation set ######
        val_dataset = SyntheticDepthDataset(ds_dir, setname='selval', transform=None,
                                            invert_depth=invert_depth,
                                            load_rgb=load_rgb, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    elif val_set == 'test':
        ###### Test set ######
        val_dataset = SyntheticDepthDataset(ds_dir, setname='test', transform=None,
                                            invert_depth=invert_depth,
                                            load_rgb=load_rgb, rgb2gray=rgb2gray)

        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=args.workers)

    return train_loader, val_loader


def get_data_set_path(args):
    args.dataset_path = config.synthetic_data
