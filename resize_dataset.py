import argparse
import glob
import os
import shutil

import cv2
import numpy as np

import config

parser = argparse.ArgumentParser(description='Eval')

# Machine selection
parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                    help="loading dataset from local or dfki machine")
parser.add_argument('--new-size', type=int, default=64,
                    help="the size of resized data")
parser.add_argument('--clear', type=str, default="false",
                    help="flag that is used to clear old dataset")
args = parser.parse_args()

new_size = args.new_size

for folder in ["train", "test", "val"]:

    if args.machine == "remote":
        original_folder = config.synthetic_data_dfki / "synthetic512" / folder
        dataset_folder = config.synthetic_data_dfki / f"synthetic{new_size}" / folder
        if not os.path.exists(str(config.synthetic_data_dfki / f"synthetic{new_size}")):
            os.mkdir(str(config.synthetic_data_dfki / f"synthetic{new_size}"))

    elif args.machine == 'local':
        original_folder = config.synthetic_data / "synthetic512" / folder
        dataset_folder = config.synthetic_data / f"synthetic{new_size}" / folder
        if not os.path.exists(str(config.synthetic_data / f"synthetic{new_size}")):
            os.mkdir(str(config.synthetic_data / f"synthetic{new_size}"))
    else:
        raise ValueError

    if not os.path.exists(str(dataset_folder)):
        os.mkdir(str(dataset_folder))
    depth_files = np.array(sorted(glob.glob(str(original_folder / "*depth0.png"), recursive=True)))
    normal_files = np.array(sorted(glob.glob(str(original_folder / "*normal0.png"), recursive=True)))
    data_files = np.array(sorted(glob.glob(str(original_folder / "*data0.json"), recursive=True)))
    img_files = np.array(sorted(glob.glob(str(original_folder / "*image0.png"), recursive=True)))

    for item in range(len(data_files)):
        if os.path.exists(str(dataset_folder / (str(item).zfill(5) + f".depth0-9.png"))):
            continue

        print(f"file {item}")
        depth = cv2.imread(depth_files[item], -1)
        img = cv2.imread(img_files[item], -1)
        normal = cv2.imread(normal_files[item], -1)
        w, h = depth.shape
        # left most valid col
        saved_case = 0
        failed_counter = 0
        while saved_case < 10 and failed_counter < 20:
            left = np.random.randint(min(int(w - new_size), int(np.argmax(np.sum(depth, axis=1) > 0))) - 1,
                                     int(w - new_size))
            right = left + new_size
            # highest valid row
            top = np.random.randint(min(int(h - new_size), int(np.argmax(np.sum(depth, axis=0) > 0))) - 1,
                                    int(h - new_size))
            below = top + new_size
            if right > 512 or top > 512:
                raise ValueError
            # depth resize
            resized_depth = depth[left: right, top:below]
            if np.count_nonzero(resized_depth) < new_size * new_size * 0.3:
                failed_counter += 1
                continue
            else:
                saved_case += 1

                # depth file
                cv2.imwrite(str(dataset_folder / (str(item).zfill(5) + f".depth0-{saved_case}.png")), resized_depth)

                # data
                shutil.copyfile(data_files[item],
                                str(dataset_folder / (str(item).zfill(5) + f".data0-{saved_case}.json")))

                # image resize
                resized_img = img[left: right, top:below]
                cv2.imwrite(str(dataset_folder / (str(item).zfill(5) + f".image0-{saved_case}.png")), resized_img)

                # normal resize
                resized_normal = normal[left: right, top:below]
                cv2.imwrite(str(dataset_folder / (str(item).zfill(5) + f".normal0-{saved_case}.png")), resized_normal)
