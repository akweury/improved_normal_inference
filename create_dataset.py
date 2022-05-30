import argparse
import os
import shutil

import config
from help_funs.data_preprocess import noisy_a_folder, convert2training_tensor

parser = argparse.ArgumentParser(description='Eval')

# Machine selection
parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                    help="loading dataset from local or dfki machine")
parser.add_argument('--data', type=str, default="synthetic", choices=['synthetic', 'real'],
                    help="choose dataset")
parser.add_argument('--max_k', type=str, default="0,1",
                    help="loading dataset from local or dfki machine")
parser.add_argument('--clear', type=str, default="false",
                    help="flag that is used to clear old dataset")

args = parser.parse_args()

if args.data == "synthetic":
    for folder in ["train", "test"]:
        original_folder = config.synthetic_data / folder
        if args.machine == "remote":
            dataset_folder = config.synthetic_data_noise_dfki / folder
        elif args.machine == 'local':
            dataset_folder = config.synthetic_data_noise / folder
        else:
            raise ValueError
        noisy_a_folder(original_folder, dataset_folder)
        if args.clear == "true":
            shutil.rmtree(str(dataset_folder / "tensor"))
        for k in args.max_k.split(','):
            print(f"K = {k}, {dataset_folder}")
            convert2training_tensor(dataset_folder, k=int(k), output_type="normal_noise")

elif args.data == "real":
    dataset_folder = config.real_data
    for k in args.max_k.split(','):
        print(f"K = {k}, {dataset_folder}")
        convert2training_tensor(dataset_folder, k=int(k), output_type="normal_noise")
else:
    raise ValueError
