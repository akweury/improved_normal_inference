
import argparse

import config
from help_funs.data_preprocess import noisy_a_folder, convert2training_tensor

parser = argparse.ArgumentParser(description='Eval')

# Machine selection
parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                    help="loading dataset from local or dfki machine")
args = parser.parse_args()

for folder in ["train", "test"]:
    original_folder = config.synthetic_data / folder
    if args.machine == "remote":
        dataset_folder = config.synthetic_data_noise_dfki / folder
    elif args.machine == 'local':
        dataset_folder = config.synthetic_data_noise / folder
    else:
        raise ValueError
    noisy_a_folder(original_folder, dataset_folder)
    for k in range(3):
        convert2training_tensor(dataset_folder, k=k, output_type="normal_noise")
