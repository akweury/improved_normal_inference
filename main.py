import config
from common.data_preprocess import noisy_a_folder, convert2training_tensor
from workspace import train
from pncnn.utils import args_parser

import workspace.nnn.network as nnn


def main():
    args = args_parser.args_parser()
    if args.noise:
        for folder in ["selval", "test", "train"]:
            original_folder = config.synthetic_data / folder
            noisy_folder = config.synthetic_data_noise / folder
            noisy_a_folder(original_folder, noisy_folder)
            convert2training_tensor(noisy_folder)

    # config experiments
    exp_path = config.ws_path / args.exp
    model = nnn.CNN()
    dataset_path = config.synthetic_data_noise

    # start the training
    train.main(args, exp_path, model, dataset_path)


if __name__ == '__main__':
    main()
