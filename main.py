import config
from common.data_preprocess import noisy_a_folder
from common.SyntheticDepthDataset import SyntheticDepthDataset
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
    # config experiments
    path = config.ws_path / args.exp
    train_dataset = SyntheticDepthDataset(config.synthetic_data_noise, setname='train')
    model = nnn.CNN()

    # start the training
    train.main(args, path, model, train_dataset)


if __name__ == '__main__':
    main()
