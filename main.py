import config
from common.data_preprocess import noisy_a_folder, convert2training_tensor
from workspace import train
from pncnn.utils import args_parser

import workspace.nnn.network as nnn
import workspace.nnn24.network as nnn24


def main():
    args = args_parser.args_parser()
    if args.noise:
        for folder in ["selval", "test", "train"]:
            original_folder = config.synthetic_data / folder
            if args.machine == "remote":
                noisy_folder = config.synthetic_data_noise_dfki / folder
            elif args.machine == 'local':
                noisy_folder = config.synthetic_data_noise / folder
            else:
                raise ValueError
            noisy_a_folder(original_folder, noisy_folder)
            convert2training_tensor(noisy_folder, args.neighbor)

    # config experiments
    exp_path = config.ws_path / args.exp
    if args.exp == "nnn":
        model = nnn.CNN()
    elif args.exp == "nnn24":
        model = nnn24.CNN()
    else:
        raise ValueError("Unknown exp path")
    if args.machine == 'local':
        dataset_path = config.synthetic_data_noise
    elif args.machine == 'remote':
        dataset_path = config.synthetic_data_noise_dfki
    else:
        raise ValueError

    # start the training
    train.main(args, exp_path, model, dataset_path)


if __name__ == '__main__':
    main()
