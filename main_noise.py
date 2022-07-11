import config
from help_funs.data_preprocess import convert2training_tensor_noise
from workspace import train
from pncnn.utils import args_parser

import workspace.albedoGated.network as noiseNet


def main():
    args = args_parser.args_parser()
    if args.machine == "remote":
        dataset_folder = config.real_data_dfki
    elif args.machine == 'local':
        dataset_folder = config.real_data
    else:
        raise ValueError

    if args.convert:
        convert2training_tensor_noise(dataset_folder)

    # config experiments
    exp_path = config.ws_path / args.exp
    if args.exp == "albedoGated":
        model = noiseNet.CNN()
    else:
        raise ValueError("Unknown exp path")

    # start the training
    train.main(args, exp_path, model, dataset_folder)


if __name__ == '__main__':
    main()
