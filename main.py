import config
# from workspace.pncnn import train, test
# from workspace.pncnn import network
#
from workspace.nnn import train, test
from workspace.nnn import network

from common.data_preprocess import noisy_a_folder
from pncnn.utils import args_parser


def main():
    # load all the args
    args = args_parser.args_parser()

    if args.noise:
        for folder in ["selval", "test", "train"]:
            original_folder = config.synthetic_data / folder
            noisy_folder = config.synthetic_data_noise / folder
            noisy_a_folder(original_folder, noisy_folder)

    if args.mode == "train":
        train.main(args, network)


    elif args.mode == 'test':
        test.main(args, network)


if __name__ == '__main__':
    main()
