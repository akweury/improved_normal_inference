import config
# from workspace.pncnn import train, test
# from workspace.pncnn import network as pncnn_net

from workspace.nnn import train as train_nnn
from workspace.nnn import test as test_nnn
from workspace.nnn import network as nnn_net

from workspace.svd import eval

from common.data_preprocess import noisy_a_folder
from pncnn.utils import args_parser
from workspace.svdn import train as svdn


# from workspace.deepfit import train


def main():
    args = args_parser.args_parser()
    if args.noise:
        for folder in ["selval", "test", "train"]:
            original_folder = config.synthetic_data / folder
            noisy_folder = config.synthetic_data_noise / folder
            noisy_a_folder(original_folder, noisy_folder)
    train_nnn.main(args, nnn_net)
    # load all the args
    # args = args_parser.args_parser()
    #

    #
    # if args.mode == "train":
    #     train.main(args, nnn_net)
    #
    #
    # elif args.mode == 'test':
    #     test.main(args, nnn_net)


if __name__ == '__main__':
    main()
