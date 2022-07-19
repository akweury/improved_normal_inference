import argparse
import json
import os
from pprint import pprint

import config
from workspace import train


def paser():
    """
    Parese command line arguments

    Args:
    opt_args: Optional args for testing the function. By default sys.argv is used

    Returns:
        args: Dictionary of args.

    Raises:
        ValueError: Raises an exception if some arg values are invalid.
    """
    # Construct the parser
    parser = argparse.ArgumentParser(description='NNN')

    # Mode selection
    parser.add_argument('--machine', type=str, default="local", help='choose the training machin, local or remote')
    parser.add_argument('--num-channels', type=int, help='choose the number of channels in the model')
    parser.add_argument('--output_type', type=str, default="normal",
                        help='choose the meaning of output tensor, rgb or normal')

    parser.add_argument('--exp', '--e', help='Experiment name')

    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='Path to latest checkpoint (default: none)')

    ########### General Dataset arguments ##########

    parser.add_argument('--dataset', type=str, default='', help='Dataset Name.')
    parser.add_argument('--batch_size', '-b', default=4, type=int, help='Mini-batch size (default: 4)')

    parser.add_argument('--train-on', default='full', type=str, help='The number of images to train on from the data.')

    ########### Training arguments ###########
    parser.add_argument('--epochs', default=20, type=int,
                        help='Total number of epochs to run (default: 30)')

    parser.add_argument('--optimizer', '-o', default='adam')

    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='Initial learning rate (default 0.001)')

    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum.')

    parser.add_argument('--loss', '-l', default='l1')
    parser.add_argument('--loss-type', default='l2')
    parser.add_argument('--init-net', type=str, default=None)

    parser.add_argument('--penalty', '-pena', default=1.2, help='penalty of output value which out of range [0-255]')
    parser.add_argument('--albedo-penalty', type=float, default=1e-4,
                        help='penalty of albedo loss')
    ########### Logging ###########
    parser.add_argument('--print-freq', default=100, type=int,
                        help='Printing evaluation criterion frequency (default: 10)')
    parser.add_argument('--angle_loss', default=False, type=bool,
                        help='Calculate angle loss and plot')
    # Parse the arguments
    args = parser.parse_args()

    # Path to the workspace directory
    ws_path = config.ws_path
    args_path = ws_path / args.exp / 'args.json'
    load_args_from_file(args_path, args)
    print_args(args)

    return args


def load_args_from_file(args_file_path, given_args):
    if os.path.isfile(args_file_path):
        with open(args_file_path, 'r') as fp:
            loaded_args = json.load(fp)

        # Replace given_args with the loaded default values
        for key, value in loaded_args.items():
            if key not in ['workspace', 'exp', 'evaluate', 'resume', 'gpu']:  # Do not overwrite these keys
                setattr(given_args, key, value)

        print('\n==> Args were loaded from file "{}".'.format(args_file_path))
    else:
        print('\n==> Args file "{}" was not found!'.format(args_file_path))


def print_args(args):
    pprint(f'==> Experiment Args:  {args} ')


def get_model(args):
    if args.exp == "sconv":
        import workspace.sconv.network as nnn

        model = nnn.CNN(args.num_channels)
    elif args.exp == "nnnn":
        import workspace.nnnn.network as nnnn

        model = nnnn.CNN(args.num_channels)
    # elif args.exp == "nnn24":
    #     model = nnn24.CNN()
    elif args.exp == "ng":
        import workspace.ng.network as ng

        model = ng.CNN(args.num_channels)
    elif args.exp == "resng":
        import workspace.resng.network as resng
        model = resng.CNN(args.num_channels)
    elif args.exp == "light":
        import workspace.light.network as light
        model = light.CNN(args.num_channels)
    elif args.exp == "fugrc":
        import workspace.fugrc.network as fugrc
        model = fugrc.CNN(args.num_channels)
    elif args.exp == "ncnn":
        import workspace.ncnn.network as ncnn

        model = ncnn.CNN(args.num_channels)
    elif args.exp == "ag":
        import workspace.ag.network as ag
        model = ag.CNN(args.num_channels)
    elif args.exp == "an":
        import workspace.an.network as an
        model = an.CNN(args.num_channels)
    elif args.exp == "an2":
        import workspace.an2.network as an2
        model = an2.CNN(args.num_channels)
    elif args.exp == "vi5":
        import workspace.vi5.network as vi5
        model = vi5.CNN(args.num_channels)
    elif args.exp == "vil10":
        import workspace.vil10.network as vil10
        model = vil10.CNN(args.num_channels)
    elif args.exp == "i5":
        import workspace.i5.network as i5
        model = i5.CNN(args.num_channels)
    elif args.exp == "albedoGated":
        import workspace.albedoGated.network as albedoGated

        model = albedoGated.CNN(args.num_channels)
    elif args.exp == "hfm":
        import workspace.hfm.network as hfm
        model = hfm.CNN()
    else:
        raise ValueError("Unknown exp path")
    return model


def get_dataset_path(args):
    if args.machine == "local":
        dataset_path = config.synthetic_data_noise_local / args.dataset
    elif args.machine == "remote":
        dataset_path = config.synthetic_data_noise_dfki / args.dataset
    else:
        raise ValueError
    return dataset_path


def main():
    args = paser()
    # config experiments
    exp_path = config.ws_path / args.exp
    model = get_model(args)
    dataset_path = get_dataset_path(args)
    # start the training
    train.main(args, exp_path, model, dataset_path)


if __name__ == '__main__':
    main()
