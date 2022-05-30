import config
from help_funs.data_preprocess import noisy_a_folder, convert2training_tensor
from workspace import train
from pncnn.utils import args_parser

import workspace.nnn.network as nnn
import workspace.nnn24.network as nnn24
import workspace.nnnn.network as nnnn
import workspace.ng.network as ng
import workspace.resng.network as resng
import workspace.nconv.network as nconv
import workspace.degares.network as degares
import workspace.ag.network as ag


def main():
    args = args_parser.args_parser()

    # config experiments
    exp_path = config.ws_path / args.exp

    if args.exp == "nnn":
        model = nnn.CNN()
    elif args.exp == "nnn24":
        model = nnn24.CNN()
    elif args.exp == "nnnn":
        model = nnnn.CNN()
    elif args.exp == "ng":
        model = ng.CNN()
    elif args.exp == "resng":
        model = resng.CNN()
    elif args.exp == "nconv":
        model = nconv.CNN()
    elif args.exp == "degares":
        model = degares.CNN()
    elif args.exp == "ag":
        model = ag.CNN()
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
