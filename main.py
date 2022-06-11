import config
import workspace.ag.network as ag
import workspace.degares.network as degares
import workspace.ncnn.network as ncnn
import workspace.ng.network as ng
import workspace.sconv.network as nnn
import workspace.nnn24.network as nnn24
import workspace.nnnn.network as nnnn
import workspace.resng.network as resng
from pncnn.utils import args_parser
from workspace import train


def main():
    args = args_parser.args_parser()

    # config experiments
    exp_path = config.ws_path / args.exp

    if args.exp == "sconv":
        model = nnn.CNN(args.num_channels)
    elif args.exp == "nnnn":
        model = nnnn.CNN(args.num_channels)
    # elif args.exp == "nnn24":
    #     model = nnn24.CNN()
    elif args.exp == "ng":
        model = ng.CNN(args.num_channels)
    elif args.exp == "resng":
        model = resng.CNN()
    elif args.exp == "degares":
        model = degares.CNN()

    elif args.exp == "ncnn":
        model = ncnn.CNN(args.num_channels)
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
