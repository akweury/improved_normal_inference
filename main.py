import config
from workspace import model
from workspace.pncnn import train, test
from workspace.pncnn import network
from preprocessing.data_preprocess import noisy_a_folder
from pncnn.utils import args_parser

args = args_parser.args_parser()

if args.noise:
    for folder in ["selval", "test", "train"]:
        original_folder = config.synthetic_data / folder
        noisy_folder = config.synthetic_data_noise / folder
        noisy_a_folder(original_folder, noisy_folder)

if args.mode == "train":
    model_param = model.init_env(args, network)
    train.main(model_param)
elif args.mode == 'test':
    model_param = model.init_env(args, network)
    test.main(model_param)

