"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import argparse
import datetime
import json
import os

import numpy as np

import config
from help_funs import chart
from workspace import eval


def get_args():
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic_noise', help="choose evaluate dataset")
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--data-type', type=str, default="normal_noise", help="choose data type")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    return args


def main(models, test_folder, args):
    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    folder_path = config.ws_path / "eval_output" / f"{eval_date}_{eval_time}"
    if not os.path.exists(str(folder_path)):
        os.mkdir(str(folder_path))

    print(f"\n\n==================== Evaluation Start =============================\n"
          f"Eval Date: {eval_date}\n"
          f"Eval Time: {eval_time}\n"
          f"Eval Models: {models.keys()}\n")

    loss_avg = {}
    time_avg = {}
    losses = []
    times = []
    sizes = []
    # evaluate CNN models one by one
    for model_idx, (name, model) in enumerate(models.items()):
        # start the evaluation
        loss_list, time_list, size_list = eval.eval(test_folder, name, model, gpu=args.gpu, data_type=args.data_type)

        loss_avg[name] = np.array(loss_list).sum() / np.array(loss_list).shape[0]
        time_avg[name] = np.array(time_list)[5:].sum() / np.array(time_list)[5:].shape[0]
        losses.append(loss_list)
        times.append(time_list[5:])
        sizes.append(size_list)

    chart.line_chart(np.array(losses), folder_path, list(models.keys()), title="Loss Evaluation", y_label="Angles",
                     cla_leg=True)
    chart.line_chart(np.array(times), folder_path, title="Time Evaluation", y_label="Milliseconds",
                     labels=list(models.keys()), cla_leg=True)
    chart.scatter_chart(np.array(sizes), np.array(losses), folder_path, "Loss Evaluation", y_label="Angles",
                        x_label="Point Number", labels=list(models.keys()))
    # save the evaluation
    saved_json = {"sequences": list(loss_avg.keys()),
                  'loss_avg': list(loss_avg.values()),
                  'time_avg': list(time_avg.values()),
                  # 'losses': losses,
                  # 'sizes': sizes,
                  # 'times': times,
                  'test_folder': str(test_folder),
                  }

    with open(str(folder_path / "evaluation.txt"), 'w') as f:
        f.write(json.dumps(saved_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    # Mode selection
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--data-type', type=str, default="normal_noise", help="choose data type")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    models = {
        # "SVD": None,
        "an3-815": config.ws_path / "an3" / "trained_model" / "128" / "checkpoint.pth.tar",
        "vil10-294": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint.pth.tar",
        "vil10-l1": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint-l1.pth.tar",
        "GCNN-c32": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint-c32.pth.tar",
        "GCNN-8": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint-c8.pth.tar",
        # "an2-666": config.ws_path / "an2" / "trained_model" / "128" / "checkpoint.pth.tar",

        # "fugrc": config.ws_path / "fugrc" / "trained_model" / "128" / "checkpoint-608.pth.tar",
        # "fugrc2": config.ws_path / "fugrc" / "trained_model" / "128" / "checkpoint-1058.pth.tar",
        # loss 13, on training
        # "SVD": None,
        # "NNNN": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint.pth.tar",  # fast, loss 14
        # "HFM": config.ws_path / "hfm" / "trained_model" / "128" / "checkpoint-288.pth.tar",  # image guided
        # "AG": config.ws_path / "ag" / "trained_model" / "512" / "checkpoint.pth.tar",
        # "NG": config.ws_path / "ng" / "trained_model" / "128" / "checkpoint.pth.tar",
        # "an2-984": config.ws_path / "an2" / "trained_model" / "512" / "checkpoint-984.pth.tar",
        # "GCNN": config.ws_path / "resng" / "trained_model" / "512" / "checkpoint-3-32.pth.tar",
        # "light1": config.ws_path / "light" / "trained_model" / "512" / "checkpoint-1.pth.tar",
        # "light2": config.ws_path / "light" / "trained_model" / "512" / "checkpoint-2.pth.tar",
        # "light": config.ws_path / "light" / "trained_model" / "512" / "checkpoint.pth.tar",
    }
    test_folder = config.synthetic_data_noise_local / "synthetic128"

    main(models, test_folder, args)
