"""
input: an image, the incomplete depth map of the image
output: a complete depth map
"""
import os
import argparse
import datetime
import numpy as np
import config

from workspace import eval
from help_funs import chart


def get_args():
    parser = argparse.ArgumentParser(description='Eval')

    # Mode selection
    parser.add_argument('--data', type=str, default='synthetic', help="choose evaluate dataset")
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    return args


def main():
    args = get_args()

    models = {
        # "SVD": None,
        "DeGaRes": config.ws_path / "degares" / "trained_model" / "checkpoint.pth.tar",
        "NNNN": config.ws_path / "nnnn" / "trained_model" / "checkpoint.pth.tar",
        "NG+2": config.ws_path / "resng" / "trained_model" / "checkpoint-6693.pth.tar",
        # "NG": config.ws_path / "ng" / "trained_model" / "checkpoint.pth.tar",
        "ResNG": config.ws_path / "resng" / "trained_model" / "checkpoint.pth.tar",
    }
    # load test model names
    if args.data == 'synthetic':
        dataset_path = config.synthetic_data_noise
    elif args.data == 'real':
        dataset_path = config.real_data
    else:
        raise ValueError

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
    # evaluate CNN models one by one
    for model_idx, (name, model) in enumerate(models.items()):
        # start the evaluation
        loss_list, time_list = eval.eval(dataset_path, name, model, gpu=args.gpu)

        loss_avg[name] = np.array(loss_list).sum() / np.array(loss_list).shape[0]
        time_avg[name] = np.array(time_list)[5:].sum() / np.array(time_list)[5:].shape[0]
        losses.append(loss_list)
        times.append(time_list[5:])

    chart.line_chart(np.array(losses), folder_path, "Loss Evaluation", y_label="Radius", labels=list(models.keys()),
                     cla_leg=True)
    chart.line_chart(np.array(times), folder_path, "Time Evaluation", y_label="Milliseconds",
                     labels=list(models.keys()))

    # save the evaluation
    with open(str(folder_path / "evaluation.txt"), 'w') as f:
        f.write(str(loss_avg))
        f.write(str(time_avg))


if __name__ == '__main__':
    main()
