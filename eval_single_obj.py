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
    parser.add_argument('--data', type=str, default='synthetic128', help="choose evaluate dataset")
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--data-type', type=str, default="normal_noise", help="choose data type")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    return args


def main(models, test_folder, args, name):
    eval_time = datetime.datetime.now().strftime("%H_%M_%S")
    eval_date = datetime.datetime.today().date()
    folder_path = config.ws_path / "eval_output" / f"{name}_{eval_date}_{eval_time}"
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
    print(f"{name}: {str(loss_avg.values())}")
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

    # test_folder = config.synthetic_data_noise_local / "synthetic128"
    for folder_name in ["baoshanlu", "bus", "dragon", "garfield", "washington"]:
        test_folder = config.synthetic_data_noise_local / "synthetic128" / "seperate" / folder_name
        models = {
            # test
            # "light-gcnn": config.paper_exp / "light" / "checkpoint-640.pth.tar",
            # "light-noc": config.paper_exp / "light" / "checkpoint-noc-499.pth.tar",
            # "light-cnn": config.paper_exp / "light" / "checkpoint-cnn-599.pth.tar",

            # "SVD": None,

            # "GCNN-GCNN": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",  # GCNN
            # "GCNN-NOC": config.paper_exp / "gcnn" / "checkpoint-noc-807.pth.tar",
            # "GCNN-CNN": config.paper_exp / "gcnn" / "checkpoint-cnn-695.pth.tar",

            "an2-8-1000": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
            # "an-8-1000": config.paper_exp / "an" / "checkpoint-818.pth.tar",
        }

        main(models, test_folder, args, folder_name)
