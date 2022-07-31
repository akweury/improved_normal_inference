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
    parser.add_argument('--data', type=str, default='synthetic512', help="choose evaluate dataset")
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
    median_avg = {}
    d5_avg = {}
    d11_avg = {}
    d22_avg = {}
    d30_avg = {}
    losses = []
    times = []
    sizes = []
    # evaluate CNN models one by one
    for model_idx, (name, model) in enumerate(models.items()):
        # start the evaluation
        loss_list, time_list, size_list, median_loss_list, d5_list, d11_list, d22_list, d30_list = eval.eval(
            test_folder, name, model, gpu=args.gpu, data_type=args.data_type)

        loss_avg[name] = np.array(loss_list).sum() / np.array(loss_list).shape[0]
        time_avg[name] = np.array(time_list)[5:].sum() / np.array(time_list)[5:].shape[0]
        median_avg[name] = np.array(median_loss_list).sum() / np.array(median_loss_list).shape[0]
        d5_avg[name] = np.array(d5_list).sum() / np.array(d5_list).shape[0]
        d11_avg[name] = np.array(d11_list).sum() / np.array(d11_list).shape[0]
        d22_avg[name] = np.array(d22_list).sum() / np.array(d22_list).shape[0]
        d30_avg[name] = np.array(d30_list).sum() / np.array(d30_list).shape[0]
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

    test_folder = config.synthetic_data_noise_local / "synthetic512"
    # test_folder = config.real_data
    models = {
        # test
        # "light-gcnn": config.paper_exp / "light" / "checkpoint-640.pth.tar",
        # "light-noc": config.paper_exp / "light" / "checkpoint-noc-499.pth.tar",
        # "light-cnn": config.paper_exp / "light" / "checkpoint-cnn-599.pth.tar",

        # "SVD": None,

        "GCNN-512": config.ws_path / "nnnn" / "output_2022-07-30_16_43_11" / "checkpoint-138.pth.tar",  # GCNN
        # "GCNN-512": config.ws_path / "nnnn" / "output_2022-07-30_16_43_11" / "model_best.pth.tar",  # GCNN
        "Trip-Net-512": config.ws_path / "an2" / "output_2022-07-30_16_35_38" / "checkpoint-32.pth.tar",  # GCNN

        # "GCNN-GCNN": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",  # GCNN
        # "GCNN-NOC": config.paper_exp / "gcnn" / "checkpoint-noc-807.pth.tar",
        # "GCNN-CNN": config.paper_exp / "gcnn" / "checkpoint-cnn-695.pth.tar",

        # "an2-8-1000": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
        # "an-8-1000": config.paper_exp / "an" / "checkpoint-818.pth.tar",

        # "an-818": config.paper_exp / "an" / "checkpoint-818.pth.tar",
        # "an-real": config.paper_exp / "an_real" / "checkpoint-499.pth.tar",
        # "vil10-8-1000": config.paper_exp / "vil10" / "checkpoint-10-8-1000-1199.pth.tar",

        # "an2-f3f": config.paper_exp / "an2" / "checkpoint-f3f-303.pth.tar",
        # "an2-f3b": config.paper_exp / "an2" / "checkpoint-f3b-308.pth.tar",
        # "an2-f1b-793": config.paper_exp / "an2" / "checkpoint-f1b-798.pth.tar",
        # "an2-f1b-801": config.paper_exp / "an2" / "checkpoint-f1b-801.pth.tar",

        # record
        # "vil-8-10-1000": config.paper_exp / "vil10" / "checkpoint-8-1000-1256.pth.tar",
        # "an2-8-1000": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",
        # "an3-3-12-1000": config.paper_exp / "an3" / "checkpoint-3-12-1000-899.pth.tar",
        # "an3-8-1000": config.paper_exp / "an3" / "checkpoint-8-1000-692.pth.tar",
        # "vil-10-1000": config.paper_exp / "vil10" / "checkpoint-10-1000.pth.tar",

        # "light-gcnn": config.paper_exp / "light" / "checkpoint-gcnn-1799.pth.tar",
        # "light-gcnn": config.paper_exp / "light" / "checkpoint-640.pth.tar",
        # "light-noc": config.paper_exp / "light" / "checkpoint-noc-499.pth.tar",
        # "light-noc": config.paper_exp / "light" / "checkpoint-noc-1299.pth.tar",
        # "light-cnn": config.paper_exp / "light" / "checkpoint-cnn-599.pth.tar",

        # "gcnn-cnn": config.paper_exp / "gcnn" / "checkpoint-cnn-858.pth.tar",
        # "gcnn-noc": config.paper_exp / "gcnn" / "checkpoint-noc-955.pth.tar",
        # "gcnn-gcnn": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",

        # "SVD": None,

    }

    main(models, test_folder, args)
