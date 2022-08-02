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

        loss_avg[name] = ("%.2f" % (np.array(loss_list).sum() / np.array(loss_list).shape[0]))
        time_avg[name] = ("%.2f" % (np.array(time_list)[5:].sum() / np.array(time_list)[5:].shape[0]))
        median_avg[name] = ("%.2f" % (np.array(median_loss_list).sum() / np.array(median_loss_list).shape[0]))
        d5_avg[name] = ("%.2f" % (1 - np.array(d5_list).sum() / np.array(d5_list).shape[0]))
        d11_avg[name] = ("%.2f" % (1 - np.array(d11_list).sum() / np.array(d11_list).shape[0]))
        d22_avg[name] = ("%.2f" % (1 - np.array(d22_list).sum() / np.array(d22_list).shape[0]))
        d30_avg[name] = ("%.2f" % (1 - np.array(d30_list).sum() / np.array(d30_list).shape[0]))
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
    print(f"mean: {str(loss_avg.values())}")
    print(f"median: {str(median_avg.values())}")
    print(f"d5: {str(d5_avg.values())}")
    print(f"d11: {str(d11_avg.values())}")
    print(f"d22: {str(d22_avg.values())}")
    print(f"d30: {str(d30_avg.values())}")
    with open(str(folder_path / "evaluation.txt"), 'w') as f:
        f.write(json.dumps(saved_json))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Eval')
    # Mode selection
    parser.add_argument('--gpu', type=int, default=0, help="choose GPU index")
    parser.add_argument('--data_type', type=str, default="normal_noise", help="choose data type")
    parser.add_argument('--data', type=str, default="synthetic", help="choose data type")
    parser.add_argument('--machine', type=str, default="local", choices=['local', 'remote'],
                        help="loading dataset from local or dfki machine")
    args = parser.parse_args()

    # test_folder = config.synthetic_data_noise_local / "synthetic128"
    for folder_name in ["baoshanlu", "bus", "dragon", "garfield", "washington"]:
        if args.machine == "local":
            # test_folder = config.real_data / "test"
            # test_folder = config.synthetic_data_noise_local / "synthetic128" / "seperate" / folder_name
            test_folder = config.synthetic_data_noise_local / "synthetic512" / "test"
            models = {
                "SVD": None,
                # "GCNN-512": config.ws_path / "nnnn" / "output_2022-07-30_16_43_11" / "checkpoint-226.pth.tar",
                # Trip Net

            }
        else:
            if args.data == "synthetic":
                test_folder = config.synthetic_data_noise_dfki / "synthetic512" / "test"
            else:
                test_folder = config.real_data_dfki / "test"
            models = {
                # "SVD": None,
                # "NNNN-512": config.model_dfki / "checkpoint-295-nnnn.pth.tar",  # GCNN

                "Trip-Net-512": config.model_dfki / "checkpoint-32.pth.tar",  # GCNN
                "Trip-Net-512-52": config.model_dfki / "an2_gnet-f4_2022-08-01_17_36_15" / "checkpoint-51.pth.tar",
                
                "Trip-Net-refine-371": config.model_dfki / "checkpoint-371.pth.tar",  # GCNN
            }

        main(models, test_folder, args, folder_name)
