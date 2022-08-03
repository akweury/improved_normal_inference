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
            test_folder, name, model, gpu=args.gpu, data_type=args.data_type, setname="test")

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

    if args.machine == "local":
        test_folder = config.synthetic_data_noise_local / "synthetic128"
        models = {
            # "SVD": None,

            # "CNN": config.ws_path / "nnnn" / "output_2022-07-30_20_41_10" / "checkpoint-899.pth.tar",  # Trip Net
            # "NOC": config.ws_path / "nnnn" / "output_2022-07-30_20_39_43" / "checkpoint-899.pth.tar",  # Trip Net
            # "GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-07-31_10_39_24" / "checkpoint-407.pth.tar",  # Trip Net
            # "GCNN-l2": config.ws_path / "nnnn" / "nnnn_gcnn_2022-07-31_10_44_30" / "checkpoint-421.pth.tar",  # Trip Net
            "CNN": config.ws_path / "nnnn" / "nnnn_cnn_2022-08-03_00_15_34" / "checkpoint-200.pth.tar",
            "NOC": config.ws_path / "nnnn" / "nnnn_gcnn_noc_2022-08-03_00_07_32" / "checkpoint-200.pth.tar",
            "GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-03_00_02_37" / "checkpoint-200.pth.tar",

            # "an2-8-1000": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
            # "f1": config.ws_path / "an2" / "an2_gnet-f1f_2022-07-30_22_33_05" / "checkpoint-403.pth.tar",
            # "f2": config.ws_path / "an2" / "an2_gnet-f2f_2022-07-30_22_33_53" / "checkpoint-380.pth.tar",
            # "f3": config.ws_path / "an2" / "an2_gnet-f3f_2022-07-30_22_34_22" / "model_best.pth.tar",
            # "f4": config.ws_path / "an2" / "an2_gnet-f4_2022-07-30_22_32_25" / "checkpoint-309.pth.tar",
        }
    else:
        test_folder = config.synthetic_data_noise_dfki / "synthetic512"
        models = {
            # "SVD": None,
            "GCNN-512": config.model_dfki / "checkpoint-179.pth.tar",  # GCNN
            "Trip-Net-512": config.model_dfki / "checkpoint-42.pth.tar",  # GCNN
            "Trip-Net-512-2": config.model_dfki / "an2_gnet-f4_2022-07-31_18_24_59" / "checkpoint-11.pth.tar",  # GCNN
        }


    main(models, test_folder, args)
