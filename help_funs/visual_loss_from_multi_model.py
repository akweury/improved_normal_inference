import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from help_funs import chart

models = {

    # "Trip-Net-F1F": config.ws_path / "an2" / "an2_gnet-f1f_2022-08-03_10_01_05" / "checkpoint-450.pth.tar",
    # "Trip-Net-F2F": config.ws_path / "an2" / "an2_gnet-f2f_2022-08-03_10_04_02" / "checkpoint-550.pth.tar",
    # "Trip-Net-F3F": config.ws_path / "an2" / "an2_gnet-f3f_2022-08-02_23_59_25" / "checkpoint-650.pth.tar",
    # "Trip-Net-F3F-2": config.ws_path / "an2" / "an2_gnet-f3f_2022-08-01_22_32_35" / "checkpoint-1327.pth.tar",
    # "Trip-Net": config.ws_path / "an2" / "an2_gnet-f4_2022-08-02_23_58_31" / "checkpoint-300.pth.tar",
    # "Trip-Net-2": config.ws_path / "an2" / "an2_gnet-f4_2022-08-01_22_31_37" / "checkpoint-1116.pth.tar",
    #
    "GCNN-Huber": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-10_08_57_59" / "checkpoint-767.pth.tar",

    # "CNN": config.ws_path / "nnnn" / "nnnn_cnn_2022-08-03_09_59_25" / "checkpoint-900.pth.tar",
    # "NOC": config.ws_path / "nnnn" / "nnnn_gcnn_noc_2022-08-03_10_00_37" / "checkpoint-894.pth.tar",
    "GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-03_10_04_18" / "checkpoint-850.pth.tar",

}

output_folder = config.paper_exp
plt.rcParams.update({'font.size': 30})
plt.figure(figsize=(30, 10))
plt.grid(True)
for model_idx, (name, model) in enumerate(models.items()):
    checkpoint = torch.load(model)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    batch_size = args.batch_size

    loss = checkpoint['losses']
    normal_loss_avg = np.sum(loss[:3], axis=0, keepdims=True) / 3

    # normal_loss_avg = checkpoint['eval_losses']

    # for idx in range(normal_loss_avg.shape[1]):
    #     if normal_loss_avg[0, idx] == 0:
    #         normal_loss_avg[0, idx] = normal_loss_avg[0, idx - 1]
    chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Training_Loss_Comparison",
                     y_label="Berhu Loss", log_y=True)
    #
    # chart.line_chart(np.array(normal_loss_avg[:, 1:]), output_folder, labels=[name], title="Test_Loss_Comparison",
    #                  y_label="Angle Error", log_y=False)

plt.show()
