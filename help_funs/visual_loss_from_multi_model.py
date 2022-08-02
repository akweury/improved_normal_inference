import matplotlib.pyplot as plt
import numpy as np
import torch

import config
from help_funs import chart

models = {

    # "Trip-Net-F1F": config.ws_path / "an2" / "an2_gnet-f1f_2022-08-02_14_46_50" / "checkpoint-100.pth.tar",
    # "Trip-Net-F2F": config.ws_path / "an2" / "an2_gnet-f2f_2022-08-02_14_46_53" / "checkpoint-100.pth.tar",
    # "Trip-Net-F3F": config.ws_path / "an2" / "an2_gnet-f3f_2022-08-02_14_47_28" / "checkpoint-100.pth.tar",
    # "Trip-Net": config.ws_path / "an2" / "an2_gnet-f4_2022-08-02_14_48_32" / "checkpoint-100.pth.tar",
    #
    "CNN": config.ws_path / "nnnn" / "nnnn_cnn_2022-08-02_14_28_34" / "checkpoint-208.pth.tar",
    "NOC": config.ws_path / "nnnn" / "nnnn_gcnn_noc_2022-08-02_14_29_25" / "checkpoint-200.pth.tar",
    "GCNN": config.ws_path / "nnnn" / "nnnn_gcnn_2022-08-02_14_31_00" / "checkpoint-249.pth.tar",

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
    # loss = checkpoint['losses']
    loss = checkpoint['eval_losses']
    normal_loss_avg = np.sum(loss[:3], axis=0, keepdims=True) / 3
    normal_loss_avg = (normal_loss_avg * 100 / 8) / 8

    for idx in range(normal_loss_avg.shape[1]):
        if normal_loss_avg[0, idx] == 0:
            normal_loss_avg[0, idx] = normal_loss_avg[0, idx - 1]
    # chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Training_Loss_Comparison",
    #                  y_label="Berhu Loss", log_y=True)

    chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Test_Loss_Comparison",
                     y_label="Angle Error", log_y=True)

plt.show()
