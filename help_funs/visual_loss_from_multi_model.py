import numpy as np
import torch

import config
from help_funs import chart

models = {
    # "GCNN-1534": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint-c32.pth.tar",

    # "GCNN": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",  # GCNN
    # "NOC": config.paper_exp / "gcnn" / "checkpoint-noc-807.pth.tar",
    # "CNN": config.paper_exp / "gcnn" / "checkpoint-cnn-695.pth.tar",
    # "Trip-Net": config.paper_exp / "an2" / "checkpoint-8-1000-655.pth.tar",  # Trip Net
    "Trip-Net-F1F": config.ws_path / "an2" / "an2_gnet-f1f_2022-07-30_22_33_05" / "checkpoint-199.pth.tar",  # Trip Net
    "Trip-Net-F2F": config.ws_path / "an2" / "an2_gnet-f2f_2022-07-30_22_33_53" / "checkpoint-189.pth.tar",  # Trip Net
    "Trip-Net-F3F": config.ws_path / "an2" / "an2_gnet-f3f_2022-07-30_22_34_22" / "checkpoint-168.pth.tar",  # Trip Net
    "Trip-Net": config.ws_path / "an2" / "an2_gnet-f4_2022-07-30_22_32_25" / "checkpoint-140.pth.tar",  # Trip Net

    # "Trip-Net-C": config.paper_exp / "an" / "checkpoint-818.pth.tar",

}

output_folder = config.paper_exp

for model_idx, (name, model) in enumerate(models.items()):
    checkpoint = torch.load(model)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    batch_size = args.batch_size
    loss = checkpoint['losses']
    normal_loss_avg = np.sum(loss[:3], axis=0, keepdims=True) / 3
    for idx in range(normal_loss_avg.shape[1]):
        if normal_loss_avg[0, idx] == 0:
            normal_loss_avg[0, idx] = normal_loss_avg[0, idx - 1]
    chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Loss Comparison",
                     y_label="Berhu Loss", log_y=False)
