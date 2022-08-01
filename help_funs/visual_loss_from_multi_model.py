import numpy as np
import torch

import config
from help_funs import chart

models = {

    # "NOC": config.ws_path / "nnnn" / "output_2022-07-30_20_39_43" / "checkpoint-459.pth.tar",
    # "CNN": config.ws_path / "nnnn" / "output_2022-07-30_20_41_10" / "checkpoint-461.pth.tar",
    # "Trip-Net-F1F": config.ws_path / "an2" / "an2_gnet-f1f_2022-07-30_22_33_05" / "checkpoint-647.pth.tar",  # Trip Net
    # "Trip-Net-F2F": config.ws_path / "an2" / "an2_gnet-f2f_2022-07-30_22_33_53" / "checkpoint-610.pth.tar",  # Trip Net
    # "Trip-Net-F3F": config.ws_path / "an2" / "an2_gnet-f3f_2022-07-30_22_34_22" / "checkpoint-299.pth.tar",  # Trip Net
    # "Trip-Net": config.ws_path / "an2" / "an2_gnet-f4_2022-07-30_22_32_25" / "checkpoint-452.pth.tar",  # Trip Net
    "An_Real": config.ws_path / "an2" / "an_real_gnet-f4_2022-08-01_10_06_35" / "checkpoint-4.pth.tar",  # Trip Net

}

output_folder = config.paper_exp

for model_idx, (name, model) in enumerate(models.items()):
    checkpoint = torch.load(model)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    batch_size = args.batch_size
    # loss = checkpoint['losses']
    loss = checkpoint['eval_losses']
    normal_loss_avg = np.sum(loss[:3], axis=0, keepdims=True) / 3
    for idx in range(normal_loss_avg.shape[1]):
        if normal_loss_avg[0, idx] == 0:
            normal_loss_avg[0, idx] = normal_loss_avg[0, idx - 1]
    # chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Training_Loss_Comparison",
    #                  y_label="Berhu Loss", log_y=False)

    chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Test_Loss_Comparison",
                     y_label="Berhu Loss", log_y=False)
