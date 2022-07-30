import numpy as np
import torch

import config
from help_funs import chart

models = {
    # "GCNN-1534": config.ws_path / "nnnn" / "trained_model" / "128" / "checkpoint-c32.pth.tar",
    # "vil10-100-1000": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint-l10-100-1000.pth.tar",
    # "vil3-100-1000": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint-l3-100-1000.pth.tar",
    # "vil-3-12-1000": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint-l1-3-12-1000.pth.tar",
    # "vil-10-1000": config.ws_path / "vil10" / "trained_model" / "128" / "checkpoint-l1-10-1000.pth.tar",

    "GCNN": config.paper_exp / "gcnn" / "checkpoint-gcnn-1099.pth.tar",  # GCNN
    "NOC": config.paper_exp / "gcnn" / "checkpoint-noc-807.pth.tar",
    "CNN": config.paper_exp / "gcnn" / "checkpoint-cnn-695.pth.tar",

}

output_folder = config.paper_exp

for model_idx, (name, model) in enumerate(models.items()):
    checkpoint = torch.load(model)
    args = checkpoint['args']
    start_epoch = checkpoint['epoch']
    batch_size = args.batch_size
    loss = checkpoint['losses'] / batch_size
    normal_loss_avg = np.sum(loss[:3], axis=0, keepdims=True) / 3
    for idx in range(normal_loss_avg.shape[1]):
        if normal_loss_avg[0, idx] == 0:
            normal_loss_avg[0, idx] = normal_loss_avg[0, idx - 1]
    chart.line_chart(np.array(normal_loss_avg), output_folder, labels=[name], title="Loss Comparison",
                     y_label="L2 Loss", log_y=False)
