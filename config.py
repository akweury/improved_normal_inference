import os
from pathlib import Path

xout_channel = 3
cout_in_channel = 3
cout_out_channel = 6
cin_channel = 6

root = Path(__file__).parents[0]
dataset = root / "dataset"
ws_path = root / "workspace"
output_path = root / "output"
pncnn_path = root / "pncnn"

# basic dataset path
geo_data = dataset / "data_geometrical_body"

# synthetic dataset path
synthetic_data = dataset / "data_synthetic"
synthetic_data_noise = dataset / "data_synthetic_noise"
synthetic_data_noise_dfki = Path("/datasets/sha/data_synthetic_noise")

# real dataset path
real_data = dataset / "data_real"

# # synthetic dataset path
# synthetic_basic_data = synthetic_data / "BasicData"
# synthetic_captured_data = synthetic_data / "TrainData"
# synthetic_captured_data_test = synthetic_data / "TestData"
# synthetic_captured_data_noise = synthetic_data / "DataNoise"

# kitti dataset path
gt_path_kitti = dataset / "data_depth_annotated" / "train" / "2011_09_26_drive_0001_sync"
depth_path_kitti = dataset / "data_depth_velodyne" / "train" / "2011_09_26_drive_0001_sync"
val_sel_cropped_path = dataset / "data_depth_selection" / "depth_selection" / "val_selection_cropped"
test_depth_path = dataset / "data_depth_selection" / "depth_selection" / "test_depth_completion_anonymous"

# pncnn workspace path
exper_kitti = pncnn_path / "workspace" / "kitti"
