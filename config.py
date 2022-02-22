import os
from pathlib import Path

root = Path(os.getcwd())

dataset = root / "dataset"
ws_path = root / "workspace"
output_path = root / "output"
pncnn_path = root / "pncnn"
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"


# basic dataset path
data_3 = synthetic_data / "BasicData" / "3"
data_3_noise = synthetic_data / "BasicData" / "3_noise"

# synthetic dataset path
synthetic_basic_data = synthetic_data / "BasicData"
synthetic_captured_data = synthetic_data / "TrainData"
synthetic_captured_data_test = synthetic_data / "TestData"
synthetic_captured_data_noise = synthetic_data / "DataNoise"

# kitti dataset path
gt_path_kitti = dataset / "data_depth_annotated" / "train" / "2011_09_26_drive_0001_sync"
depth_path_kitti = dataset / "data_depth_velodyne" / "train" / "2011_09_26_drive_0001_sync"
val_sel_cropped_path = dataset / "data_depth_selection" / "depth_selection" / "val_selection_cropped"
test_depth_path = dataset / "data_depth_selection" / "depth_selection" / "test_depth_completion_anonymous"

# pncnn workspace path
exper_kitti = pncnn_path / "workspace" / "kitti"
