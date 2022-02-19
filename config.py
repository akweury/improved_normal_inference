import os
from pathlib import Path

root = Path(os.getcwd())
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"
synthetic_captured_data = synthetic_data / "TrainData"
synthetic_captured_data_noise = synthetic_data / "DataNoise"
synthetic_basic_data = synthetic_data / "BasicData"
synthetic_basic_data_noise = synthetic_data / "BasicDataNoise"

output_path = root / "output"

exper_synthetic = root / "pncnn" / "workspace" / "synthetic"
exper_kitti = root / "pncnn" / "workspace" / "kitti"

gt_path_kitti = root / "data_depth_annotated" / "train" / "2011_09_26_drive_0001_sync"
depth_path_kitti = root / "data_depth_velodyne" / "train" / "2011_09_26_drive_0001_sync"
val_sel_cropped_path = root / "data_depth_selection" / "depth_selection" / "val_selection_cropped"
test_depth_path = root / "data_depth_selection" / "depth_selection" / "test_depth_completion_anonymous"

# filename = get_file_name(200, "ormal", "synthetic")
# print(f"{filename} exists {os.path.exists(filename)}")
