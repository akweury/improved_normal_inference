import os
from pathlib import Path

root = Path(os.getcwd())
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"
synthetic_captured_data = synthetic_data / "CapturedData"
synthetic_basic_data = synthetic_data / "BasicShapeData" / "3"

gt_basic_output_path = synthetic_basic_data / "gt"
gt_captured_output_path = synthetic_captured_data / "gt"

knn_output_path = synthetic_data / "knn" / "3"

# filename = get_file_name(200, "ormal", "synthetic")
# print(f"{filename} exists {os.path.exists(filename)}")
