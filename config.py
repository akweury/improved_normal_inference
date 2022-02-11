import os
from pathlib import Path

root = Path(os.getcwd())
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"
synthetic_captured_data = synthetic_data / "TrainData"
synthetic_basic_data = synthetic_data / "BasicData"

output_path = root / "output"

# filename = get_file_name(200, "ormal", "synthetic")
# print(f"{filename} exists {os.path.exists(filename)}")
