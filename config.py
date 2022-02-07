import os
from pathlib import Path

root = Path(os.getcwd())
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"
synthetic_captured_data = synthetic_data / "CapturedData"

knn_output_path = synthetic_data / "knn"


def get_file_name(idx, file_type, data_type):
    if data_type == "synthetic":
        file_path = synthetic_captured_data
    elif data_type == "real":
        file_path = real_data
    else:
        raise ValueError

    if file_type == "data":
        suffix = ".json"
    elif file_type in ["depth", "image", "normal"]:
        suffix = ".png"
    elif file_type == "pointcloud":
        suffix = ".ply"
    else:
        print("File type not exist.")
        raise ValueError

    file_name = str(file_path / "0000") + str(idx) + f".{file_type}0" + suffix

    return file_name


def get_output_file_name(idx, method, param):
    if method == "knn":
        file_path = knn_output_path
    else:
        raise ValueError

    file_name = str(file_path / "0000") + str(idx) + f".normal_{method}_{param}.png"
    return file_name

# filename = get_file_name(200, "ormal", "synthetic")
# print(f"{filename} exists {os.path.exists(filename)}")
