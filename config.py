import os
from pathlib import Path

root = Path(os.getcwd())
real_data = root / "RealDataSet"
synthetic_data = root / "SyntheticDataSet"
synthetic_captured_data = synthetic_data / "CapturedData"

gt_output_path = synthetic_captured_data / "gt"
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
    elif file_type == "image":
        suffix = "Gray.png"
    elif file_type == "depth":
        suffix = ".png"
    elif file_type == "pointcloud":
        suffix = ".ply"
    else:
        print("File type not exist.")
        raise ValueError

    file_name = str(file_path / str(idx).zfill(5)) + f".{file_type}0" + suffix

    return file_name


def get_output_file_name(idx, file_type, method, param=0):
    if file_type == "normal":
        if method == "gt":
            file_path = gt_output_path
        elif method == "knn":
            file_path = knn_output_path
        else:
            raise ValueError
    else:
        raise ValueError

    file_name = str(file_path / str(idx).zfill(5)) + f".{file_type}_{method}_{param}.png"
    return file_name

# filename = get_file_name(200, "ormal", "synthetic")
# print(f"{filename} exists {os.path.exists(filename)}")
