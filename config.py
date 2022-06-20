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
paper_pic = root / "paper" / 'akweury' / "Figures"

# basic dataset path
geo_data = dataset / "data_geometrical_body"

# synthetic dataset path
synthetic_data = dataset / "data_synthetic"
synthetic_data_dfki = Path("/datasets/sha/data_synthetic")
synthetic_data_nobg = dataset / "data_synthetic_nobg"
synthetic_data_noise_local = dataset / "data_synthetic_noise"
synthetic_data_noise_dfki = Path("/datasets/sha/data_synthetic_noise")

# real dataset path
real_data = dataset / "data_real"
real_data_dfki = Path("/datasets/sha/data_real")

# pretrained model
resng_model = ws_path / "resng" / "trained_model" / "checkpoint.pth.tar"
