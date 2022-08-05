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
paper_exp = root / "paper" / 'exp_result'

# basic dataset path
geo_data = dataset / "data_geometrical_body"

# synthetic dataset path
synthetic_data = dataset / "data_synthetic"
synthetic_data_dfki = Path("/netscratch/sha/data_synthetic")
synthetic_data_noise_local = dataset / "data_synthetic_noise"
synthetic_data_noise_dfki = Path("/netscratch/sha/data_synthetic_noise")
synthetic_data_noise_pc2103 = Path("/datasets/sha/data_synthetic_noise")

# real dataset path
real_data = dataset / "data_real"
real_data_dfki = Path("/netscratch/sha/data_real")
real_data_pc2103 = Path("/datasets/sha/data_real")

# model path
model_dfki = Path("/netscratch/sha/models")

# pretrained model

light_c64 = paper_exp / "light" / "checkpoint-640.pth.tar"
light_noc = paper_exp / "light" / "checkpoint-noc-499.pth.tar"
light_cnn = paper_exp / "light" / "checkpoint-cnn-599.pth.tar"

gcnn_cnn = ws_path / "nnnn" / "nnnn_cnn_2022-08-03_09_59_25" / "checkpoint-900.pth.tar"
gcnn_noc = ws_path / "nnnn" / "nnnn_gcnn_noc_2022-08-03_10_00_37" / "checkpoint-894.pth.tar"
gcnn_gcnn = ws_path / "nnnn" / "nnnn_gcnn_2022-08-03_10_04_18" / "checkpoint-850.pth.tar"

an2_trip_net = ws_path / "an2" / "output_2022-07-30_16_35_38" / "checkpoint-32.pth.tar"
an2_trip_net_remote = Path("/netscratch/sha/models/checkpoint-32.pth.tar")
gcnn_512_remote = Path("/netscratch/sha/models/checkpoint-269.pth.tar")
