from workspace.pncnn import train
from workspace import model
from workspace.pncnn import network
import config
from preprocessing.data_preprocess import noisy_a_folder
for folder in ["selval", "test", "train"]:
    original_folder = config.synthetic_data / folder
    noisy_folder = config.synthetic_data_noise / folder
    noisy_a_folder(original_folder, noisy_folder)

model_param = model.init_env(network)

train.main(model_param)


