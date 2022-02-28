from workspace.pncnn import train
from workspace import model
from workspace.pncnn import network

model_param = model.init_env(network)

train.main(model_param)


