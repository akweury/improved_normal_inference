# Improved Normal Inference

---


### Command for remote machine
---

###### Create dataset

CUDA_VISIBLE_DEVICES=1 python3 create_dataset.py --machine remote

###### resume a training work

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp nconv --args json --mode train --batch_size 8 --resume model_name

###### start a new training work

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp resng --args json --mode train --batch_size 8

###### evaluate the test dataset (no visualisation)

CUDA_VISIBLE_DEVICES=1 python3 eval.py --machine remote --data synthetic --noise true --gpu 1

###### Copy the trained model from remote

scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/noise_net/output.zip D:
\TUK\improved_normal_inference\workspace\noise_net

