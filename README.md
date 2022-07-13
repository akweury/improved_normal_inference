# Improved Normal Inference

### Command for remote machine
serv-3305.kl.dfki.de (kiew)
serv-3306.kl.dfki.de (koeln)
#### copy dataset from local to remote

```
scp D:\TUK\improved_normal_inference\dataset\data_synthetic\synthetic512-5000.zip serv-3305.kl.dfki.de:/netscratch/sha/data_synthetic/synthetic512
```

#### Create dataset

```
CUDA_VISIBLE_DEVICES=2 python3 create_dataset.py --data synthetic512 --machine remote --max_k 0 --clear true

srun \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python3 create_dataset.py --data synthetic512 --machine remote --max_k 0 --clear true
  

```

#### resize dataset

```
CUDA_VISIBLE_DEVICES=0 python3 resize_dataset.py --machine remote --new-size 64
```

#### resume a training work

```
CUDA_VISIBLE_DEVICES=2 python3 main.py --machine remote --exp ag --dataset synthetic512 --print-freq 100 --batch_size 4 --train-on 1000 --resume /home/sha/improved_normal_inference/workspace/nnnn/trained_model/checkpoint.pth.tar
```

#### start a new training work

```
CUDA_VISIBLE_DEVICES=2 python3 main.py --machine remote --exp albedoGated --dataset synthetic512 --batch_size 8 --train-on 50 

srun \
  -p RTX3090
  --ntasks=1 --gpus-per-task=1 --cpus-per-gpu=6
  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python3 main.py --machine remote --exp albedoGated --dataset synthetic512 --batch_size 32 --train-on 1000 

```

#### evaluate the test dataset (no visualisation)

```
CUDA_VISIBLE_DEVICES=0 python3 eval_visual.py --machine remote --data synthetic_noise_dfki --datasize synthetic128
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/resng/output.zip D:\TUK\improved_normal_inference\workspace\resng
```
