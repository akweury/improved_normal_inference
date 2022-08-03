# Improved Normal Inference

### Command for remote machine

serv-3305.kl.dfki.de (kiew)
serv-3306.kl.dfki.de (koeln)

#### copy dataset from local to remote

```
scp D:\TUK\improved_normal_inference\dataset\data_synthetic\synthetic512-5000.zip serv-3305.kl.dfki.de:/netscratch/sha/data_synthetic/synthetic512
```

##### copy model to remote

#### Create dataset

```
CUDA_VISIBLE_DEVICES=2 python3 create_dataset.py --data synthetic512 --machine remote --max_k 0 --clear false

srun \
  --ntasks=1 \
  --cpus-per-task=10 \
  --mem=42G \
  --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh \
  --container-workdir="`pwd`" \
  --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
  python3 create_dataset_multi_lights.py --machine remote --max_k 0 --data real
  

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
      --job-name="INI-an" \
      --time=7-00:00 \
      -p RTX3090 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an --dataset synthetic128 --batch_size 8 --lightNumUse 1 --print-freq 1 --lr-scheduler 8,1000


    
    srun \
      --job-name="INI-an2-f1" \
      --time=3-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an2 --dataset synthetic128 --batch_size 8 --lightNumUse 1 --lr-scheduler 8,1000 --print-freq 1 --net_type gnet-f1f

    srun \
      --job-name="real_refine_gcnn" \
      --time=3-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an_real --dataset synthetic512 --batch_size 2 --lightNumUse 1 --lr-scheduler 8,1000 --print-freq 1 --net_type gnet-f4 --lr 0.001 --refine-net gcnn




    srun \
      --job-name="INI-cnn" \
      --time=7-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp nnnn --dataset synthetic128 --batch_size 8 --lr-scheduler 8,1000 --print-freq 1 \
      --net_type cnn


    srun \
      --job-name="INI-light-gcnn" \
      --time=7-00:00 \
      -p batch \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=30G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp light --dataset synthetic128 --batch_size 8 --lr-scheduler 8,1000 --resume /home/sha/improved_normal_inference/workspace/light/output_2022-07-27_09_37_37/checkpoint-1099.pth.tar
```

#### evaluate the test dataset (no visualisation)

```
CUDA_VISIBLE_DEVICES=0 python3 eval_single_obj.py --machine pc-2103 --data real  --data_type normal
CUDA_VISIBLE_DEVICES=0 python3 eval_visual.py --machine pc-2103 --datasize synthetic512 --data synthetic_noise_pc2103 
CUDA_VISIBLE_DEVICES=0 python3 eval_visual.py --machine pc-2103 --data real_pc2103 --data_type normal

    srun \
      --job-name="INI-eval" \
      --time=7-00:00 \
      -p batch \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=30G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    python3 eval_single_obj.py  --machine remote --data real --data_type normal
    
    
        srun \
      --job-name="INI-eval" \
      --time=7-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=24G \
      --cpus-per-gpu=2 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    python3 eval_single_obj.py  --machine remote --data synthetic --data_type normal_noise
    
    
        srun \
      --job-name="INI-eval" \
      --time=7-00:00 \
      -p batch \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=30G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
    python3 eval_visual.py  --machine remote --datasize synthetic512 --data synthetic_noise_dfki
    
    
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/resng/output.zip D:\TUK\improved_normal_inference\workspace\resng
```
