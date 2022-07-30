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
  python3 create_dataset_multi_lights.py --machine remote --max_k 0
  

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
      --job-name="INI-an-8-1000" \
      --time=7-00:00 \
      -p RTX3090 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an --dataset synthetic128 --batch_size 8 --lightNumUse 1 --lr-scheduler 8,1000






    srun \
      --job-name="INI-an-real" \
      --time=7-00:00 \
      -p RTX3090 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an_real --dataset real --batch_size 4 --lightNumUse 1 --lr-scheduler 8,1000 








    
    srun \
      --job-name="INI-an2" \
      --time=7-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an2 --dataset synthetic512 --batch_size 8 --lightNumUse 1 --lr-scheduler 8,1000 --resume /home/sha/improved_normal_inference/workspace/an2/output_2022-07-27_21_31_59/checkpoint-303.pth.tar 


    srun \
      --job-name="INI-an3-3-12-1000" \
      --time=7-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp an3 --dataset synthetic128 --batch_size 8 --lightNumUse 1 --lr-scheduler 3,12,1000 --resume /home/sha/improved_normal_inference/workspace/an3/output_2022-07-23_08_00_18/checkpoint-407.pth.tar



    srun \
      --job-name="vil-10-8-1000" \
      --time=7-00:00 \
      -p RTXA6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp vil10 --dataset synthetic128 --lightNumUse 10 --num-channels 128 --lr-scheduler 8,1000 --resume /home/sha/improved_normal_inference/workspace/vil10/output_2022-07-23_16_18_46/checkpoint-201.pth.tar

srun \
--job-name="INI-vi5-full" \
--time=7-00:00 \
 -p RTX3090 \
 --ntasks=1 \
 --gpus-per-task=1  \
 --mem=64G  \
 --cpus-per-gpu=6 \
 --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.08-py3.sqsh  \
 --container-workdir="`pwd`" \
 --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
 python3 main.py --machine remote --exp vi5 --dataset synthetic128 --batch_size 32  --resume /home/sha/improved_normal_inference/workspace/light/output_2022-07-14_10_41_23/checkpoint-2074.pth.tar


    srun \
      --job-name="INI-nnnn-cnn-huber" \
      --time=7-00:00 \
      -p RTX6000 \
      --ntasks=1 \
      --gpus-per-task=1 \
      --mem=32G \
      --cpus-per-gpu=4 \
      --container-image=/netscratch/enroot/nvcr.io_nvidia_pytorch_21.10-py3.sqsh \
      --container-workdir="`pwd`" \
      --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds:/ds:ro,"`pwd`":"`pwd`" \
      python3 main.py --machine remote --exp nnnn --dataset synthetic128 --batch_size 8 --lr-scheduler 8,1000 --print-freq 1 --resume /home/sha/improved_normal_inference/workspace/nnnn/output_2022-07-24_12_08_51/checkpoint-199.pth.tar


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
CUDA_VISIBLE_DEVICES=0 python3 eval_visual.py --machine remote --data synthetic_noise_dfki --datasize synthetic128
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/resng/output.zip D:\TUK\improved_normal_inference\workspace\resng
```
