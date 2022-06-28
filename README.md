# Improved Normal Inference

### Command for remote machine

#### copy dataset from local to remote

```
scp D:\TUK\improved_normal_inference\dataset\synthetic256.zip sha@pc-2103:/datasets/sha/data_synthetic/synthetic256
```

#### Create dataset

```
CUDA_VISIBLE_DEVICES=0 python3 create_dataset.py --data synthetic128 --machine remote --max_k 0 --clear true
```

#### resume a training work

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --machine remote --exp hfm --dataset synthetic128 --print-freq 100 --num-channels 32 --batch_size 4 --train-on 1000 --resume /home/sha/improved_normal_inference/workspace/nnnn/trained_model/checkpoint.pth.tar
```

#### start a new training work

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --machine remote --exp resng --dataset synthetic128 --batch_size 32 --train-on 1000 --num-channels 128
```

#### evaluate the test dataset (no visualisation)

```
CUDA_VISIBLE_DEVICES=0 python3 eval_visual.py --machine remote --data synthetic_noise_dfki --datasize synthetic128
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/resng/output.zip D:\TUK\improved_normal_inference\workspace\resng
```
