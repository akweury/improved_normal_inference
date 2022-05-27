# Improved Normal Inference

### Command for remote machine

#### Create dataset

```
CUDA_VISIBLE_DEVICES=1 python3 create_dataset.py --machine remote --max_k 2
```

#### resume a training work

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --machine remote --exp resng --print-freq 100 --batch_size 8 --train-on 500 --resume model-name
```

#### start a new training work

```
CUDA_VISIBLE_DEVICES=1 python3 main.py --machine remote --exp degares --print-freq 100 --batch_size 5 --train-on 5
```

#### evaluate the test dataset (no visualisation)

```
CUDA_VISIBLE_DEVICES=1 python3 eval.py --machine remote --data synthetic --noise true --gpu 1
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/nnnn/output.zip D:\TUK\improved_normal_inference\workspace\nnnn
```
