# Improved Normal Inference

### Command for remote machine

#### copy dataset from local to remote

```
scp D:\TUK\improved_normal_inference\dataset\synthetic128.zip sha@pc-2103:/datasets/sha/data_synthetic/synthetic128
```

#### Create dataset

```
CUDA_VISIBLE_DEVICES=0 python3 create_dataset.py --data synthetic128 --machine remote --max_k 0 --clear true
```

#### resume a training work

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --machine remote --exp ag --print-freq 100 --num-channels 32 --batch_size 4 --train-on 500 --resume /home/sha/improved_normal_inference/workspace/nnnn/trained_model/checkpoint.pth.tar
```

#### start a new training work

```
CUDA_VISIBLE_DEVICES=0 python3 main.py --machine remote --exp fugrc --num-channels 4 --dataset synthetic128dfki
```

#### evaluate the test dataset (no visualisation)

```
CUDA_VISIBLE_DEVICES=1 python3 eval.py --machine remote --data synthetic --noise true --gpu 1
```

#### Copy the trained model from remote

```
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/nnnn/output.zip D:\TUK\improved_normal_inference\workspace\nnnn
```
