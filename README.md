## Meeting

https://teams.microsoft.com/l/meetup-join/19%3ameeting_MWE1ODlkZmUtN2JmZC00YmM5LWFlODEtMDNkNzU5MTBjOWFi%40thread.v2/0?context=%7b%22Tid%22%3a%2261a9f1bd-7ea0-4068-b231-bb4a6bfcb700%22%2c%22Oid%22%3a%22a279078f-08f4-4a36-bb80-abdf1a0d89f7%22%7d

## usage

ssh sha@pc-2103

# remote command

## resume

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp nnn24 --args json --mode train --batch_size 8 --resume
/home/sha/improved_normal_inference/workspace/nnn24/trained_model/full_normal/checkpoint-1740.pth.tar

D:\TUK\improved_normal_inference\workspace\noise_net\output_2022-05-09_15_37_05\checkpoint-98.pth.tar
## new training work

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp noise_net --args json --mode train --batch_size 8

scp D:\TUK\improved_normal_inference\dataset\data_real.zip sha@pc-2103:/datasets/sha

scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/nnnn/output.zip D:
\TUK\improved_normal_inference\workspace\nnnn

# local command

--exp pncnn --args json --mode train --cpu True --batch_size 1 --epochs 10 --exp pncnn --args json --mode test --cpu
True --batch_size 1

# Improved Normal Inference

---
## Usage
The output folder will save normal maps with a copy of ground truth

1. run ply_generator.py to generate ply file, which is based on ground truth normal map.
2. run knn_normal.py to generate knn normal map

---
### TODO List:
- reformat the code as a decent starting point
- A proper evaluation metric
- A standard basic model dataset for very first validation of algorithms



