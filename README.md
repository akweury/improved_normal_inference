## Meeting

https://teams.microsoft.com/l/meetup-join/19%3ameeting_MWE1ODlkZmUtN2JmZC00YmM5LWFlODEtMDNkNzU5MTBjOWFi%40thread.v2/0?context=%7b%22Tid%22%3a%2261a9f1bd-7ea0-4068-b231-bb4a6bfcb700%22%2c%22Oid%22%3a%22a279078f-08f4-4a36-bb80-abdf1a0d89f7%22%7d

## usage

ssh sha@pc-2103

# remote command

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp nnn24 --args json --mode train --batch_size 16 --machine remote

scp D:\TUK\MasterThesisJingyuan\MasterThesisJingyuan\TestProject\CapturedData\data_synthetic\train.zip sha@pc-2103:
/home/sha/improved_normal_inference/dataset/data_synthetic

scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/nnn24/output.zip D:
\TUK\improved_normal_inference\workspace\nnn24

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



