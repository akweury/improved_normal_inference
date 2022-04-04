## usage

ssh sha@pc-2103

# remote command

CUDA_VISIBLE_DEVICES=0 python3 main.py --exp nnn --args json --mode train --batch_size 8 --machine remote

scp C:\Users\shaji\TestProject\CapturedData\data_geometrical_body.zip sha@pc-2103:
/home/sha/improved_normal_inference/dataset

scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/nnn/output.zip D:
\TUK\improved_normal_inference\workspace\nnn

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



