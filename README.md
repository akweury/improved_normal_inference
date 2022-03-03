## usage
ssh sha@pc-2103

CUDA_VISIBLE_DEVICES=2 python3 main.py --ws synthetic --exp pncnn --args json --cpu False --mode train

scp C:\Users\shaji\TestProject\CapturedData\data_geometrical_body.zip sha@pc-2103:/home/sha/improved_normal_inference/dataset
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/pncnn/checkpoint-99.pth.tar  C:\Users\shaji\PycharmProjects\MA\improved_normal_inference\workspace\pncnn
scp sha@pc-2103:/home/sha/improved_normal_inference/workspace/pncnn/output.zip C:\Users\shaji\PycharmProjects\MA\improved_normal_inference\workspace\pncnn

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



