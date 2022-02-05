import os
from os import path
import cv2 as cv
import numpy as np
from plyfile import PlyData,PlyElement

k = 6

ply_file_path = "./SyntheticDataSet/CapturedData/"
file_idx = 0
file_name = ply_file_path + "0000" + str(file_idx) + ".pointcloud" + str(file_idx) + ".ply"
while path.exists(file_name):
    ply_data = PlyData.read(file_name)
    vertex = ply_data['vertex']
    knn = cv.ml.KNearest_create()
    knn.train(vertex, cv.ml.ROW_SAMPLE)
    for v in vertex:
        ret, results, neighbours, dist = knn.findNearest(v,k)

    file_idx += 1
    file_name = ply_file_path + "0000" + str(file_idx) + ".pointcloud" + str(file_idx) + ".ply"
