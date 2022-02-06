import os
from os import path
import cv2 as cv
import numpy as np
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

k = 6

file_path = "./SyntheticDataSet/CapturedData/"
file_idx = 0
file_name = file_path + "0000" + str(file_idx) + ".pointcloud0.ply"
while path.exists(file_name):
    ply_data = PlyData.read(file_name)
    vertex = np.c_[ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']]
    dist, ind = KDTree(vertex).query(vertex, k)
    vertexs_neighbors = vertex[ind]
    normals = np.zeros(shape=vertex.shape)
    for i in range(vertexs_neighbors.shape[0]):
        svd = np.linalg.svd(vertexs_neighbors[i] - np.mean(vertexs_neighbors[i], axis=1, keepdims=True))
        normals[i] = svd[-1][:, -1] * 0.5 + 0.5
        normals[i][1] = 1 - normals[i][1]
        normals[i] = (normals[i] * 255).astype(np.int32)

    normal_image = cv.imread(file_path + "0000" + str(file_idx) + ".normal0.png")
    normal_idx = 0
    for i in range(normal_image.shape[0]):
        for j in range(normal_image.shape[1]):
            if np.sum(normal_image[i][j]) != 0:
                normal_image[i][j] = normals[normal_idx]
                normal_idx += 1

    cv.imwrite(file_path + "0000" + str(file_idx) + ".normal_knn_0.png", normal_image)
    file_idx += 1
    file_name = file_path + "0000" + str(file_idx) + ".pointcloud0.ply"
