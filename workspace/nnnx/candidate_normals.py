import random

import numpy as np
import cv2 as cv
import itertools
import math


def normal_point2view_point(normal, point, view_point):
    if np.dot(normal, (point - view_point)) > 0:
        normal = -normal
    return normal


def normal2RGB(normals):
    h, w, c = normals.shape
    for i in range(h):
        for j in range(w):
            normals[i, j] = normals[i, j] * 0.5 + 0.5
            normals[i, j, 2] = 1 - normals[i, j, 2]
            normals[i, j] = (normals[i, j] * 255)
    rgb = cv.normalize(normals, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    return rgb


def rgb2normal(color):
    color_norm = color / np.linalg.norm(color)
    color_norm[2] = 1 - color_norm[2]
    return color_norm * 2 - 1


def generate_candidates_random(vertex, x, y, k=2):
    center_vertex = vertex[x, y].reshape(1, 3)
    height, width = vertex.shape[:2]

    candidate_normals = np.zeros(shape=(10, 10, 3))
    if center_vertex.sum() == 0:
        return candidate_normals

    # get all the vectors on the plane
    neighbours = vertex[max(x - k, 0):min(x + k + 1, width - 1), max(0, y - k):min(y + k + 1, height - 1)]
    neighbours = neighbours.reshape(neighbours.shape[0] * neighbours.shape[1], 3)
    neighbours = np.delete(neighbours, [neighbours.shape[0] // 2], axis=0)  # delete center
    plane_vectors = neighbours - center_vertex
    # shuffle the vectors
    np.random.shuffle(plane_vectors)
    random_indices = np.random.choice(plane_vectors.shape[0], size=8, replace=False)
    plane_vectors = plane_vectors[random_indices, :]

    vec_num = plane_vectors.shape[0]
    candidate_normals_all = np.zeros(shape=(100, 3))
    # get all combinations of vectors
    sample_indices = []
    for choosen_vector_nums in reversed(range(3, vec_num + 1)):
        sample_indices = sample_indices + list(itertools.combinations(range(vec_num), choosen_vector_nums))

    candidate_normals_num = int(len(sample_indices))

    if candidate_normals_num > 100:
        sample_indices = random.choices(sample_indices, k=100)

    i = 0
    for sample_index in sample_indices:

        # calculate normals using svd
        # print(f'sample_index={sample_index}')
        candidate_vectors = np.take(plane_vectors, sample_index, axis=0)
        u, s, vh = np.linalg.svd(candidate_vectors)
        candidate_normal = vh.T[:, -1]
        normal = normal_point2view_point(candidate_normal, center_vertex.reshape(3), np.array([0, 0, 0]))
        if np.linalg.norm(normal) != 1:
            normal = normal / np.linalg.norm(normal)

        candidate_normals_all[i] = normal
        i += 1

    # if candidates less than 100, copy last normal, until 100 normals candidates generated
    if candidate_normals_num < 100:
        for j in range(i, 100):
            candidate_normals_all[j] = candidate_normals_all[i - 1]

    # order all the normals
    candidate_normals_all.sort(axis=0)

    # only pick first 100 candidates
    candidate_normals = candidate_normals_all[:100, :].reshape(10, 10, 3)

    return candidate_normals


def generate_candidates(vertex, x, y, k=1):
    center_vertex = vertex[x, y].reshape(1, 3)
    height, width = vertex.shape[:2]

    candidate_normals = np.zeros(shape=(10, 10, 3))
    if center_vertex.sum() == 0:
        return candidate_normals

    # get all the vectors on the plane
    neighbours = vertex[max(x - k, 0):min(x + k + 1, width - 1), max(0, y - k):min(y + k + 1, height - 1)]
    neighbours = neighbours.reshape(neighbours.shape[0] * neighbours.shape[1], 3)
    neighbours = np.delete(neighbours, [neighbours.shape[0] // 2], axis=0)  # delete center
    plane_vectors = neighbours - center_vertex
    # shuffle the vectors
    np.random.shuffle(plane_vectors)

    vec_num = plane_vectors.shape[0]

    # get all combinations of vectors
    candidate_normals_num = int(np.sum([math.comb(vec_num, i) for i in range(3, vec_num + 1)]))
    # print(f'candidate_normals_num={candidate_normals_num}')
    # print(f'vec_num={vec_num}')

    candidate_normals_all = np.zeros(shape=(max(candidate_normals_num, 100), 3))

    i = 0
    for choosen_vector_nums in reversed(range(3, vec_num + 1)):
        sample_indices = list(itertools.combinations(range(vec_num), choosen_vector_nums))
        for sample_index in sample_indices:

            # calculate normals using svd
            # print(f'sample_index={sample_index}')
            candidate_vectors = np.take(plane_vectors, sample_index, axis=0)
            u, s, vh = np.linalg.svd(candidate_vectors)
            candidate_normal = vh.T[:, -1]
            normal = normal_point2view_point(candidate_normal, center_vertex.reshape(3), np.array([0, 0, 0]))
            if np.linalg.norm(normal) != 1:
                normal = normal / np.linalg.norm(normal)

            candidate_normals_all[i] = normal
            i += 1

    # if candidates less than 100, copy last normal, until 100 normals candidates generated
    if candidate_normals_num < 100:
        for j in range(i, 100):
            candidate_normals_all[j] = candidate_normals_all[i - 1]

    np.random.shuffle(candidate_normals_all)
    # only pick first 100 candidates
    candidate_normals = candidate_normals_all[:100, :].reshape(10, 10, 3)

    return candidate_normals


def generate_candidates_all(vertex, k=1):
    height, width = vertex.shape[:2]
    mask = vertex.sum(axis=2) == 0

    # neighbors are all the elements in a square matrix without center
    # the side length of square matrix is 2 * k + 1
    neighbours_num = (2 * k + 1) ** 2 - 1

    # for each vertex, a set of candidate normals will be calculated from  planes constructing by
    # some or all it's neighbors and itself, all the combination of neighbors will be considered to calculate a normal
    normal_per_vertex_num = np.sum([math.comb(neighbours_num, i) for i in range(3, neighbours_num)])

    candidate_normals_grid = np.zeros(shape=(height, width, normal_per_vertex_num, 3))
    for i in range(k, height):
        for j in range(k, width):
            if not mask[i, j]:
                candidate_normals = []
                # get all the vectors on the plane
                neighbours = vertex[i - k:i + k, j - k:j + k]
                neighbors = neighbours.reshape(neighbours.shape[0] * neighbours.shape[1], 3)
                neighbors = np.delete(neighbors, np.where(neighbors == vertex[i, j]), axis=0)  # delete center
                plane_vectors = neighbors - vertex[i, j]
                vec_num = plane_vectors.shape[0]

                # get all combinations of vectors
                samples_all = []
                for choosen_vector_nums in range(3, vec_num):
                    sample_indices = list(itertools.combinations(range(vec_num), choosen_vector_nums))
                    for sample_index in sample_indices:

                        candidate_vectors = plane_vectors[sample_index]
                        u, s, vh = np.linalg.svd(candidate_vectors)
                        candidate_normal = vh.T[:, -1]
                        normal = normal_point2view_point(candidate_normal, vertex[i][j], np.array([0, 0, 0]))
                        if np.linalg.norm(normal) != 1:
                            normal = normal / np.linalg.norm(normal)
                        candidate_normals.append(normal)

                candidate_normals_grid[i, j] = candidate_normals

    return candidate_normals_grid