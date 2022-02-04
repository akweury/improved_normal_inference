import numpy as np
import cv2
import torch
import json


def load_8bitImage(root):
    img = cv2.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    img = torch.tensor(img).unsqueeze(2)
    img = np.array(img)
    img[np.isnan(img)] = 0

    return img.astype(np.float32)


def load_24bitNormal(root, R):
    nml = cv2.imread(root, -1)
    nml = np.array(nml, dtype=np.float32)
    nml = np.array(nml)
    nml[np.isnan(nml)] = 0
    nml = nml[:, :, [2, 1, 0]]
    mask = (nml[:, :, 0] == 0) & (nml[:, :, 1] == 0) & (nml[:, :, 2] == 0)

    nml[:, :, 0] = (~mask) * (nml[:, :, 0] / 255.0 * 2.0 - 1.0)
    nml[:, :, 1] = (~mask) * (nml[:, :, 1] / 255.0 * 2.0 - 1.0)
    nml[:, :, 2] = (~mask) * (1.0 - nml[:, :, 2] / 255.0 * 2.0)

    nml = R.transpose(0, 1) @ torch.tensor(nml).permute(2, 0, 1).reshape(3, -1)
    nml = nml.reshape(3, 512, 512).permute(1, 2, 0)

    return np.array(nml).astype(np.float32)


def load_scaled16bitImage(root, minVal, maxVal):
    img = cv2.imread(root, -1)
    img = np.array(img, dtype=np.float32)
    mask = (img == 0)
    img = img / 65535 * (maxVal - minVal) + minVal
    img[np.isnan(img)] = 0

    img = torch.tensor((~mask) * img).unsqueeze(2)
    img = np.array(img)

    return img.astype(np.float32)


def depth2vertex(depth, K, R, t):
    camOrig = -R.transpose(0, 1) @ t.unsqueeze(1)

    X = torch.arange(0, depth.size(2)).repeat(depth.size(1), 1) - K[0, 2]
    Y = torch.transpose(torch.arange(0, depth.size(1)).repeat(depth.size(2), 1), 0, 1) - K[1, 2]
    Z = torch.ones(depth.size(1), depth.size(2)) * K[0, 0]
    Dir = torch.cat((X.unsqueeze(2), Y.unsqueeze(2), Z.unsqueeze(2)), 2)

    vertex = Dir * (depth.squeeze(0) / torch.norm(Dir, dim=2)).unsqueeze(2).repeat(1, 1, 3)
    vertex = R.transpose(0, 1) @ vertex.permute(2, 0, 1).reshape(3, -1)
    vertex = camOrig.unsqueeze(1).repeat(1, 512, 512) + vertex.reshape(3, 512, 512)
    vertex = vertex.permute(1, 2, 0)
    return np.array(vertex)


def cameraVisualization():
    points = []
    for p in range(-50, 51):
        ps = float(p) / 100.0
        points.append([ps, 0.5, 0.5])
        points.append([ps, -0.5, 0.5])
        points.append([ps, 0.5, -0.5])
        points.append([ps, -0.5, -0.5])
        points.append([0.5, ps, 0.5])
        points.append([-0.5, ps, 0.5])
        points.append([0.5, ps, -0.5])
        points.append([-0.5, ps, -0.5])
        points.append([0.5, 0.5, ps])
        points.append([0.5, -0.5, ps])
        points.append([-0.5, 0.5, ps])
        points.append([-0.5, -0.5, ps])

    for p in range(-30, 31):
        ps = float(p) / 100.0
        points.append([ps, 0.3, 0.3 + 0.8])
        points.append([ps, -0.3, 0.3 + 0.8])
        points.append([ps, 0.3, -0.3 + 0.8])
        points.append([ps, -0.3, -0.3 + 0.8])
        points.append([0.3, ps, 0.3 + 0.8])
        points.append([-0.3, ps, 0.3 + 0.8])
        points.append([0.3, ps, -0.3 + 0.8])
        points.append([-0.3, ps, -0.3 + 0.8])
        points.append([0.3, 0.3, ps + 0.8])
        points.append([0.3, -0.3, ps + 0.8])
        points.append([-0.3, 0.3, ps + 0.8])
        points.append([-0.3, -0.3, ps + 0.8])

    return points


def lightVisualization():
    points = []
    for px in range(-5, 6):
        for py in range(-5, 6):
            for pz in range(-5, 6):
                p = np.array([px, py, pz]).astype(np.float) / 10
                if np.linalg.norm(p) > 0:
                    points.append(p / np.linalg.norm(p))
    return points


def writePLY(vertex, normal, image, mask, filename, cameraPoints=None, lightPoints=None):
    numPoints = np.sum(mask)
    if cameraPoints != None:
        numPoints += len(cameraPoints)
    if lightPoints != None:
        numPoints += len(lightPoints)

    file = open(filename, "w")
    file.write('ply\n')
    file.write('format ascii 1.0\n')
    file.write('element vertex ' + str(numPoints) + '\n')
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property float nx\n')
    file.write('property float ny\n')
    file.write('property float nz\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('end_header\n')

    for i in range(vertex.shape[0]):
        for j in range(vertex.shape[1]):
            if mask[i, j] == True:
                file.write(str(vertex[i, j, 0]) + ' ' + str(vertex[i, j, 1]) + ' ' + str(vertex[i, j, 2]) + ' ')
                file.write(str(normal[i, j, 0]) + ' ' + str(normal[i, j, 1]) + ' ' + str(normal[i, j, 2]) + ' ')
                file.write(
                    str(int(image[i, j, 0])) + ' ' + str(int(image[i, j, 0])) + ' ' + str(int(image[i, j, 0])) + '\n')
    if cameraPoints != None:
        for i in range(len(cameraPoints)):
            file.write(str(cameraPoints[i][0]) + ' ' + str(cameraPoints[i][1]) + ' ' + str(cameraPoints[i][2]) + ' ')
            file.write(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) + ' ')
            file.write(str(int(255)) + ' ' + str(int(0)) + ' ' + str(int(0)) + '\n')
    if lightPoints != None:
        for i in range(len(lightPoints)):
            file.write(str(lightPoints[i][0]) + ' ' + str(lightPoints[i][1]) + ' ' + str(lightPoints[i][2]) + ' ')
            file.write(str(0.0) + ' ' + str(0.0) + ' ' + str(0.0) + ' ')
            file.write(str(int(255)) + ' ' + str(int(255)) + ' ' + str(int(0)) + '\n')

    file.close()


# ----------------------------------------------------------------------------------
path = 'SyntheticDataSet/CapturedData/00000.'
# path = 'RealDataSet/00000.'

f = open(path + 'data0.json')
data = json.load(f)
f.close()

image = load_8bitImage(path + 'image0.png')
depth = load_scaled16bitImage(path + 'depth0.png', data['minDepth'], data['maxDepth'])
vertex = depth2vertex(torch.tensor(depth).permute(2, 0, 1), torch.tensor(data['K']), torch.tensor(data['R']),
                      torch.tensor(data['t']))
normal = load_24bitNormal(path + 'normal0.png', torch.tensor(data['R']))
mask = np.sum(np.abs(vertex), axis=2) != 0

cameraPoints = cameraVisualization()
for i in range(len(cameraPoints)):
    cameraPoints[i] = np.array(data['R']).transpose() @ cameraPoints[i] / 4 - np.array(
        data['R']).transpose() @ np.array(data['t'])
lightPoints = lightVisualization()
for i in range(len(lightPoints)):
    lightPoints[i] = lightPoints[i] / 8 + np.array(data['lightPos'])

writePLY(vertex, normal, image, mask, path + 'pointCloud0.ply', cameraPoints=cameraPoints, lightPoints=lightPoints)
