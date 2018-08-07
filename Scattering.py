"""
Code that wraps around Matlab ScatNet to compute scattering transforms
"""
import numpy as np
import scipy.io as sio
import subprocess
import matplotlib.pyplot as plt
import os
import sys

def getPrefix():
    return 'scatnet-0.2'

def getScatteringTransform(imgs, renorm=True):
    intrenorm = 0
    if renorm:
        intrenorm = 1
    prefix = getPrefix()
    argimgs = np.zeros((imgs[0].shape[0], imgs[0].shape[1], len(imgs)))
    for i, img in enumerate(imgs):
        argimgs[:, :, i] = img
    sio.savemat("%s/x.mat"%prefix, {"x":argimgs, "renorm":intrenorm})
    subprocess.call(["matlab", "-nodisplay", "-nosplash", "-nodesktop", "-r", \
                    "cd %s; getScatteringImages; exit;"%(prefix)])
    res = sio.loadmat("%s/res.mat"%prefix)['res']
    images = []
    for i in range(len(res[0])):
        image = []
        for j in range(len(res[0][i][0])):
            image.append(res[0][i][0][j])
        images.append(image)
    return images

def flattenCoefficients(images):
    ret = []
    for im in images:
        ret.append(im.flatten())
    return np.array(ret)

def poolFeatures(image, res):
    M = int(image.shape[0]/res)
    N = int(image.shape[1]/res)
    ret = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            ret[i, j] = np.mean(image[i*res:(i+1)*res, j*res:(j+1)*res])
    return ret

