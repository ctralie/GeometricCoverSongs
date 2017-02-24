import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda.cumath
import skcuda.misc
import skcuda.linalg as linalg
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio
from CSMSSMTools import *

from pycuda.compiler import SourceModule

bitonicSort_ = None
getSumSquares_ = None
finishCSM_ = None

def initParallelAlgorithms():
    global bitonicSort_
    fin = open("ParallelAlgorithms/bitonicSort.cu")
    mod = SourceModule(fin.read())
    fin.close()
    bitonicSort_ = mod.get_function("bitonicSort")

    global finishCSM_
    global getSumSquares_
    fin = open("ParallelAlgorithms/CSMHelper.cu")
    mod = SourceModule(fin.read())
    fin.close()
    finishCSM_ = mod.get_function("finishCSM")
    getSumSquares_ = mod.get_function("getSumSquares")

    #Run each of the algorithms on dummy data so that they're pre-compiled

    #1) Bitonic Sort
    X = np.random.randn(16, 16)
    N = np.int32(16)
    NPow2 = N
    NThreads = N/2
    XG = gpuarray.to_gpu(X)
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(X.shape[0], 1), shared=4*NPow2)

    linalg.init()
    #2) Other primitive operations
    NegXDotX = linalg.dot(XG, XG)
    XPlusX = skcuda.misc.add(XG, XG)
    XSqr = skcuda.misc.multiply(XG, XG)
    XSqr = skcuda.misc.sum(XSqr, 1)
    XPlusCol = skcuda.misc.add_matvec(XG, XSqr, 0)

def bitonicSort(XG):
    N = np.int32(XG.shape[1])
    NPow2 = np.int32(2**np.ceil(np.log2(N)))
    N2 = NPow2/2
    NThreads = min(N2, 512)
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(XG.shape[0], 1), shared=4*NPow2)

def testBitonicSort(N, doPlot = False):
    X = np.array(np.random.rand(N, N), dtype=np.float32)
    XG = gpuarray.to_gpu(X)

    tic = time.time()
    bitonicSort(XG)
    toc = time.time()
    GPUTime = toc-tic

    tic = time.time()
    X2 = np.sort(X, 1)
    toc = time.time()
    CPUTime = toc-tic

    print("N = %i"%N)
    print("Elapsed Time CPU: %g"%CPUTime)
    print("Elapsed Time GPU: %g (Ratio %.3g)"%(GPUTime, CPUTime/GPUTime))
    print("AllClose: ", np.allclose(np.sort(X, 1), XG.get()))

    if doPlot:
        plt.subplot(121)
        plt.imshow(X, interpolation = 'none')
        plt.subplot(122)
        plt.imshow(XG.get(), interpolation = 'none')
        plt.show()
    return (CPUTime, GPUTime)

def testBitonicSortTimeRatios(sizes, NTrials):
    np.random.seed(100)
    CPUTimes = np.zeros((len(sizes), NTrials))
    GPUTimes = np.zeros((len(sizes), NTrials))
    for i in range(len(sizes)):
        N = sizes[i]
        for t in range(NTrials):
            (CPUTimes[i, t], GPUTimes[i, t]) = testBitonicSort(N)
    sio.savemat("Timings.mat", {"CPUTimes":CPUTimes, "GPUTimes":GPUTimes})

def roundUpPow2(x):
    return np.array(int(2**np.ceil(np.log2(float(x)))), dtype=np.int32)

def getCSMGPU(XG, YG):
    #YGT = linalg.transpose(YG)
    x = 1

def testCSM(M, N, NOthers):
    X = np.array(np.random.randn(M, 25*25), dtype = np.float32)
    Y = np.array(np.random.randn(N*NOthers, 25*25), dtype = np.float32)

    tic = time.time()
    XG = gpuarray.to_gpu(X)
    YG = gpuarray.to_gpu(Y)
    XSqr = gpuarray.to_gpu(np.array(np.zeros(X.shape[0]), dtype=np.float32))
    YSqr = gpuarray.to_gpu(np.array(np.zeros(Y.shape[0]), dtype=np.float32))
    toc = time.time()
    print "GPU Copy Time: ", toc - tic

    tic = time.time()
    CSM1 = getCSM(Y, X)
    toc = time.time()
    CPUTime = toc-tic

    ticg = time.time()

    #Step 1: Sum of squares across rows
    dim = np.array(X.shape[1], dtype=np.int32)
    dimpow2 = roundUpPow2(dim)
    NThreads = min(dimpow2, 512)
    getSumSquares_(YG, YSqr, dim, dimpow2, block=(NThreads, 1, 1), grid=(YG.shape[0], 1), shared=4*dimpow2)
    getSumSquares_(XG, XSqr, dim, dimpow2, block=(NThreads, 1, 1), grid=(XG.shape[0], 1), shared=4*dimpow2)

    #Step 2: Do multiplication part
    tic = time.time()
    XGT = linalg.transpose(XG)
    CSM = linalg.dot(YG, XGT)
    print "Elapsed time multiply: ", time.time() - tic

    #Step 3: Add everything together
    Mp = np.array(XG.shape[0], dtype=np.int32)
    Np = np.array(YG.shape[0], dtype=np.int32)
    MPow2 = roundUpPow2(XG.shape[0])
    #CSM is N x M
    tic = time.time()
    finishCSM_(CSM, XSqr, YSqr, Np, Mp, MPow2, block=(int(MPow2), 1, 1), grid=(YG.shape[0], 1))
    print "Elapsed Time Finish: ", time.time() - toc


    tocg = time.time()
    GPUTime = tocg - ticg

    #print("Elapsed Time CPU: %g"%CPUTime)
    #print("Elapsed Time GPU: %g"%GPUTime)

    CSM = CSM.get()
    CSM = CSM[0:N, :]
    CSM1 = CSM1[0:N, :]
    plt.subplot(131)
    plt.imshow(CSM1, interpolation = 'none', cmap = 'afmhot')
    plt.subplot(132)
    plt.imshow(CSM, interpolation = 'none', cmap = 'afmhot')
    plt.subplot(133)
    plt.imshow(CSM1 - CSM, interpolation = 'none', cmap = 'spectral')
    plt.show()



if __name__ == '__main__':
    np.random.seed(100)
    initParallelAlgorithms()
    testCSM(1000, 1000, 60)

if __name__ == '__main__2':
    initParallelAlgorithms()
    testBitonicSort(3000, True)
