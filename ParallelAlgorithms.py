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
CSM = None

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

def roundUpPow2(x):
    return np.int32(int(2**np.ceil(np.log2(float(x)))))

def bitonicSort(XG):
    N = np.int32(XG.shape[1])
    NPow2 = np.int32(2**np.ceil(np.log2(N)))
    N2 = NPow2/2
    NThreads = min(N2, 512)
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(XG.shape[0], 1), shared=4*NPow2)

def getCSMGPU(XG, YG):
    tbegin = time.time()
    GPUNeg2 = gpuarray.to_gpu(np.array([-2.0], dtype=np.float32))
    YGT = linalg.transpose(YG)
    XSqr = skcuda.misc.multiply(XG, XG)
    XSqr = skcuda.misc.sum(XSqr, 1)
    YSqr = skcuda.misc.multiply(YG, YG)
    YSqr = skcuda.misc.sum(YSqr, 1)
    C = linalg.dot(XG, YGT)
    C = skcuda.misc.multiply(GPUNeg2, C)
    skcuda.misc.add_matvec(C, XSqr, 0, C)
    skcuda.misc.add_matvec(C, YSqr, 1, C)
    return C

def getCSMGPU2(XG, YG):
    #Step 1: Sum of squares across rows
    dim = np.int32(XG.shape[1])
    dimpow2 = roundUpPow2(dim)
    NThreads = np.int32(min(dimpow2, 512))
    XSqr = gpuarray.empty(XG.shape[0], np.float32)
    YSqr = gpuarray.empty(YG.shape[0], np.float32)
    getSumSquares_(XG, XSqr, dim, dimpow2, block=(NThreads, 1, 1), grid=(XG.shape[0], 1), shared=4*dimpow2)
    getSumSquares_(YG, YSqr, dim, dimpow2, block=(NThreads, 1, 1), grid=(YG.shape[0], 1), shared=4*dimpow2)

    #Step 2: Do multiplication part
    YGT = linalg.transpose(YG)
    CSM = linalg.dot(XG, YGT)

    #Step 3: Add everything together
    Mp = np.array(XG.shape[0], dtype=np.int32)
    Np = np.array(YG.shape[0], dtype=np.int32)
    MPow2 = roundUpPow2(XG.shape[0])
    NThreads = min(MPow2, 512)
    #CSM is N x M
    finishCSM_(CSM, XSqr, YSqr, Np, Mp, MPow2, block=(NThreads, 1, 1), grid=(YG.shape[0], 1))
    return (CSM, XSqr, YSqr)

def testBitonicSort(N, doPlot = False):
    X = np.array(np.random.rand(N*400, N), dtype=np.float32)
    tic = time.time()
    XG = gpuarray.to_gpu(X)
    toc = time.time()
    print("Elapsed memcopy time: %g"%(toc-tic))

    tic = time.time()
    bitonicSort(XG)
    skcuda.misc.add(XG, XG)
    toc = time.time()
    GPUTime = toc-tic

    tic = time.time()
    X2 = np.sort(X, 1)
    toc = time.time()
    CPUTime = toc-tic

    tic = time.time()
    J = np.argpartition(X, N/10, 1)
    toc = time.time()
    print("Elapsed Time Partition: %g"%(toc-tic))

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

def testCSMTimes():
    N = 800
    K = 1
    X = np.array(np.random.randn(N, 25*25), dtype = np.float32)
    Y = np.array(np.random.randn(N*K, 25*25), dtype=np.float32)
    XG = gpuarray.to_gpu(X)
    YG = gpuarray.to_gpu(Y)

    tic = time.time()
    CSM = getCSM(X, Y)
    CPUTime = time.time() - tic

    tic = time.time()
    #(CSMG, XSqr, YSqr) = getCSMGPU2(XG, YG)
    CSMG = getCSMGPU(XG, YG)
    CSMG = CSMG.get()
    GPUTime = time.time() - tic

    #plt.plot(XSqr.get(), np.sum(X**2, 1), '.')
    #plt.show()

    print("CPUTime: %g"%CPUTime)
    print("GPUTime: %g"%GPUTime)



    plt.subplot(131)
    plt.imshow(CSM, interpolation = 'none', cmap = 'afmhot')
    plt.subplot(132)
    plt.imshow(CSMG, interpolation = 'none', cmap = 'afmhot')
    plt.subplot(133)
    plt.imshow(CSM - CSMG, interpolation = 'none', cmap = 'spectral')
    plt.show()


if __name__ == '__main__2':
    np.random.seed(100)
    initParallelAlgorithms()
    t = testCSM(1000, 1000, 60, doPlot = False)
    print("Return time: %g"%(time.time() - t))

if __name__ == '__main__':
    initParallelAlgorithms()
    #testBitonicSort(800, False)
    testCSMTimes()
