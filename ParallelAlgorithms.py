import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt
import time
import scipy.io as sio

from pycuda.compiler import SourceModule

bitonicSort_ = None

def initParallelAlgorithms():
    global bitonicSort_
    fin = open("ParallelAlgorithms/bitonicSort.cu")
    mod = SourceModule(fin.read())
    fin.close()
    bitonicSort_ = mod.get_function("bitonicSort")

    #Run each of the algorithms on dummy data so that they're pre-compiled
    X = np.random.randn(16, 16)
    N = np.int32(16)
    NPow2 = N
    NThreads = N/2
    XG = gpuarray.to_gpu(X)
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(X.shape[0], 1), shared=4*NPow2)

def bitonicSort(XG):
    N = np.int32(XG.shape[1])
    NPow2 = np.int32(2**np.ceil(np.log2(N)))
    N2 = NPow2/2
    NThreads = min(N2, 512)
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(XG.shape[0], 1), shared=4*NPow2)

def testBitonicSortTimeRatios(sizes, NTrials):
    initParallelAlgorithms()
    np.random.seed(100)
    CPUTimes = np.zeros((len(sizes), NTrials))
    GPUTimes = np.zeros((len(sizes), NTrials))
    for i in range(len(sizes)):
        N = sizes[i]
        for t in range(NTrials):
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
            CPUTimes[i, t] = CPUTime
            GPUTimes[i, t] = GPUTime
    sio.savemat("Timings.mat", {"CPUTimes":CPUTimes, "GPUTimes":GPUTimes})

if __name__ == '__main__':
    testBitonicSortTimeRatios(range(10, 3000, 10), 10)

if __name__ == '__main__2':
    initParallelAlgorithms()
    np.random.seed(100)
    N = 3000
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

    print("Elapsed Time CPU: %g"%CPUTime)
    print("Elapsed Time GPU: %g (Ratio %.3g)"%(GPUTime, CPUTime/GPUTime))

    plt.subplot(121)
    plt.imshow(X, interpolation = 'none')
    plt.subplot(122)
    plt.imshow(XG.get(), interpolation = 'none')
    #plt.subplot(133)
    XDiff = X - XG.get()
    #plt.imshow(XDiff, interpolation = 'none')
    #print np.max(XDiff)
    plt.show()
