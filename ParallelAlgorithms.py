import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import matplotlib.pyplot as plt
import time

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
    print "N = %i, NPow2 = %i, N2 = %i, NThreads = %i"%(N, NPow2, N2, NThreads)

    tic = time.time()
    bitonicSort_(XG, N, NPow2, block=(NThreads, 1, 1), grid=(X.shape[0], 1), shared=4*NPow2)
    toc = time.time()
    print "Elapsed Time Bitonic GPU: ", toc-tic

if __name__ == '__main__':
    initParallelAlgorithms()
    np.random.seed(100)
    X = np.array(np.random.rand(1000, 1000), dtype=np.float32)
    XG = gpuarray.to_gpu(X)

    bitonicSort(XG)

    tic = time.time()
    X2 = np.sort(X, 1)
    toc = time.time()
    print "Elapsed Time CPU: ", toc-tic

    plt.subplot(121)
    plt.imshow(X, interpolation = 'none')
    plt.subplot(122)
    plt.imshow(XG.get(), interpolation = 'none')
    plt.show()
