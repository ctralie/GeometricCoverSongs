import SequenceAlignment as SA
import _SequenceAlignment as SAC
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import sys

def compareTimes():
    """
    Show how much raw C saves time wise with a double for loop
    """
    D = np.random.rand(1000, 1000)
    D = D < 0.1
    D = np.array(D, dtype='double')

    start = time.time()
    ans = SAC.swalignimpconstrained(D)
    end = time.time()
    print("Time elapsed C: %g seconds, ans = %g"%(end-start, ans))

    start = time.time()
    ans = SA.swalignimpconstrained(D)[0]
    end = time.time()
    print("Time elapsed raw python: %g seconds, ans = %g"%(end - start, ans))

def testBacktrace():
    np.random.seed(100)
    t = np.linspace(0, 1, 300)
    t1 = t
    X1 = 0.3*np.random.randn(400, 2)
    X1[50:50+len(t1), 0] = np.cos(2*np.pi*t1)
    X1[50:50+len(t1), 1] = np.sin(4*np.pi*t1)
    t2 = t**2
    X2 = 0.3*np.random.randn(350, 2)
    X2[0:len(t2), 0] = np.cos(2*np.pi*t2)
    X2[0:len(t2), 1] = np.sin(4*np.pi*t2)
    CSM = getCSM(X1, X2)
    CSM = CSMToBinaryMutual(CSM, 0.1)
    (maxD, D, path) = SA.SWBacktrace(CSM)
    sio.savemat("D.mat", {"D":D})
    print("maxD = %g"%maxD)

    plt.subplot(221)
    plt.plot(X1[:, 0], X1[:, 1])
    plt.subplot(222)
    plt.plot(X2[:, 0], X2[:, 1])
    plt.subplot(223)
    plt.imshow(CSM, cmap = 'afmhot')
    plt.subplot(224)
    plt.imshow(D, cmap = 'afmhot')
    plt.hold(True)
    path = np.array(path)
    plt.title("%g"%maxD)
    plt.scatter(path[:, 1], path[:, 0], 20, edgecolor = 'none')
    plt.show()

if __name__ == "__main__":
    compareTimes()
    #testBacktrace()
