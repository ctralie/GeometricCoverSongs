import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('SequenceAlignment')
import _SequenceAlignment
import SequenceAlignment
import scipy.misc

#Get a self-similarity matrix
def getSSM(x, DPixels, doPlot = False):
    D = np.sum(x**2, 1)[:, None]
    D = D + D.T - 2*x.dot(x.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    if doPlot:
        plt.subplot(121)
        plt.imshow(D, interpolation = 'none')
        plt.subplot(122)
        plt.imshow(scipy.misc.imresize(D, (DPixels, DPixels)), interpolation = 'none')
        plt.show()
    if not (D.shape[0] == DPixels):
        return (D, scipy.misc.imresize(D, (DPixels, DPixels)))
    return (D, D)

#############################################################################
## Code for dealing with cross-similarity matrices
#############################################################################

def getCSM(X, Y):
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getCSMEMD1D(X, Y):
    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1]
    XC = np.cumsum(X, 1)
    YC = np.cumsum(Y, 1)
    D = np.zeros((M, N))
    for k in range(K):
        xc = XC[:, k]
        yc = YC[:, k]
        D += np.abs(xc[:, None] - yc[None, :])
    return D

def getCSMCosine(X, Y):
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y**2, 1))
    YNorm[YNorm == 0] = 1
    D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
    D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
    return D

#Turn a cross-similarity matrix into a binary cross-simlarity matrix
#If Kappa = 0, take all neighbors
#If Kappa < 1 it is the fraction of mutual neighbors to consider
#Otherwise Kappa is the number of mutual neighbors to consider
def CSMToBinary(D, Kappa):
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    cols = np.argsort(D, 1)
    temp, rows = np.meshgrid(np.arange(M), np.arange(N))
    cols = cols[:, 0:NNeighbs].flatten()
    rows = rows[:, 0:NNeighbs].flatten()
    ret = np.zeros((N, M))
    ret[rows, cols] = 1
    return ret

#Take the binary AND between the nearest neighbors in one direction
#and the other
def CSMToBinaryMutual(D, Kappa):
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa)
    return B1*B2.T

#Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
#computing of all of the smith waterman scores for a pair of songs.
#Features1 and Features2 are Mxk and Nxk matrices of features, respectively
#The type of cross-similarity can also be specified
def getCSMSmithWatermanScores(args, doPlot = False):
    [Features1, Features2, Kappa, Type] = args
    if Type == "Euclidean":
        CSM = getCSM(Features1, Features2)
    elif Type == "Cosine":
        CSM = getCSMCosine(Features1, Features2)
    elif Type == "EMD1D":
        CSM = getCSMEMD1D(Features1, Features2)
    DBinary = CSMToBinaryMutual(CSM, Kappa)
    if doPlot:
        (maxD, D) = SequenceAlignment.swalignimpconstrained(DBinary)
        plt.subplot(131)
        plt.imshow(CSM, interpolation = 'none')
        plt.title('CSM')
        plt.subplot(132)
        plt.imshow(DBinary, interpolation = 'none')
        plt.title("CSM Binary K=%g"%Kappa)
        plt.subplot(133)
        plt.imshow(D, interpolation = 'none')
        plt.title("Smith Waterman Score = %g"%maxD)
    return _SequenceAlignment.swalignimpconstrained(DBinary)
