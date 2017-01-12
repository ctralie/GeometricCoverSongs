import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import scipy.io as sio
from scipy import sparse
import time

from CSMSSMTools import *
from SimilarityFusion import *

def getDiffusionMap(SSM, Kappa, t = -1, includeDiag = True, thresh = 5e-4, NEigs = 51):
    """
    :param SSM: Metric between all pairs of points
    :param Kappa: Number in (0, 1) indicating a fraction of nearest neighbors
                used to autotune neighborhood size
    :param 
    """
    N = SSM.shape[0]
    #Use the letters from the delaPorte paper
    K = getW(SSM, int(Kappa*N))
    if not includeDiag:
        np.fill_diagonal(K, np.zeros(N))
    RowSumSqrt = np.sqrt(np.sum(K, 1))
    DInvSqrt = sparse.diags([1/RowSumSqrt], [0])

    #Symmetric normalized Laplacian
    Pp = (K/RowSumSqrt[None, :])/RowSumSqrt[:, None]
    Pp[Pp < thresh] = 0
    Pp = sparse.csr_matrix(Pp)

    lam, X = sparse.linalg.eigsh(Pp, NEigs, which='LM')
    lam = lam/lam[-1] #In case of numerical instability

    #Check to see if autotuning
    if t > -1:
        lamt = lam**t
    else:
        #Autotuning diffusion time
        lamt = np.array(lam)
        lamt[0:-1] = lam[0:-1]/(1-lam[0:-1])

    #Do eigenvector version
    V = DInvSqrt.dot(X) #Right eigenvectors
    M = V*lamt[None, :]
    return M/RowSumSqrt[:, None] #Put back into orthogonal Euclidean coordinates

def getPinchedCircle(N):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    x = np.zeros((N, 2))
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    return x

def getTorusKnot(N, p, q):
    t = np.linspace(0, 2*np.pi, N+1)[0:N]
    X = np.zeros((N, 3))
    r = np.cos(q*t) + 2
    X[:, 0] = r*np.cos(p*t)
    X[:, 1] = r*np.sin(p*t)
    X[:, 2] = -np.sin(q*t)
    return X

if __name__ == '__main__':
    zeroReturn = True
    N = 1000
    X = getTorusKnot(N, 2, 5)
    sio.savemat("X.mat", {"X":X})
    tic = time.time()
    [SSM, _] = getSSM(X, N)
    toc = time.time()
    print "Elapsed time SSM: ", toc - tic
    Kappa = 0.02

    t = -1
    plt.clf()
    tic = time.time()
    M = getDiffusionMap(SSM, Kappa, t)
    toc = time.time()
    print "Total time diffusion: ", toc-tic
    (SSM, _) = getSSM(M, N)
    plt.subplot(121)
    plt.plot(M[:, -3], M[:, -2], '.')
    plt.axis('equal')
    plt.subplot(122)
    plt.imshow(SSM)
    plt.savefig("Diffusion%i.png"%t)

    sio.savemat("X.mat", {"X":X, "M":M})
