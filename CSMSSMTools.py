"""
Programmer: Chris Tralie
Purpose: A variety of tools for computing self-similarity matrices (SSMs)
and cross-similarity matrices (CSMs), with a particular
emphasis on speeding up Euclidean SSMs/CSMs.
"""
import numpy as np
import scipy.misc
import scipy.interpolate
import matplotlib.pyplot as plt
import SequenceAlignment.SequenceAlignment as SA
import SequenceAlignment._SequenceAlignment as SAC
from SimilarityFusion import *
import time
from multiprocessing import Pool as PPool

def imresize(D, dims, kind='cubic', use_scipy=False):
    """
    Resize a floating point image
    Parameters
    ----------
    D : ndarray(M1, N1)
        Original image
    dims : tuple(M2, N2)
        The dimensions to which to resize
    kind : string
        The kind of interpolation to use
    use_scipy : boolean
        Fall back to scipy.misc.imresize.  This is a bad idea
        because it casts everything to uint8, but it's what I
        was doing accidentally for a while
    Returns
    -------
    D2 : ndarray(M2, N2)
        A resized array
    """
    if use_scipy:
        return scipy.misc.imresize(D, dims)
    else:
        M, N = dims
        x1 = np.array(0.5 + np.arange(D.shape[1]), dtype=np.float32)/D.shape[1]
        y1 = np.array(0.5 + np.arange(D.shape[0]), dtype=np.float32)/D.shape[0]
        x2 = np.array(0.5 + np.arange(N), dtype=np.float32)/N
        y2 = np.array(0.5 + np.arange(M), dtype=np.float32)/M
        f = scipy.interpolate.interp2d(x1, y1, D, kind=kind)
        return f(x2, y2)

def getSSM(X, DPixels, doPlot = False):
    """
    Compute a Euclidean self-similarity image between a set of points
    :param X: An Nxd matrix holding the d coordinates of N points
    :param DPixels: The image will be resized to this dimensions
    :param doPlot: If true, show a plot comparing the original/resized images
    :return: A tuple (D, DResized)
    """
    D = np.sum(X**2, 1)[:, None]
    D = D + D.T - 2*X.dot(X.T)
    D[D < 0] = 0
    D = 0.5*(D + D.T)
    D = np.sqrt(D)
    if doPlot:
        plt.subplot(121)
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.subplot(122)
        plt.imshow(imresize(D, (DPixels, DPixels)), interpolation = 'nearest', cmap = 'afmhot')
        plt.show()
    if not (D.shape[0] == DPixels):
        return (D, imresize(D, (DPixels, DPixels)))
    return (D, D)

def getSSMAltMetric(X, A, DPixels, doPlot = False):
    """
    Compute a self-similarity matrix under an alternative metric specified
    by the symmetric positive definite matrix A^TA, so that the squared
    Euclidean distance under this metric between two vectors x and y is
    (x-y)^T*A^T*A*(x-y)
    :param X: An Nxd matrix holding the d coordinates of N points
    :param DPixels: The image will be resized to this dimensions
    :param doPlot: If true, show a plot comparing the original/resized images
    :return: A tuple (D, DResized)
    """
    X2 = X.dot(A.T)
    return getSSM(X2, DPixels, doPlot)

#############################################################################
## Code for dealing with cross-similarity matrices
#############################################################################

def getCSM(X, Y):
    """
    Return the Euclidean cross-similarity matrix between the M points
    in the Mxd matrix X and the N points in the Nxd matrix Y.
    :param X: An Mxd matrix holding the coordinates of M points
    :param Y: An Nxd matrix holding the coordinates of N points
    :return D: An MxN Euclidean cross-similarity matrix
    """
    C = np.sum(X**2, 1)[:, None] + np.sum(Y**2, 1)[None, :] - 2*X.dot(Y.T)
    C[C < 0] = 0
    return np.sqrt(C)

def getCSMEMD1D(X, Y):
    """
    Compute an approximate of the earth mover's distance
    between the M points in the Mxd matrix X and the N points
    in the Nxd matrix Y
    :param X: Mxd matrix
    :param Y: Nxd matrix
    :return D: An MxN distance matrix
    """
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
    """
    Return the cosine distance between all vectors in X
    and all vectors in Y
    :param X: Mxd matrix
    :param Y: Nxd matrix
    :return D: An MxN distance matrix
    """
    XNorm = np.sqrt(np.sum(X**2, 1))
    XNorm[XNorm == 0] = 1
    YNorm = np.sqrt(np.sum(Y**2, 1))
    YNorm[YNorm == 0] = 1
    D = (X/XNorm[:, None]).dot((Y/YNorm[:, None]).T)
    D = 1 - D #Make sure distance 0 is the same and distance 2 is the most different
    return D

def getOTI(C1, C2, doPlot = False):
    """
    Get the optimial transposition of the first chroma vector
    with respect to the second one
    :param C1: Chroma vector 1
    :param C2: Chroma vector 2
    :param doPlot: Plot the agreements over all shifts
    :returns: An index by which to rotate the first chroma vector
    to match with the second
    """
    NChroma = len(C1)
    shiftScores = np.zeros(NChroma)
    for i in range(NChroma):
        shiftScores[i] = np.sum(np.roll(C1, i)*C2)
    if doPlot:
        plt.plot(shiftScores)
        plt.title("OTI")
        plt.show()
    return np.argmax(shiftScores)

def getCSMCosineOTI(X, Y, C1, C2):
    """
    Get the cosine distance between each row of X
    and each row of Y after doing a global optimal
    transposition change from X to Y
    :param X: Mxd matrix
    :param Y: Nxd matrix
    :param C1: Global chroma vector 1
    :param C2: Global chroma vector 2
    :return D: An MxN distance matrix
    """
    NChromaBins = len(C1)
    ChromasPerBlock = int(X.shape[1]/NChromaBins)
    oti = getOTI(C1, C2)
    X1 = np.reshape(X, (X.shape[0], ChromasPerBlock, NChromaBins))
    X1 = np.roll(X1, oti, axis=2)
    X1 = np.reshape(X1, [X.shape[0], ChromasPerBlock*NChromaBins])
    return getCSMCosine(X1, Y)

def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix, using partitions instead of
    nearest neighbors for speed
    :param D: M x N cross-similarity matrix
    :param Kappa:
        If Kappa = 0, take all neighbors
        If Kappa < 1 it is the fraction of mutual neighbors to consider
        Otherwise Kappa is the number of mutual neighbors to consider
    :returns B: MxN binary cross-similarity matrix
    """
    N = D.shape[0]
    M = D.shape[1]
    if Kappa == 0:
        return np.ones((N, M))
    elif Kappa < 1:
        NNeighbs = int(np.round(Kappa*M))
    else:
        NNeighbs = Kappa
    J = np.argpartition(D, NNeighbs, 1)[:, 0:NNeighbs]
    I = np.tile(np.arange(N)[:, None], (1, NNeighbs))
    V = np.ones(I.size)
    [I, J] = [I.flatten(), J.flatten()]
    ret = sparse.coo_matrix((V, (I, J)), shape=(N, M))
    return ret.toarray()

def CSMToBinaryMutual(D, Kappa):
    """
    Take the binary AND between the nearest neighbors in one
    direction and the other
    :param D: MxN cross-similarity matrix
    :param Kappa: (as in CSMToBinary)
    :returns B: MxN mutual binary cross-similarity matrix
    """
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1*B2

def getCSMType(Features1, O1, Features2, O2, Type):
    """
    A wrapper around all of the cross-similarity functions
    which automatically determines which one to use based
    on the type passed in
    """
    if Type == "Euclidean":
        return getCSM(Features1, Features2)
    elif Type == "Cosine":
        return getCSMCosine(Features1, Features2)
    elif Type == "CosineOTI":
        return getCSMCosineOTI(Features1, Features2, O1['ChromaMean'], O2['ChromaMean'])
    elif Type == "EMD1D":
        return getCSMEMD1D(Features1, Features2)
    print("Error: Unknown CSM type ", Type)
    return None


######################################################
##      Ordinary CSM and Smith Waterman Tests       ##
######################################################

def getCSMSmithWatermanScores(Features1, O1, Features2, O2, Kappa, Type, doPlot = False):
    """
    Compute the Smith Waterman score between two songs
    using a single feature set
    :param Features1: Mxk matrix of features in song 1
    :param O1: Auxiliary info for song 1
    :param Features2: Nxk matrix of features in song 2
    :param O2: Auxiliary info for song 2
    :param Kappa: Nearest neighbors param for CSM
    :param Type: Type of CSM to use
    :param doPlot: If True, plot the results of Smith waterman
    :returns: Score if doPlot = False, or dictionary of
        {'score', 'DBinary', 'D', 'maxD', 'CSM'}
        if doPlot is True
    """
    CSM = getCSMType(Features1, O1, Features2, O2, Type)
    DBinary = CSMToBinaryMutual(CSM, Kappa)
    if doPlot:
        (maxD, D) = SA.swalignimpconstrained(DBinary)
        plt.subplot(131)
        plt.imshow(CSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.title('CSM')
        plt.subplot(132)
        plt.imshow(1-DBinary, interpolation = 'nearest', cmap = 'gray')
        plt.title("CSM Binary, $\kappa$=%g"%Kappa)
        plt.subplot(133)
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Smith Waterman Score = %g"%maxD)
        return {'score':maxD, 'DBinary':DBinary, 'D':D, 'maxD':maxD, 'CSM':CSM}
    return SAC.swalignimpconstrained(DBinary)

######################################################
##        Early OR Merge Smith Waterman Tests       ##
######################################################
def getCSMSmithWatermanScoresORMerge(AllFeatures1, O1, AllFeatures2, O2, Kappa, CSMTypes, doPlot = False):
    """
    Compute the Smith Waterman score between two songs
    after doing a binary OR on individual feature sets
    :param AllFeatures1: A dictionary of Mxk matric of
        features in song 1
    :param O1: Auxiliary info for song 1
    :param AllFeatures2: A dictionary of Nxk matrix of
        features in song 2
    :param O2: Auxiliary info for song 2
    :param Kappa: Nearest neighbors param for CSM
    :param CSMTypes: Dictionary of types of CSMs for each
        feature
    :param doPlot: If True, plot the results of the fusion
        and of Smith Waterman
    :returns: Score if doPlot = False, or dictionary of
        {'score', 'DBinary', 'D', 'maxD'}
        if doPlot is True
    """
    CSMs = []
    DsBinary = []
    Features = list(AllFeatures1)
    #Compute all CSMs
    for i in range(len(Features)):
        F = Features[i]
        CSMs.append(getCSMType(AllFeatures1[F], O1, AllFeatures2[F], O2, CSMTypes[F]))
        DsBinary.append(CSMToBinaryMutual(CSMs[i], Kappa))
    #Do an OR merge
    DBinary = np.zeros(DsBinary[0].shape)
    for D in DsBinary:
        DBinary += D
    DBinary[DBinary > 0] = 1
    if doPlot:
        #TODO: I have no idea why I'm seeing a large gap
        (maxD, D) = SA.swalignimpconstrained(DBinary)
        N = len(CSMs)
        for i in range(N):
            print("plt.subplot(2, %i, %i)"%(N+1, i+1))
            plt.subplot(2, N+1, i+1)
            plt.imshow(CSMs[i], interpolation = 'nearest', cmap = 'afmhot')
            plt.title('CSM %s'%Features[i])
            plt.subplot(2, N+1, N+2+i)
            plt.imshow(1-DsBinary[i], interpolation = 'nearest', cmap = 'gray')
            plt.title("CSM Binary %s K=%g"%(Features[i], Kappa))
        plt.subplot(2, N+1, 2*N+2)
        plt.imshow(DBinary, interpolation = 'nearest', cmap = 'afmhot')
        plt.title('CSM Binary OR Merged')
        plt.subplot(2, N+1, N+1)
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Smith Waterman Score = %g"%maxD)
        return {'score':maxD, 'DBinary':DBinary, 'D':D, 'maxD':maxD}
    return SAC.swalignimpconstrained(DBinary)

######################################################
##          Early Fusion Smith Waterman Tests       ##
######################################################
def getCSMSmithWatermanScoresEarlyFusionFull(AllFeatures1, O1, AllFeatures2, O2, Kappa, K, NIters, CSMTypes, doPlot = False, conservative = False):
    """
    Compute the Smith Waterman score between two songs
    after doing early similarity network fusion on
    individual feature sets
    :param AllFeatures1: A dictionary of Mxk matric of
        features in song 1
    :param O1: Auxiliary info for song 1
    :param AllFeatures2: A dictionary of Nxk matrix of
        features in song 2
    :param O2: Auxiliary info for song 2
    :param Kappa: Nearest neighbors param for CSM
    :param CSMTypes: Dictionary of types of CSMs for each
        feature
    :param doPlot: If True, plot the results of the fusion
        and of Smith Waterman
    :param conservative: Whether to use a percentage of the
        closest distances instead of mutual nearest neighbors
        (False by default, but useful for audio synchronization)
    :returns:
        if doPlot = False
            {'score', 'CSM', 'DBinary', 'OtherCSMs'}
        if doPlot = True
            {'score', 'CSM', 'DBinary', 'D', 'maxD', 'path'}
    """
    CSMs = [] #Individual CSMs
    Ws = [] #W built from fused CSMs/SSMs
    Features = list(AllFeatures1)
    OtherCSMs = {}
    #Compute all CSMs and SSMs
    for i in range(len(Features)):
        F = Features[i]
        SSMA = getCSMType(AllFeatures1[F], O1, AllFeatures1[F], O1, CSMTypes[F])
        SSMB = getCSMType(AllFeatures2[F], O2, AllFeatures2[F], O2, CSMTypes[F])
        CSMAB = getCSMType(AllFeatures1[F], O1, AllFeatures2[F], O2, CSMTypes[F])
        CSMs.append(CSMAB)
        OtherCSMs[F] = CSMAB
        #Build W from CSM and SSMs
        Ws.append(getWCSMSSM(SSMA, SSMB, CSMAB, K))
    tic = time.time()
    D = doSimilarityFusionWs(Ws, K, NIters, 1)
    toc = time.time()
    t1 = toc - tic
    N = AllFeatures1[Features[0]].shape[0]
    CSM = D[0:N, N::] + D[N::, 0:N].T
    #sio.savemat("CSM.mat", {"CSM":CSM})
    #Note that the CSM is in probabalistic weight form, so the
    #"nearest neighbors" are actually those with highest weight.  So
    #apply monotonic exp(-CSM) to fix this

    if conservative:
        x = CSM.flatten()
        x = x[np.argsort(-x)]
        cutoff = x[int(3*np.sqrt(CSM.size))]
        DBinary = np.array(CSM)
        DBinary[CSM < cutoff] = 0
        DBinary[DBinary > 0] = 1
    else:
        DBinary = CSMToBinaryMutual(np.exp(-CSM), Kappa)

    if doPlot:
        print("Elapsed Time Similarity Fusion: %g"%t1)
        N = len(CSMs)
        for i in range(N):
            plt.subplot(3, N+1, i+1)
            plt.imshow(CSMs[i], interpolation = 'nearest', cmap = 'afmhot')
            plt.title('CSM %s'%Features[i])
            plt.subplot(3, N+1, N+2+i)
            thisDBinary = CSMToBinaryMutual(CSMs[i], Kappa)
            plt.imshow(1-thisDBinary, interpolation = 'nearest', cmap = 'gray')
            plt.title("CSM Binary %s K=%g"%(Features[i], Kappa))
            (maxD, D) = SA.swalignimpconstrained(thisDBinary)
            plt.subplot(3, N+1, 2*N+3+i)
            plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
            plt.title("Score = %g"%maxD)
        plt.subplot(3, N+1, N+1)
        plt.imshow(CSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("CSM W Fused")
        plt.subplot(3, N+1, 2*N+2)
        plt.imshow(1-DBinary, interpolation = 'nearest', cmap = 'gray')
        plt.title('CSM Binary W Fused')
        plt.subplot(3, N+1, 3*N+3)
        (maxD, D, path) = SA.SWBacktrace(DBinary)
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("Fused Score = %g"%maxD)
        return {'score':maxD, 'CSM':CSM, 'DBinary':DBinary, 'D':D, 'maxD':maxD, 'path':path}
    return {'score':SAC.swalignimpconstrained(DBinary), 'CSM':CSM, 'DBinary':DBinary, 'OtherCSMs':OtherCSMs}

def getCSMSmithWatermanScoresEarlyFusion(AllFeatures1, O1, AllFeatures2, O2, Kappa, K, NIters, CSMTypes, doPlot = False):
    """
    Just return the score from getCSMSmithWatermanScoresEarlyFusionFull,
    using the same parameters
    """
    return getCSMSmithWatermanScoresEarlyFusionFull(AllFeatures1, O1, AllFeatures2, O2, Kappa, K, NIters, CSMTypes, doPlot)['score']
