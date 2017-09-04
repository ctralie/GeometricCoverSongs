import numpy as np
import matplotlib.pyplot as plt
import SequenceAlignment.SequenceAlignment as SA
import SequenceAlignment._SequenceAlignment as SAC
from SimilarityFusion import *
import scipy.misc
import time
from multiprocessing import Pool as PPool

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
        plt.imshow(scipy.misc.imresize(D, (DPixels, DPixels)), interpolation = 'nearest', cmap = 'afmhot')
        plt.show()
    if not (D.shape[0] == DPixels):
        return (D, scipy.misc.imresize(D, (DPixels, DPixels)))
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

def getOTI(C1, C2, doPlot = False):
    """
    Get the optimial transposition of the first chroma vector
    with respet to the second one
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
    NChromaBins = len(C1)
    ChromasPerBlock = X.shape[1]/NChromaBins
    oti = getOTI(C1, C2)
    #print "oti = ", oti
    X1 = np.reshape(X, (X.shape[0], ChromasPerBlock, NChromaBins))
    X1 = np.roll(X1, oti, axis=2)
    X1 = np.reshape(X1, [X.shape[0], ChromasPerBlock*NChromaBins])
    return getCSMCosine(X1, Y)

def CSMToBinary(D, Kappa):
    """
    Turn a cross-similarity matrix into a binary cross-simlarity matrix
    If Kappa = 0, take all neighbors
    If Kappa < 1 it is the fraction of mutual neighbors to consider
    Otherwise Kappa is the number of mutual neighbors to consider
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
    Take the binary AND between the nearest neighbors in one direction
    and the other
    """
    B1 = CSMToBinary(D, Kappa)
    B2 = CSMToBinary(D.T, Kappa).T
    return B1*B2

def getCSMType(Features1, O1, Features2, O2, Type):
    if Type == "Euclidean":
        return getCSM(Features1, Features2)
    elif Type == "Cosine":
        return getCSMCosine(Features1, Features2)
    elif Type == "CosineOTI":
        return getCSMCosineOTI(Features1, Features2, O1['ChromaMean'], O2['ChromaMean'])
    elif Type == "EMD1D":
        return getCSMEMD1D(Features1, Features2)
    print "Error: Unknown CSM type ", Type
    return None


######################################################
##      Ordinary CSM and Smith Waterman Tests       ##
######################################################

def getCSMSmithWatermanScores(args, doPlot = False):
    """
    Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
    computing of all of the smith waterman scores for a pair of songs.
    Features1 and Features2 are Mxk and Nxk matrices of features, respectively
    The type of cross-similarity can also be specified
    """
    [Features1, O1, Features2, O2, Kappa, Type] = args
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

def getScores(Features, OtherFeatures, Kappa, CSMType):
    """
    Features: An array of arrays of features at different tempo levels:
    [[Tempolevel1Features1, Tempolevel1Feature2, ...],
    [TempoLevel2Features1, TempoLevel2Features2]]
    Returns: NxN array of scores, and corresponding NxNx2 array
    of the best tempo indices
    """
    NTempos = len(Features)
    parpool = PPool(processes = 8)
    N = len(Features[0])
    Scores = np.zeros((N, N))
    BestTempos = np.zeros((N, N, 2), dtype=np.int32)
    for ti in range(NTempos):
        for i in range(N):
            print("Comparing song %i of %i tempo level %i"%(i, N, ti))
            for tj in range(NTempos):
                Z = zip([Features[ti][i]]*N, [OtherFeatures[ti][i]]*N, Features[tj], OtherFeatures[tj], [Kappa]*N, [CSMType]*N)
                s = np.zeros((2, Scores.shape[1]))
                s[0, :] = Scores[i, :]
                s[1, :] = parpool.map(getCSMSmithWatermanScores, Z)
                Scores[i, :] = np.max(s, 0)
                #Update which tempo combinations were the best
                BestTempos[i, Scores[i, :] == s[0, :], :] = [ti, tj]
    return (Scores, BestTempos)


######################################################
##        Early OR Merge Smith Waterman Tests       ##
######################################################
def getCSMSmithWatermanScoresORMerge(args, doPlot = False):
    [AllFeatures1, O1, AllFeatures2, O2, Kappa, CSMTypes] = args
    CSMs = []
    DsBinary = []
    Features = AllFeatures1.keys()
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


def getScoresEarlyORMerge(AllFeatures, OtherFeatures, Kappa, CSMTypes):
    NTempos = len(AllFeatures)
    parpool = PPool(processes = 8)
    N = len(AllFeatures[0])
    Scores = np.zeros((N, N))
    BestTempos = np.zeros((N, N, 2), dtype=np.int32)
    for ti in range(NTempos):
        for i in range(N):
            tic = time.time()
            print("Comparing song %i of %i tempo level %i"%(i, N, ti))
            for tj in range(NTempos):
                Z = zip([AllFeatures[ti][i]]*N, [OtherFeatures[ti][i]]*N, AllFeatures[tj], OtherFeatures[tj], [Kappa]*N, [CSMTypes]*N)
                s = np.zeros((2, Scores.shape[1]))
                s[0, :] = Scores[i, :]
                s[1, :] = parpool.map(getCSMSmithWatermanScoresORMerge, Z)
                Scores[i, :] = np.max(s, 0)
                #Update which tempo combinations were the best
                BestTempos[i, Scores[i, :] == s[0, :], :] = [ti, tj]
            toc = time.time()
            print "Elapsed time: ", toc-tic
    return (Scores, BestTempos)


######################################################
##          Early Fusion Smith Waterman Tests       ##
######################################################
def getCSMSmithWatermanScoresEarlyFusionFull(args, doPlot = False, conservative = False):
    [AllFeatures1, O1, AllFeatures2, O2, Kappa, K, NIters, CSMTypes] = args
    CSMs = [] #Individual CSMs
    Ws = [] #W built from fused CSMs/SSMs
    Features = AllFeatures1.keys()
    #Compute all CSMs and SSMs
    for i in range(len(Features)):
        F = Features[i]
        SSMA = getCSMType(AllFeatures1[F], O1, AllFeatures1[F], O1, CSMTypes[F])
        SSMB = getCSMType(AllFeatures2[F], O2, AllFeatures2[F], O2, CSMTypes[F])
        CSMAB = getCSMType(AllFeatures1[F], O1, AllFeatures2[F], O2, CSMTypes[F])
        CSMs.append(CSMAB)
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
        print "Elapsed Time Similarity Fusion: ", t1
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
    return {'score':SAC.swalignimpconstrained(DBinary), 'CSM':CSM, 'DBinary':DBinary}

def getCSMSmithWatermanScoresEarlyFusion(args, doPlot = False):
    return getCSMSmithWatermanScoresEarlyFusionFull(args, doPlot)['score']


def getScoresEarlyFusion(AllFeatures, OtherFeatures, Kappa, K, NIters, CSMTypes):
    NTempos = len(AllFeatures)
    parpool = PPool(processes = 8)
    N = len(AllFeatures[0])
    Scores = np.zeros((N, N))
    BestTempos = np.zeros((N, N, 2), dtype=np.int32)
    for ti in range(NTempos):
        for i in range(N):
            print("Comparing song %i of %i tempo level %i"%(i, N, ti))
            tic = time.time()
            for tj in range(NTempos):
                Z = zip([AllFeatures[ti][i]]*N, [OtherFeatures[ti][i]]*N, AllFeatures[tj], OtherFeatures[tj], [Kappa]*N, [K]*N, [NIters]*N, [CSMTypes]*N)
                s = np.zeros((2, Scores.shape[1]))
                s[0, :] = Scores[i, :]
                s[1, :] = parpool.map(getCSMSmithWatermanScoresEarlyFusion, Z)
                Scores[i, :] = np.max(s, 0)
                #Update which tempo combinations were the best
                BestTempos[i, Scores[i, :] == s[0, :], :] = [ti, tj]
            toc = time.time()
            print "Elapsed time: ", toc-tic
    return (Scores, BestTempos)
