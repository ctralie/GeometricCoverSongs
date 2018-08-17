"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
import scipy.ndimage
from scipy import sparse 
import matplotlib.pyplot as plt
import scipy.io as sio
import librosa
import argparse
from CSMSSMTools import getCSM, getCSMCosine, CSMToBinaryMutual
from SimilarityFusion import doSimilarityFusion
from HKS import *
from pyMIRBasic.Chroma import *
from pyMIRBasic.MFCC import *
from pyMIRBasic.AudioIO import *
import os
import json
import subprocess
from sklearn.decomposition import PCA
import time
from ripser import ripser, plot_dgms
from DGMTools import *
from Scattering import *

"""
TODO: Try SNF with different window lengths to better capture multiresolution structure
"""

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

class PrettyFloat(float):
    def __repr__(self):
        return '%.4g' % self
def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

def getDiffusionMap(W, t = -1, includeDiag = True, thresh = 5e-4, NEigs = 51):
    """
    :param W: A similarity matrix
    :param t: Diffusion parameter.  If -1, do Autotuning
    :param includeDiag: If true, include recurrence to diagonal in the markov
        chain.  If false, zero out diagonal
    :param thresh: Threshold below which to zero out entries in markov chain in
        the sparse approximation
    :param NEigs: The number of eigenvectors to use in the approximation
    """
    if not includeDiag:
        np.fill_diagonal(W, np.zeros(W.shape[0]))
    RowSumSqrt = np.sqrt(np.sum(W, 1))
    DInvSqrt = sparse.diags([1/RowSumSqrt], [0])

    #Symmetric normalized
    Pp = (W/RowSumSqrt[None, :])/RowSumSqrt[:, None]
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

def exportToLoopDitty(XAudio, Fs, hopSize, winFac, pX, outfilename):
    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
    sio.wavfile.write("temp.wav", Fs, XAudio)
    if os.path.exists("temp.mp3"):
        os.remove("temp.mp3")
    subprocess.call(["avconv", "-i", "temp.wav", "temp.mp3"])

    #Load in mp3 as base64, then cleanup audio files
    fin = open("temp.mp3", "rb")
    mp3data = fin.read()
    mp3data = mp3data.encode("base64")
    fin.close()

    X = np.zeros((pX.shape[0], 4))
    X[:, 0:3] = pX
    interval = hopSize*winFac/float(Fs)
    X[:, -1] = interval*np.arange(X.shape[0])

    #Send features as JSON
    res = {'hopSize':hopSize, 'Fs':Fs, 'X':pretty_floats(X.tolist()), 'mp3data':mp3data}
    fout = open(outfilename, "w")
    fout.write(json.dumps(res))
    fout.close()

def getFusedSimilarity(XMFCC, XChroma, winFac, winsPerBlock, K):
    #Compute features in intervals evenly spaced by the hop size
    #but average within "winFac" intervals of hopSize
    N = min(XMFCC.shape[1], XChroma.shape[1])
    XMFCC = XMFCC[:, 0:N]
    XChroma = XChroma[:, 0:N]
    nHops = N-winFac+1
    intervals = np.arange(0, nHops, winFac)
    intervals = librosa.util.fix_frames(intervals, x_min=0, x_max=N)
    chroma = librosa.util.sync(XChroma, intervals)
    mfcc = librosa.util.sync(XMFCC, intervals)

    n_frames = min(chroma.shape[1], mfcc.shape[1])
    chroma = chroma[:, :n_frames]
    mfcc = mfcc[:, :n_frames]

    #Do a delay embedding and compute SSMs
    XChroma = librosa.feature.stack_memory(chroma, n_steps=winsPerBlock, mode='edge').T
    XMFCC = librosa.feature.stack_memory(mfcc, n_steps=winsPerBlock, mode='edge').T
    DChroma = getCSMCosine(XChroma, XChroma) #Cosine distance (scaled so 0 is most similar)
    DMFCC = getCSM(XMFCC, XMFCC) #Euclidean distance

    #Run similarity network fusion
    Ds = [DMFCC, DChroma]
    WFused = doSimilarityFusion(Ds, K = K, NIters = 10, regDiag = 1, regNeighbs=0.5)
    return WFused


def getFusedSimilarityAudio(XAudio, Fs, hopSize, winFac, winsPerBlock, K):
    """
    Come up with a representation of recurrence based on similarity network fusion (SNF) 
    of averaged/stack delayed Chromas and MFCCs

    Parameters
    ----------
    XAudio: ndarray (NSamples, 1)
        An array of mono audio samples
    Fs: int
        Sample rate
    hopSize: int
        Hop size
    winFac: int
        Number of frames to average (i.e. factor by which to downsample)
    winsPerBlock: int 
        Number of aggregated windows per sliding window block
    K: int
        Number of nearest neighbors in SNF
    """
    XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = 12)
    XMFCC = getMFCCsLibrosa(XAudio, Fs, int(Fs/4), hopSize, lifterexp = 0.6, NMFCC = 20)
    return getFusedSimilarity(XMFCC, XChroma, winFac, winsPerBlock, K)


def promoteDiagonal(W, bias):
    """
    Make things off diagonal less similar
    """
    N = W.shape[0]
    I, J = np.meshgrid(np.arange(N), np.arange(N))
    weight = bias + (1.0-bias)*(1.0 - np.abs(I-J)/float(N))
    weight = weight**4
    return weight*W

def getCovers1000DGms(winFac, winsPerBlock, K, bias):
    from Covers1000 import getSongPrefixes
    AllSongs = getSongPrefixes()
    plt.figure(figsize=(12, 4))
    for i, filePrefix in enumerate(AllSongs):
        matfilename = "%s_DGMs_Raw.mat"%filePrefix
        if os.path.exists(matfilename):
            print("Skipping %i"%i)
            continue
        tic = time.time()
        print("Computing features for %i of %i..."%(i, len(AllSongs)))
        print("filePrefix = %s"%filePrefix)
        X = sio.loadmat("%s_MFCC.mat"%filePrefix)
        XMFCC = X['XMFCC']
        X = sio.loadmat("%s_HPCP.mat"%filePrefix)
        XChroma = X['XHPCP']
        W = getFusedSimilarity(XMFCC, XChroma, winFac, winsPerBlock, K)
        #W = promoteDiagonal(W, bias)
        np.fill_diagonal(W, 0)
        IRips = ripser(-W, distance_matrix=True, maxdim=1)['dgms'][1]
        [X, Y] = np.meshgrid(np.arange(W.shape[0]), np.arange(W.shape[1]))
        W[X < Y] = 0
        IMorse = doImageSublevelsetFiltration(-W)
        toc = time.time()
        print("Elapsed Time: %.3g"%(toc-tic))
        sio.savemat(matfilename, {"IRips":IRips, "IMorse":IMorse})
        
        plt.clf()
        plt.subplot(131)
        plt.imshow(np.log(W+5e-2), cmap = 'afmhot')
        plt.subplot(132)
        plt.scatter(IRips[:, 0], IRips[:, 1])
        plt.title("Rips (%i points)"%(IRips.shape[0]))
        plt.subplot(133)
        plt.scatter(IMorse[:, 0], IMorse[:, 1])
        plt.title("Superlevelset Filtration (%i points)"%IMorse.shape[0])
        plt.savefig("%s_DGMS_Raw.png"%filePrefix, bbox_inches='tight')

def compareCovers1000Dgms():
    from Covers1000 import getSongPrefixes
    AllSongs = getSongPrefixes()
    AllPIs = []
    persThresh = 0.02
    for i, filePrefix in enumerate(AllSongs):
        print("Getting persistence image %i of %i"%(i, len(AllSongs)))
        matfilename = "%s_DGMs.mat"%filePrefix
        res = sio.loadmat(matfilename)
        IRips, IMorse = res['IRips'], res['IMorse']
        I = IRips
        if I.size > 0:
            I = I[np.abs(I[:, 0]-I[:, 1])>persThresh, :]
        PI = getPersistenceImage(I, [-1.5, 0, 0, 1.5], 0.05, psigma=0.1)['PI']
        """
        plt.subplot(121)
        plot_dgms(I, lifetime=True)
        plt.subplot(122)
        plt.imshow(PI, cmap='afmhot')
        plt.show()
        """
        AllPIs.append(PI.flatten())
    AllPIs = np.array(AllPIs)
    D = getCSM(AllPIs, AllPIs)
    sio.savemat("Covers1000PIs.mat", {"D":D})


def getCovers1000Scattering(winFac, winsPerBlock, K, bias):
    from Covers1000 import getSongPrefixes
    AllSongs = getSongPrefixes()
    Ds = []
    res = 512
    # Step 1: Compute all resized similarity images
    for i, filePrefix in enumerate(AllSongs):
        print("Computing features for %i of %i..."%(i, len(AllSongs)))
        print("filePrefix = %s"%filePrefix)
        X = sio.loadmat("%s_MFCC.mat"%filePrefix)
        XMFCC = X['XMFCC']
        X = sio.loadmat("%s_HPCP.mat"%filePrefix)
        XChroma = X['XHPCP']
        W = getFusedSimilarity(XMFCC, XChroma, winFac, winsPerBlock, K)
        W[np.isnan(W)] = 1
        # Fill first two diagonals with zeros
        pix = np.arange(W.shape[0])
        I, J = np.meshgrid(pix, pix)
        W[np.abs(I - J) <= 1] = 0
        Ds.append(imresize(W, (res, res)))
    # Step 2: Compute scattering transforms
    # Do in batches of 10
    AllScattering = []
    for i in range(len(AllSongs)/10):
        AllScattering += getScatteringTransform(Ds[i*10:(i+1)*10], renorm=False)
    EuclideanFeats = np.array([])
    ScatteringFeats = np.array([])
    ScatteringFeatsPooled = np.array([])
    plt.figure(figsize=(15, 10))
    for i, (filePrefix, images) in enumerate(zip(AllSongs, AllScattering)):
        if EuclideanFeats.size == 0:
            EuclideanFeats = np.zeros((len(AllSongs), Ds[i].size), dtype=np.float32)
        EuclideanFeats[i, :] = Ds[i].flatten()
        print("Saving scattering transform for %s"%filePrefix)
        scattering = np.array([])
        scatteringpooled = np.array([np.mean(images[0])])
        plt.clf()
        plt.subplot(2, len(images), len(images)+1)
        plt.imshow(Ds[i], cmap = 'afmhot')
        plt.title("Original")
        for k in range(len(images)):
            norm = np.sqrt(np.sum(images[k]**2))
            if norm == 0:
                norm = 1
            images[k] /= norm
            scattering = np.concatenate((scattering, images[k].flatten()))
            plt.subplot(2, len(images), k+1)
            plt.imshow(images[k], cmap='afmhot')
            plt.title("Scattering %i"%k)
            if k > 0:
                plt.subplot(2, len(images), len(images)+1+k)
                pooled = poolFeatures(images[k], images[0].shape[0])
                plt.imshow(pooled, cmap = 'afmhot')
                scatteringpooled = np.concatenate((scatteringpooled, pooled.flatten()))
                plt.title("Scattering %i Pooled"%k)
            plt.savefig("%s_Scattering.png"%filePrefix, bbox_inches='tight')
        if ScatteringFeats.size == 0:
            ScatteringFeats = np.zeros((len(AllSongs), scattering.size), dtype=np.float32)
            ScatteringFeatsPooled = np.zeros((len(AllSongs), scatteringpooled.size), dtype=np.float32)
        scattering[np.isnan(scattering)] = 1
        scatteringpooled[np.isnan(scatteringpooled)] = 1
        ScatteringFeats[i, :] = scattering.flatten()
        ScatteringFeatsPooled[i, :] = scatteringpooled.flatten()
    sio.savemat("Covers1000Euclidean.mat", {"D":getCSM(EuclideanFeats, EuclideanFeats)})
    EuclideanFeats = None
    sio.savemat("Covers1000Scattering.mat", {"D":getCSM(ScatteringFeats, ScatteringFeats)})
    ScatteringFeats = None
    sio.savemat("Covers1000ScatteringPooled.mat", {"D":getCSM(ScatteringFeatsPooled, ScatteringFeatsPooled)})
        

def doMJExample():
    hopSize=512
    winFac = 10
    winsPerBlock = 20
    K = 20
    pooling = True

    print("Getting fused SSM for song 1...")
    XAudio1, Fs = getAudioLibrosa("CSMViewer/MJ.mp3")
    W1 = getFusedSimilarityAudio(XAudio1, Fs, hopSize, winFac, winsPerBlock, K)
    np.fill_diagonal(W1, 0)

    print("Getting fused SSM for song 2...")
    XAudio2, Fs = getAudioLibrosa("CSMViewer/MJBad.mp3")
    W2 = getFusedSimilarityAudio(XAudio2, Fs, hopSize, winFac, winsPerBlock, K)
    np.fill_diagonal(W2, 0)

    X1 = getDiffusionMap(W1)
    X2 = getDiffusionMap(W2)

    Ds = [imresize(W1, (512, 512)), imresize(W2, (512, 512))]
    images = getScatteringTransform(Ds)

    plt.figure(figsize=(12, 12))
    plt.subplot(241)
    plt.imshow(np.log(W1+5e-2), cmap = 'afmhot')
    plt.title("Song 1")
    for i in range(3):
        plt.subplot(2, 4, i+2)
        if i > 0 and pooling:
            images[0][i] = poolFeatures(images[0][i], images[0][0].shape[0])
        plt.imshow(images[0][i], cmap = 'afmhot')
    plt.subplot(245)
    plt.imshow(np.log(W2+5e-2), cmap = 'afmhot')
    for i in range(3):
        plt.subplot(2, 4, i+6)
        if i > 0 and pooling:
            images[1][i] = poolFeatures(images[1][i], images[1][0].shape[0])
        plt.imshow(images[1][i], cmap = 'afmhot')
    plt.title("Song 2")
    plt.show()

    #exportToLoopDitty(XAudio1, Fs, hopSize, winFac, X1[:, -4:-1], "MJdiffusionDiag.json")
    #exportToLoopDitty(XAudio2, Fs, hopSize, winFac, X2[:, -4:-1], "AAFdiffusionDiag.json")


if __name__ == '__main__':
    #doMJExample()
    #getCovers1000DGms(winFac=10, winsPerBlock=20, K=20, bias=0.3)
    getCovers1000Scattering(winFac=10, winsPerBlock=20, K=20, bias=0.3)
    #compareCovers1000Dgms()
