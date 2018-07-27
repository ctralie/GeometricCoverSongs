"""
Programmer: Chris Tralie
Purpose: To provide an interface for loading music, computing features, and
doing similarity fusion on those features to make a weighted adjacency matrix
"""
import numpy as np
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

"""
TODO: Try SNF with different window lengths to better capture multiresolution structure
"""

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


def getFusedSimilarityHKS(XAudio, Fs, hopSize, winFac, winsPerBlock, K, NEigs, ss):
    """
    Come up with a representation of recurrence based on similarity network fusion (SNF) 
    of averaged/stack delayed Chromas and MFCCs, and compute the HKS of this using
    the weighted Laplacian

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
    NEigs: int
        Number of eigenvectors to use in HKS
    ss: ndarray (S, 1)
        An array of spatial scales at which to sample the HKS
    """
    XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = 12)
    XMFCC = getMFCCsLibrosa(XAudio, Fs, int(Fs/4), hopSize, lifterexp = 0.6, NMFCC = 20)

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
    WFused = doSimilarityFusion(Ds, K = K, NIters = 10, regDiag = 1, regNeighbs=0.0)
    np.fill_diagonal(WFused, 0)
    (eigvalues, eigvectors, L) = getLaplacianEigsDense(WFused, NEigs)
    hks = getHKS(eigvalues, eigvectors, ss, scaleinv=False)
    return {'hks':hks, 'W':WFused, 'eigvalues':eigvalues, 'eigvectors':eigvectors}

if __name__ == '__main__':
    hopSize=512
    winFac = 5
    winsPerBlock = 20
    K = 20
    NEigs = 20

    # Come up with log-sampled spatial scale limits
    lims = [0.1, 2]
    T = 200
    alpha = float(lims[1]/lims[0])**(1.0/T)
    ss = lims[0]*alpha**np.arange(T)
    # Come up with log-sampled time scale limits
    ts = np.linspace(0, 1, 100)
    ss = np.linspace(0, 10, T)

    print("Getting HKS for song 1...")
    XAudio1, Fs = getAudioLibrosa("CSMViewer/MJ.mp3")
    res = getFusedSimilarityHKS(XAudio1, Fs, hopSize, winFac, winsPerBlock, K, NEigs, ss)
    hks1 = res['hks']
    W1 = res['W']
    hks1 = np.log(hks1)
    #hks1 = np.abs(np.fft.fft(hks1, axis=1))[:, 1::]


    print("Getting HKS for song 2...")
    XAudio2, Fs = getAudioLibrosa("CSMViewer/AAF.mp3")
    res = getFusedSimilarityHKS(XAudio2, Fs, hopSize, winFac, winsPerBlock, K, NEigs, ss)
    hks2 = res['hks']
    W2 = res['W']
    hks2 = np.log(hks2)
    #hks2 = np.abs(np.fft.fft(hks2, axis=1))[:, 1::]

    sio.savemat('hks.mat', {'hks1':hks1, 'hks2':hks2, 'W1':W1, 'W2':W2})





    W = W2
    N = W.shape[0]
    tdiff = np.linspace(0, 1, N)
    I, J = np.meshgrid(tdiff, tdiff)
    tdiffs = np.abs(I-J)
    S = ss.size
    T = ts.size
    plt.figure(figsize=(12, 12))
    idxs = [400, 500, 600, 1240]
    for i, t in enumerate(ts):
        thisW = W*(tdiffs < t)
        (eigvalues, eigvectors, L) = getLaplacianEigsDense(thisW, NEigs)
        hks = getHKS(eigvalues, eigvectors, ss, scaleinv=False)
        hks = np.log(hks)
        plt.clf()
        plt.subplot(221)
        plt.imshow(thisW, cmap = 'afmhot')
        plt.title("W")
        for idx in idxs:
            plt.scatter(idx, idx, 50)
        plt.subplot(222)
        plt.imshow(hks, aspect='auto')
        plt.title("HKS")
        plt.subplot(223)
        for idx in idxs:
            plt.plot(hks[idx, :])
        plt.subplot(224)
        plt.plot(eigvalues)
        plt.title("Eigenvalues")
        plt.savefig("%i.png"%i, bbox_inches='tight')


    print("Doing Smith Waterman...")
    Kappa = 0.1
    CSM = getCSM(hks1, hks2)
    DBinary = CSMToBinaryMutual(CSM, Kappa)
    import SequenceAlignment.SequenceAlignment as SA
    #(maxD, D) = SA.swalignimpconstrained(DBinary)
    plt.subplot(131)
    plt.imshow(CSM, interpolation = 'nearest', cmap = 'afmhot')
    plt.title('CSM')
    plt.subplot(132)
    plt.imshow(1-DBinary, interpolation = 'nearest', cmap = 'gray')
    plt.title("CSM Binary, $\kappa$=%g"%Kappa)
    plt.subplot(133)
    #plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
    #plt.title("Smith Waterman Score = %g"%maxD)
    plt.show()

    plt.subplot(221)
    plt.imshow(W1, cmap = 'afmhot', interpolation='none')
    plt.subplot(222)
    plt.imshow(hks1, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.subplot(223)
    plt.imshow(W2, cmap = 'afmhot', interpolation='none')
    plt.subplot(224)
    plt.imshow(hks2, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.show()

    pca = PCA(n_components=3)
    exportToLoopDitty(XAudio1, Fs, hopSize, winFac, pca.fit_transform(hks1), "MJhks.json")
    exportToLoopDitty(XAudio2, Fs, hopSize, winFac, pca.fit_transform(hks2), "AAFhks.json")

    X1 = getDiffusionMap(W1)
    X2 = getDiffusionMap(W2)
    plt.subplot(131)
    plt.imshow(X1, cmap = 'afmhot', aspect='auto')
    plt.subplot(132)
    plt.imshow(X2, cmap = 'afmhot', aspect='auto')
    plt.subplot(133)
    plt.imshow(getCSM(X1[:, 0:-1], X2[:, 0:-1]), interpolation = 'nearest', cmap = 'afmhot')
    plt.show()

    exportToLoopDitty(XAudio1, Fs, hopSize, winFac, X1[:, -4:-1], "MJdiffusion.json")
    exportToLoopDitty(XAudio2, Fs, hopSize, winFac, X2[:, -4:-1], "AAFdiffusion.json")