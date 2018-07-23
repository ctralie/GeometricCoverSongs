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
from CSMSSMTools import getCSM, getCSMCosine
from SimilarityFusion import doSimilarityFusion
from Laplacian import *
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

def exportToLoopDitty(XAudio, Fs, hopSize, winFac, hks, outfilename):
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

    pca = PCA(n_components=3)
    X = np.zeros((hks.shape[0], 4))
    X[:, 0:3] = pca.fit_transform(hks)

    interval = hopSize*winFac/float(Fs)
    X[:, -1] = interval*np.arange(X.shape[0])

    #Send features as JSON
    res = {'hopSize':hopSize, 'Fs':Fs, 'X':pretty_floats(X.tolist()), 'mp3data':mp3data}
    fout = open(outfilename, "w")
    fout.write(json.dumps(res))
    fout.close()


def getFusedSimilarityHKS(XAudio, Fs, hopSize, winFac, winsPerBlock, K, NEigs):
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
    hks = getHKS(eigvalues, eigvectors, np.linspace(0, 2, 100), scaleinv=False)
    return {'hks':hks, 'W':WFused}

if __name__ == '__main__':
    hopSize=512
    winFac = 5
    winsPerBlock = 20
    K = 10
    NEigs = 50


    XAudio1, Fs = getAudioLibrosa("CSMViewer/MJ.mp3")
    res = getFusedSimilarityHKS(XAudio1, Fs, hopSize, winFac, winsPerBlock, K, NEigs)
    hks1 = res['hks']
    W1 = res['W']
    hks1 = np.log(hks1)

    XAudio2, Fs = getAudioLibrosa("CSMViewer/AAF.mp3")
    res = getFusedSimilarityHKS(XAudio2, Fs, hopSize, winFac, winsPerBlock, K, NEigs)
    hks2 = res['hks']
    W2 = res['W']
    hks2 = np.log(hks2)


    plt.subplot(221)
    plt.imshow(W1, cmap = 'afmhot', interpolation='none')
    plt.subplot(222)
    plt.imshow(hks1, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.subplot(223)
    plt.imshow(W2, cmap = 'afmhot', interpolation='none')
    plt.subplot(224)
    plt.imshow(hks2, cmap = 'afmhot', interpolation='none', aspect='auto')
    plt.show()
    exportToLoopDitty(XAudio1, Fs, hopSize, winFac, hks1, "MJ.json")
    exportToLoopDitty(XAudio2, Fs, hopSize, winFac, hks2, "AAF.json")