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
import librosa.display
import argparse
from CSMSSMTools import getCSM, getCSMCosine, CSMToBinaryMutual
from SimilarityFusion import doSimilarityFusion
import os
import json
import subprocess
from sklearn.decomposition import PCA
import time



def chrompwr(X, P=.5):
    """
    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[np.where(CMn==0)] = 1
    # normalize each input col, raise to power
    CMp = np.power(X/CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn==0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)

def btchroma_to_fftmat(btchroma, win=75):
    """
    Stack the flattened result of fft2 on patches 12 x win
    Translation of my own matlab function
    -> python: TBM, 2011-11-05, TESTED
    """
    # 12 semitones
    nchrm, nbeats = btchroma.shape
    assert nchrm == 12, 'beat-aligned matrix transposed?'
    if nbeats < win:
        return None
    # output
    fftmat = np.zeros((nchrm * win, nbeats - win + 1))
    for i in range(nbeats-win+1):
        F = scipy.fftpack.fft2(btchroma[:,i:i+win])
        F = np.sqrt(np.real(F)**2 + np.imag(F)**2)
        patch = scipy.fftpack.fftshift(F)
        fftmat[:, i] = patch.flatten()
    return fftmat



def getCovers1000ChromaFTM2D(do_plots = False):
    """
    Get the Fourier Magnitude Coefficients for Covers100 songs
    """
    from Covers1000 import getSongPrefixes
    PWR = 1.96
    WIN = 75
    C = 5

    AllSongs = getSongPrefixes()
    plt.figure(figsize=(12, 4))
    AllFeats = []
    for i, filePrefix in enumerate(AllSongs):
        print("Doing %s"%filePrefix)
        hpcp = sio.loadmat("%s_HPCP.mat"%filePrefix)['XHPCP']
        beats = sio.loadmat("%s_Beats.mat"%filePrefix)['beats0'].flatten()
        # Use madmom beats
        intervals = librosa.util.fix_frames(beats, x_min = 0, x_max=hpcp.shape[1])
        chroma = librosa.util.sync(hpcp, intervals)
        chroma = chrompwr(chroma, PWR)
        # Get all 2D FFT magnitude shingles
        feats = btchroma_to_fftmat(chroma, WIN).T
        Norm = np.sqrt(np.sum(feats**2, 1))
        Norm[Norm == 0] = 1
        feats = np.log(C*feats/Norm[:, None] + 1)
        feats = np.median(feats, 0) # Median aggregate
        AllFeats.append(feats.flatten())

        if do_plots:
            feats = np.reshape(feats, (12, WIN))
            plt.clf()
            plt.subplot(211)
            librosa.display.specshow(chroma, y_axis='chroma')
            plt.title("Beat-Synchronous Chromagram")
            plt.subplot(212)
            plt.imshow(feats, cmap = 'afmhot', aspect = 'auto')
            plt.title("Aggregated FFT2 Shingle")
            plt.savefig("%s_ChromaFFT2.png"%filePrefix, bbox_inches='tight')
    
    AllFeats = np.array(AllFeats)
    D = getCSM(AllFeats, AllFeats)
    sio.savemat("Covers10002DFTM.mat", {"D":D})
        


if __name__ == '__main__':
    getCovers1000ChromaFTM2D()