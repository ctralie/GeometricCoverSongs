#Programmer: Chris Tralie
#Purpose: To replicate my experiments from ISMIR2015 in Python with librosa
#Do fine-grained timbral MFCCs
import numpy as np
import sys
import scipy.io as sio
from scipy.interpolate import interp1d
import time
import cv2
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool
from CSMSSMTools import *
import librosa
import subprocess

sys.path.append('SequenceAlignment')
import _SequenceAlignment
import SequenceAlignment
NMFCC = 20

#Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
#computing of all of the beat-synchronous self-similarity matrices
def getSSMs(args):
    (filename, BeatsPerBlock, DPixels, TempoBias) = args
    [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
    hopSize = 512
    #Step 1: Load audio and extract beat onsets
    path = "covers32k/" + filename + ".ogg"
    XAudio, Fs = librosa.load(path)
    XAudio = librosa.core.to_mono(XAudio)
    (tempo, beats) = librosa.beat.beat_track(XAudio, Fs, start_bpm = TempoBias, hop_length = hopSize)# hop_length = self.hopSize)
    
    #Step 2: Compute Mel-Spaced log STFTs
    winSize = int(np.round((60.0/tempo)*Fs))
    S = librosa.core.stft(XAudio, winSize, 512)
    M = librosa.filters.mel(Fs, winSize)
    X = M.dot(np.abs(S))
    X = librosa.core.logamplitude(X)
    X = np.dot(librosa.filters.dct(NMFCC, X.shape[0]), X) #Make MFCC
    
    #Step 3: Compute SSMs in each block
    NBeats = len(beats)-1
    NPixels = DPixels*(DPixels-1)/2
    ND = NBeats - BeatsPerBlock
    Y = np.zeros((ND, NPixels), dtype = 'float32')
    for i in range(ND):
        i1 = beats[i]
        i2 = beats[i+BeatsPerBlock]
        x = X[:, i1:i2].T
        #Mean-center x
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm
        D = getSSM(xn, DPixels)
        Y[i, :] = D[I < J]
    return Y


#############################################################################
## Code for running the experiments
#############################################################################

#Instead of looking in set 2 to compare to set 1, report mean rank,
#mean reciprocal rank, and median rank of identified track
#as well as top-01, top-10, top-25, top-50, and top-100
def runCovers80ExperimentAllSongs(BeatsPerBlock, Kappa, DPixels, topsidx = [1, 25, 50, 100]):
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    NSongs = len(files1) #Should be 80
    files = files1 + files2
    N = len(files)
    
    #Set up the parallel pool
    parpool = Pool(processes = 8)    
    
    #Precompute all SSMs for all tempo biases (can be stored in memory since dimensions are small)
    TempoBiases = [60, 120, 180]
    SSMs = []
    for i in range(len(TempoBiases)):
        Z = zip(files, [BeatsPerBlock]*N, [DPixels]*N, [TempoBiases[i]]*N)
        SSMs.append(parpool.map(getSSMs, Z))
    Scores = np.zeros((N, N))
    for ti in range(len(TempoBiases)):
        for i in range(N):
            print "Comparing song %i of %i tempo level %i"%(i, N, ti)
            for tj in range(len(TempoBiases)):
                Z = zip([SSMs[ti][i]]*N, SSMs[tj], [Kappa]*N)
                s = np.zeros((2, Scores.shape[1]))
                s[0, :] = Scores[i, :]
                s[1, :] = parpool.map(getCSMSmithWatermanScores, Z)
                Scores[i, :] = np.max(s, 0)

    sio.savemat("AllScores.mat", {"Scores":Scores})

    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf) 
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    for i in range(N):
        cover = (i+NSongs)%N #The index of the correct song
        print "%i, %i"%(i, cover)
        for k in range(N):
            if idx[i, k] == cover:
                ranks[i] = k+1
                break
    print ranks
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print "MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR)
    tops = np.zeros(len(topsidx))
    for i in range(len(tops)):
        tops[i] = np.sum(ranks <= topsidx[i])
        print "Top-%i: %i"%(topsidx[i], tops[i])
    return (Scores, MR, MRR, MDR, tops)


#############################################################################
## Entry points for running the experiments
#############################################################################

if __name__ == '__main__2':
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    print files1[0]
    args = (files1[0], 20, 50, 60)
    Ds = getSSMs(args)
    sio.savemat('DsPy.mat', {'DsPy':Ds})

if __name__ == '__main__':
    BeatsPerBlock = 20
    Kappa = 0.1
    (Scores, MR, MRR, MDR, tops) = runCovers80ExperimentAllSongs(BeatsPerBlock, Kappa, 50, topsidx = [1, 25, 50, 100])
