#Programmer: Chris Tralie
#Purpose: To replicate my experiments from ISMIR2015 in Python with librosa
#Do fine-grained timbral MFCCs
import numpy as np
import sys
import scipy.io as sio
from scipy.interpolate import interp1d
from scipy import signal
import time
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool
from CSMSSMTools import *
import librosa
import subprocess

NMFCC = 20

#Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
#computing of all of the beat-synchronous features
def getFeatures(args):
    #Unpack parameters
    (filename, BeatsPerBlock, TempoBias, FeatureParams) = args
    print "Getting features for %s..."%filename
    DPixels = FeatureParams['DPixels']
    NCurv = FeatureParams['NCurv']
    CurvDelta = FeatureParams['CurvDelta']
    NJump = FeatureParams['NJump']
    D2Samples = FeatureParams['D2Samples']
    
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
    
    #Step 3: Compute features in z-normalized blocks
    NBeats = len(beats)-1
    NPixels = DPixels*(DPixels-1)/2
    ND = NBeats - BeatsPerBlock
    SSMs = np.zeros((ND, NPixels), dtype = np.float32)
    D2s = np.zeros((ND, D2Samples), dtype = np.float32)
    Jumps = np.zeros((ND, NJump), dtype = np.float32)
    Curvs = np.zeros((ND, NCurv), dtype = np.float32)
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
        
        #Compute SSM and D2 histogram
        (DOrig, D) = getSSM(xn, DPixels)
        SSMs[i, :] = D[I < J]
        [IO, JO] = np.meshgrid(np.arange(DOrig.shape[0]), np.arange(DOrig.shape[0]))
        D2s[i, :] = np.histogram(DOrig[IO < JO], bins = D2Samples, range = (0, 2))[0]
        D2s[i, :] = D2s[i, :]/np.sum(D2s[i, :]) #Normalize
        
        #Compute jump
        jump = xn[1::, :] - xn[0:-1, :]
        jump = np.sqrt(np.sum(jump**2, 1))
        jump = np.concatenate(([0], jump))
        
        #Compute curvature approximation as ratio of geodesic to Euclidean distance
        geodesic = np.cumsum(jump)
        geodesic = geodesic[CurvDelta*2::] - geodesic[0:-CurvDelta*2]
        euclidean = xn[CurvDelta*2::, :] - xn[0:-CurvDelta*2, :]
        euclidean = np.sqrt(np.sum(euclidean**2, 1))
        geodesic[euclidean == 0] = 0
        euclidean[euclidean == 0] = 1
        #curv = geodesic/euclidean
        curv = geodesic
        
        #Resample jump and curvature
        jump = signal.resample(jump, NJump)
        Jumps[i, :] = jump
        curv = signal.resample(curv, NCurv)
        Curvs[i, :] = curv
    return {'SSMs':SSMs, 'D2s':D2s, 'Jumps':Jumps, 'Curvs':Curvs}

def getScores(Features, CSMType):
    N = len(Features[0])
    Scores = np.zeros((N, N))
    for ti in range(len(Features)):
        for i in range(N):
            print "Comparing song %i of %i tempo level %i"%(i, N, ti)
            for tj in range(len(TempoBiases)):
                Z = zip([Features[ti][i]]*N, Features[tj], [Kappa]*N, [CSMType]*N)
                s = np.zeros((2, Scores.shape[1]))
                s[0, :] = Scores[i, :]
                s[1, :] = parpool.map(getCSMSmithWatermanScores, Z)
                Scores[i, :] = np.max(s, 0)
    return Scores

def getEvalStatistics(ScoresParam, N, NSongs, topsidx):
    Scores = np.array(ScoresParam)
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
    return (MR, MRR, MDR, tops)

#############################################################################
## Code for running the experiments
#############################################################################

#Instead of looking in set 2 to compare to set 1, report mean rank,
#mean reciprocal rank, and median rank of identified track
#as well as top-01, top-10, top-25, top-50, and top-100
def runCovers80ExperimentAllSongs(BeatsPerBlock, Kappa, Params, topsidx = [1, 25, 50, 100]):
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
    D2s = []
    Jumps = []
    Curvs = []
    for i in range(len(TempoBiases)):
        Z = zip(files, [BeatsPerBlock]*N, [TempoBiases[i]]*N, [Params]*N)
        Features = parpool.map(getFeatures, Z)
        SSMs.append(Features['SSMs'])
        D2s.append(Features['D2s'])
        Jumps.append(Features['Jumps'])
        Curvs.append(Features['Curvs'])
    ScoresSSMs = getScores(SSMs, "Euclidean")
    ScoresD2s = getScores(D2s, "EMD1D")
    ScoresJumps = getScores(Jumps, "Euclidean")
    ScoresCurvs = getScores(Curvs, "Euclidean")

    sio.savemat("AllScores.mat", {"ScoresSSMs":ScoresSSMs, "ScoresD2s":ScoresD2s, "ScoresJumps":ScoresJumps, "ScoresCurvs":ScoresCurvs})

    print("Scores SSMs")
    getEvalStatistics(ScoresSSMs, N, NSongs, topsidx)
    print("Scores D2s")
    getEvalStatistics(ScoresD2s, N, NSongs, topsidx)
    print("Scores Jumps")
    getEvalStatistics(ScoresJumps, N, NSongs, topsidx)
    print("Scores Curvs")
    getEvalStatistics(ScoresCurvs, N, NSongs, topsidx)


#############################################################################
## Entry points for running the experiments
#############################################################################

if __name__ == '__main__':
    BeatsPerBlock = 20
    Kappa = 0.1
    Params = {'DPixels':50, 'NCurv':50, 'NJump':50, 'D2Samples':50, 'CurvDelta':5}
    runCovers80ExperimentAllSongs(BeatsPerBlock, Kappa, Params, topsidx = [1, 25, 50, 100])

if __name__ == '__main__2':
    BeatsPerBlock = 20
    Kappa = 0.1
    
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    
    Params = {'DPixels':50, 'NCurv':50, 'NJump':50, 'D2Samples':20, 'CurvDelta':5}
    args = (files1[75], BeatsPerBlock, 120, Params)
    print "Getting features for %s..."%files1[75] 
    Features1 = getFeatures(args)
    args = (files2[75], BeatsPerBlock, 120, Params)
    print "Getting features for %s..."%files2[75]
    Features2 = getFeatures(args)
    
    plt.figure(figsize=(16, 48))
    getCSMSmithWatermanScores([Features1['SSMs'], Features2['SSMs'], Kappa, "Euclidean"], True)
    plt.savefig("SSMs75.svg", dpi=200, bbox_inches='tight')
    
    getCSMSmithWatermanScores([Features1['D2s'], Features2['D2s'], Kappa, "Euclidean"], True)
    plt.savefig("D2Euclidean75.svg", dpi=200, bbox_inches='tight')

    getCSMSmithWatermanScores([Features1['D2s'], Features2['D2s'], Kappa, "EMD1D"], True)
    plt.savefig("D2EMD75.svg", dpi=200, bbox_inches='tight')

    getCSMSmithWatermanScores([Features1['Jumps'], Features2['Jumps'], Kappa, "Euclidean"], True)
    plt.savefig("Jumps75.svg", dpi=200, bbox_inches='tight')
    
    getCSMSmithWatermanScores([Features1['Curvs'], Features2['Curvs'], Kappa, "Euclidean"], True)
    plt.savefig("Curvs75.svg", dpi=200, bbox_inches='tight')
