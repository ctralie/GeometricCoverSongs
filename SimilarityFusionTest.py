import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import os
import time
import sys
sys.path.append('SequenceAlignment')
import SequenceAlignment
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
from SimilarityFusion import *

#Synthetic example
if __name__ == "__main__2":
    np.random.seed(100)
    N = 200
    D = np.ones((N, N)) + 0.1*np.random.randn(N, N)
    D[D < 0] = 0
    I = np.arange(100)
    D[I, I] = 0

    I = np.zeros(40, dtype=np.int64)
    I[0:20] = 15 + np.arange(20)
    I[20::] = 50 + np.arange(20)
    J = I + 100
    D1 = 1.0*D
    D1[I, J] = 0

    I2 = np.arange(30, dtype=np.int64) + 20
    J2 = I2 + 60
    D2 = 1.0*D
    D2[I2, J2] = 0

    plt.subplot(121)
    plt.imshow(D1)
    plt.subplot(122)
    plt.imshow(D2)
    plt.show()

    doSimilarityFusion([D1, D2], K = 5, NIters = 20, reg = 1, PlotNames = ["D1", "D2"])

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180

    index = 8
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    fileprefix = "Covers80%i"%index

    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'
    #fileprefix = "BlurredLines"

    FeatureParams = {'DPixels':200, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':20, 'D2Samples':40, 'MFCCSamplesPerBlock':200, 'GeodesicDelta':10, 'NGeodesic':400, 'lifterexp':0.6, 'MFCCBeatsPerBlock':20, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    #FeatureParams = {'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}


    featuresfile = "%sFeatures.txt"%filename1
    if not os.path.exists(featuresfile):
        print "Getting features for %s..."%filename1
        (XAudio, Fs) = getAudio(filename1)
        (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize)
        (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

        print "Getting features for %s..."%filename2
        (XAudio, Fs) = getAudio(filename2)
        (tempo, beats) = getBeats(XAudio, Fs, TempoBias2, hopSize)
        (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

        pickle.dump({"Features1":Features1, "Features2":Features2, "O1":O1, "O2":O2}, open(featuresfile, "w"))
    else:
        X = pickle.load(open(featuresfile))
        Features1 = X['Features1']
        Features2 = X['Features2']
        O1 = X['O1']
        O2 = X['O2']

    Features = ['SSMs', 'Chromas']
    Features1b = {}
    Features2b = {}
    for F in Features:
        Features1b[F] = Features1[F]
        Features2b[F] = Features2[F]
    Features1 = Features1b
    Features2 = Features2b
    Kappa = 0.1
    K = 20
    NIters = 20
    getCSMSmithWatermanScoresEarlyFusion([Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes], doPlot = True)
    plt.show()
