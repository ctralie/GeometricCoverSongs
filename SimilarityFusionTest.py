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

    index = 75
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

    X1 = Features1['SSMs']
    X2 = Features2['SSMs']
    Y1 = Features1['Chromas']
    Y2 = Features2['Chromas']

    SSMX1 = getCSM(X1, X1)
    SSMX2 = getCSM(X2, X2)
    CSMX1X2 = getCSM(X1, X2)

    SSMY1 = getCSMCosine(Y1, Y1)
    SSMY2 = getCSMCosine(Y2, Y2)
    CSMY1Y2 = getCSMCosine(Y1, Y2)

    W1 = getWCSMSSM(SSMX1, SSMX2, CSMX1X2, 20)
    W2 = getWCSMSSM(SSMY1, SSMY2, CSMY1Y2, 20)

    # plt.subplot(121)
    # plt.imshow(W1, interpolation = 'none')
    # plt.subplot(122)
    # plt.imshow(W2, interpolation = 'none')
    # plt.show()

    tic = time.time()
    D = doSimilarityFusionWs([W1, W2], 20, 20, 1, ['SSMs', 'Chromas'])
    toc = time.time()
    print "Elapsed time: ", toc-tic

    N = X1.shape[0]
    M = X2.shape[0]
    CSM = D[0:N, N::] + D[N::, 0:N].T

    Kappa = 0.1
    plt.subplot(331)
    plt.imshow(CSMX1X2)
    plt.title("CSM SSMs")
    plt.subplot(334)
    DBinary = CSMToBinaryMutual(CSMX1X2, Kappa)
    plt.imshow(DBinary)
    (maxD, D) = SequenceAlignment.swalignimpconstrained(DBinary)
    plt.subplot(337)
    plt.imshow(D)
    plt.title("Score = %g"%maxD)

    plt.subplot(332)
    plt.imshow(CSMY1Y2)
    plt.title("CSM HPCPs")
    plt.subplot(335)
    DBinary = CSMToBinaryMutual(CSMY1Y2, Kappa)
    plt.imshow(DBinary)
    (maxD, D) = SequenceAlignment.swalignimpconstrained(DBinary)
    plt.subplot(338)
    plt.imshow(D)
    plt.title("Score = %g"%maxD)

    plt.subplot(333)
    plt.imshow(CSM)
    plt.title("CSM Fused")
    plt.subplot(336)
    DBinary = CSMToBinaryMutual(np.exp(-CSM), Kappa)
    plt.imshow(DBinary)
    (maxD, D) = SequenceAlignment.swalignimpconstrained(DBinary)
    plt.subplot(339)
    plt.imshow(D)
    plt.title("Score = %g"%maxD)

    plt.show()
