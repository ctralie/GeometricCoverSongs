import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import os
import time
import SequenceAlignment.SequenceAlignment as SA
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
from SimilarityFusion import *
from Covers80Experiments import *

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

def makeISMIRPlot(index):
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180

    
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    fileprefix = "Covers80%i"%index
    artist1 = getCovers80ArtistName(files1[index])
    artist2 = getCovers80ArtistName(files2[index])

    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'
    #fileprefix = "BlurredLines"
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

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
    FeatureNames = {'SSMs':'MFCC SSMs', 'Chromas':'HPCP Blocks'}
    Features1b = {}
    Features2b = {}
    for F in Features:
        Features1b[F] = Features1[F]
        Features2b[F] = Features2[F]
    Features1 = Features1b
    Features2 = Features2b
    Kappa = 0.1
    K = 20
    NIters = 3
    #getCSMSmithWatermanScoresEarlyFusion([Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes], doPlot = True)
    #plt.show()
    
    CSMs = [] #Individual CSMs
    Ws = [] #W built from fused CSMs/SSMs
    Features = Features1.keys()
    #Compute all CSMs and SSMs
    plt.figure(figsize=(15, 20))
    for i in range(len(Features)):
        F = Features[i]
        SSMA = getCSMType(Features1[F], O1, Features1[F], O1, CSMTypes[F])
        SSMB = getCSMType(Features2[F], O2, Features2[F], O2, CSMTypes[F])
        CSMAB = getCSMType(Features1[F], O1, Features2[F], O2, CSMTypes[F])
        CSMs.append(CSMAB)
        M = SSMA.shape[0]
        N = SSMB.shape[0]
        #Build W from CSM and SSMs
        Ws.append(getWCSMSSM(SSMA, SSMB, CSMAB, K))
        plt.subplot(4, len(Features)+1, i+1)
        W = np.array(Ws[-1])
        W = W - np.diag(np.diag(W))
        plt.imshow(np.max(W) - W, cmap = 'gray', interpolation = 'nearest')
        plt.hold(True)
        plt.plot([M, M+N], [0, 0], 'c', linewidth=10)
        plt.plot([M, M], [0, M], 'c', linewidth=6)
        plt.plot([M, M+N], [M, M], 'c', linewidth=6)
        plt.plot([M+N, M+N], [0, M], 'c', linewidth=10)
        plt.xlim([0, W.shape[1]])
        plt.ylim([W.shape[0], 0])
        plt.title("$W_{AB}$ for %s Features"%FeatureNames[F])
        plt.xlabel("Concatenated Beat Index")
        plt.ylabel("Concatenated Beat Index")
        plt.subplot(4, len(Features)+1, (len(Features)+1)*1 + 1 + i)
        C = Ws[-1]
        C = C[0:M, M::]
        plt.imshow(np.max(C)-C, cmap = 'gray', interpolation = 'nearest')

        plt.title("$W_{AB}$ CSM Part %s"%FeatureNames[F])
        plt.xlabel("%s Beat Index"%artist2)
        plt.ylabel("%s Beat Index"%artist1)
        
        #Get binary CSM
        B = CSMToBinaryMutual(CSMAB, 0.1)
        plt.subplot(4, len(Features)+1, (len(Features)+1)*2 + 1 + i)
        plt.imshow(1-B, cmap = 'gray', interpolation = 'nearest')
        plt.title("Binary CSM %s"%FeatureNames[F])
        plt.xlabel("%s Beat Index"%artist2)
        plt.ylabel("%s Beat Index"%artist1)
        
        #Do Smith Waterman
        (maxD, D) = SA.swalignimpconstrained(B)
        plt.subplot(4, len(Features)+1, (len(Features)+1)*3 + 1 + i)
        plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
        plt.title("Smith Waterman Score = %.3g"%maxD)
        makeColorbar(4, len(Features)+1, (len(Features)+1)*3 + 1 + i)
        
    D = doSimilarityFusionWs(Ws, K, NIters, 1)
    D = D - np.diag(np.diag(D))
    plt.subplot(4, len(Features)+1, len(Features)+1)
    plt.imshow(np.max(D) - D, cmap = 'gray', interpolation = 'nearest')
    plt.hold(True)
    plt.plot([M, M+N], [0, 0], 'c', linewidth=10)
    plt.plot([M, M], [0, M], 'c', linewidth=6)
    plt.plot([M, M+N], [M, M], 'c', linewidth=6)
    plt.plot([M+N, M+N], [0, M], 'c', linewidth=10)
    plt.xlim([0, W.shape[1]])
    plt.ylim([W.shape[0], 0])
    plt.title("SNF Result $P$")
    plt.xlabel("Concatenated Beat Index")
    plt.ylabel("Concatenated Beat Index")
    
    plt.subplot(4, len(Features)+1, 2*(len(Features)+1))
    C = D[0:SSMA.shape[0], SSMA.shape[1]::]
    plt.imshow(np.max(C) - C, cmap = 'gray', interpolation = 'nearest')
    plt.title("$P$ CSM Part")
    plt.xlabel("%s Beat Index"%artist2)
    plt.ylabel("%s Beat Index"%artist1)
    
    
    #Now get binary CSMs
    #Get binary CSM
    C = np.exp(-C)
    B = CSMToBinaryMutual(C, 0.2)
    plt.subplot(4, len(Features)+1, (len(Features)+1)*3)
    plt.imshow(1-B, cmap = 'gray', interpolation = 'nearest')
    plt.title("$P$ Binary CSM")
    plt.xlabel("%s Beat Index"%artist2)
    plt.ylabel("%s Beat Index"%artist1)
    
    #Do Smith Waterman
    (maxD, D) = SA.swalignimpconstrained(B)
    plt.subplot(4, len(Features)+1, (len(Features)+1)*4)
    plt.imshow(D, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Smith Waterman Score = %.3g"%maxD)
    makeColorbar(4, len(Features)+1, (len(Features)+1)*4)
    
    
    plt.savefig("EarlySNFExample_%i.svg"%index, bbox_inches = 'tight')


if __name__ == '__main__':
    makeISMIRPlot(6)
    #for index in range(1, 80):
    #    makeISMIRPlot(index)
