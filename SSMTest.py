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

def extractSSM(D, DPixels):
    [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
    DRet = np.zeros((DPixels, DPixels))
    DRet[I < J] = D
    DRet = DRet + DRet.T
    return DRet

if __name__ == '__main__':
    Kappa = 0.2
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

    DPixels = 200
    FeatureParams = {'DPixels':DPixels, 'MFCCBeatsPerBlock':20, 'DiffusionKappa':0.05, 'tDiffusion':-1}
    (XAudio, Fs) = getAudio(filename1)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize)
    (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))
    SSMs = Features1['SSMs']
    SSMsD = Features1['SSMsDiffusion']
    N = SSMs.shape[0]
    for i in range(N):
        SSM = extractSSM(SSMs[i, :], DPixels)
        SSMD = extractSSM(SSMsD[i, :], DPixels)
        plt.subplot(121)
        plt.imshow(SSM)
        plt.title('SSM %i'%i)
        plt.subplot(122)
        plt.imshow(SSMD, interpolation = 'none')
        plt.savefig("SSMs%i.png"%i)
        sio.savemat("SSM.mat", {'SSM':SSM, 'SSMD':SSMD})

if __name__ == '__main__2':
    M = sio.loadmat("SSM.mat")
    SSM = M['SSM']
    N = SSM.shape[0]
    Kappa = 0.2

    t = 5
    plt.clf()
    tic = time.time()
    M = getDiffusionMap(SSM, Kappa, t)
    toc = time.time()
    print "Total time diffusion: ", toc-tic
    (SSMM, _) = getSSM(M, N)
    plt.subplot(131)
    plt.imshow(SSM)
    plt.subplot(132)
    plt.plot(M[:, -3], M[:, -2], '.')
    plt.axis('equal')
    plt.subplot(133)
    plt.imshow(SSMM)
    plt.show()
