
import numpy as np
import sys
import scipy.io as sio
import scipy.misc
from scipy.interpolate import interp1d
from scipy import signal
import time
import pickle
import matplotlib.pyplot as plt
from CSMSSMTools import *
from CurvatureTools import *
from MusicFeatures import *
import librosa
import subprocess

#Need to specify hopSize as as parameter so that beat onsets
#Align with MFCC and chroma windows
def getBlockWindowFeatures(args):
    #Unpack parameters
    (XAudio, Fs, tempo, beats, hopSize, FeatureParams) = args
    NBeats = len(beats)-1
    winSize = int(np.round((60.0/tempo)*Fs))
    Features = {}

    #########################
    #  MFCC-Based Features  #
    #########################
    #Step 1: Determine which features have been specified and allocate space
    usingMFCC = False
    [MFCCSamplesPerBlock, DPixels, NGeodesic, NJump, NCurv, NTors, D2Samples] = [-1]*7
    #Default parameters
    GeodesicDelta = 10
    CurvSigma = 40
    NMFCC = 20
    MFCCBeatsPerBlock = 20
    NMFCCBlocks = 0
    lifterexp = 0.6
    if 'NMFCC' in FeatureParams:
        NMFCC = FeatureParams['NMFCC']
        usingMFCC = True
    if 'lifterexp' in FeatureParams:
        lifterexp = FeatureParams['lifterexp']
        usingMFCC = True
    if 'MFCCBeatsPerBlock' in FeatureParams:
        MFCCBeatsPerBlock = FeatureParams['MFCCBeatsPerBlock']
        usingMFCC = True

    NMFCCBlocks = NBeats - MFCCBeatsPerBlock

    if 'MFCCSamplesPerBlock' in FeatureParams:
        MFCCSamplesPerBlock = FeatureParams['MFCCSamplesPerBlock']
        Features['MFCCs'] = np.zeros((NMFCCBlocks, MFCCSamplesPerBlock*NMFCC))
    if 'DPixels' in FeatureParams:
        DPixels = FeatureParams['DPixels']
        NPixels = DPixels*(DPixels-1)/2
        [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
        Features['SSMs'] = SSMs = np.zeros((NMFCCBlocks, NPixels), dtype = np.float32)
        usingMFCC = True
    if 'GeodesicDelta' in FeatureParams:
        GeodesicDelta = FeatureParams['GeodesicDelta']
        usingMFCC = True
    if 'NGeodesic' in FeatureParams:
        NGeodesic = FeatureParams['NGeodesic']
        Features['Geodesics'] = np.zeros((NMFCCBlocks, NGeodesic))
        usingMFCC = True
    if 'NJump' in FeatureParams:
        NJump = FeatureParams['NJump']
        Features['Jumps'] = np.zeros((NMFCCBlocks, NJump), dtype = np.float32)
        usingMFCC = True
    if 'NCurv' in FeatureParams:
        NCurv = FeatureParams['NCurv']
        Features['Curvs'] = np.zeros((NMFCCBlocks, NCurv), dtype = np.float32)
        usingMFCC = True
    if 'NTors' in FeatureParams:
        NTors = FeatureParams['NTors']
        Features['Tors'] = np.zeros((NMFCCBlocks, NTors), dtype = np.float32)
        usingMFCC = True
    if 'D2Samples' in FeatureParams:
        D2Samples = FeatureParams['D2Samples']
        Features['D2s'] = np.zeros((NMFCCBlocks, D2Samples), dtype = np.float32)
        usingMFCC = True
    if 'CurvSigma' in FeatureParams:
        CurvSigma = FeatureParams['CurvSigma']
        usingMFCC = True

    #Step 3: Compute Mel-Spaced log STFTs
    XMFCC = np.array([])
    if usingMFCC:
        XMFCC = getMFCCs(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)

    #Step 4: Compute MFCC-based features in z-normalized blocks
    for i in range(NMFCCBlocks):
        i1 = beats[i]
        i2 = beats[i+MFCCBeatsPerBlock]
        x = XMFCC[:, i1:i2].T
        #Mean-center x
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm

        #Straight block-windowed MFCC
        if MFCCSamplesPerBlock > -1:
            xnr = scipy.misc.imresize(xn, (MFCCSamplesPerBlock, xn.shape[1]))
            Features['MFCCs'][i, :] = xnr.flatten()

        #Compute SSM and D2 histogram
        SSMRes = xn.shape[0]
        if DPixels > -1:
            SSMRes = DPixels
        if DPixels > -1 or D2Samples > -1:
            (DOrig, D) = getSSM(xn, SSMRes)
        if DPixels > -1:
            Features['SSMs'][i, :] = D[I < J]
        if D2Samples > -1:
            [IO, JO] = np.meshgrid(np.arange(DOrig.shape[0]), np.arange(DOrig.shape[0]))
            Features['D2s'][i, :] = np.histogram(DOrig[IO < JO], bins = D2Samples, range = (0, 2))[0]
            Features['D2s'][i, :] = Features['D2s'][i, :]/np.sum(Features['D2s'][i, :]) #Normalize

        #Compute geodesic distance
        if NGeodesic > -1:
            jump = xn[1::, :] - xn[0:-1, :]
            jump = np.sqrt(np.sum(jump**2, 1))
            jump = np.concatenate(([0], jump))
            geodesic = np.cumsum(jump)
            geodesic = geodesic[GeodesicDelta*2::] - geodesic[0:-GeodesicDelta*2]
            Features['Geodesics'][i, :] = signal.resample(geodesic, NGeodesic)

        #Compute velocity/curvature/torsion
        MaxOrder = 0
        if NTors > -1:
            MaxOrder = 3
        elif NCurv > -1:
            MaxOrder = 2
        elif NJump > -1:
            MaxOrder = 1
        if MaxOrder > 0:
            curvs = getCurvVectors(xn, MaxOrder, CurvSigma)
            if MaxOrder > 2:
                tors = np.sqrt(np.sum(curvs[3]**2, 1))
                Features['Tors'][i, :] = signal.resample(tors, NTors)
            if MaxOrder > 1:
                curv = np.sqrt(np.sum(curvs[2]**2, 1))
                Features['Curvs'][i, :] = signal.resample(curv, NCurv)
            jump = np.sqrt(np.sum(curvs[1]**2, 1))
            Features['Jumps'][i, :] = signal.resample(jump, NJump)

    ###########################
    #  Chroma-Based Features  #
    ###########################
    #Step 1: Figure out which features are requested and allocate space
    usingChroma = False
    NChromaBlocks = 0
    ChromaBeatsPerBlock = 20
    if 'ChromaBeatsPerBlock' in FeatureParams:
        ChromaBeatsPerBlock = FeatureParams['ChromaBeatsPerBlock']
        NChromaBlocks = NBeats - ChromaBeatsPerBlock

    return Features

def compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix):
    print "Getting features for %s..."%filename1
    (XAudio, Fs) = getAudio(filename1)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize)
    Features1 = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    print "Getting features for %s..."%filename2
    (XAudio, Fs) = getAudio(filename2)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias2, hopSize)
    Features2 = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    Results = {'filename1':filename1, 'filename2':filename2, 'TempoBias1':TempoBias1, 'TempoBias2':TempoBias2, 'hopSize':hopSize, 'FeatureParams':FeatureParams, 'CSMTypes':CSMTypes, 'Kappa':Kappa}
    plt.figure(figsize=(16, 48))
    for FeatureName in Features1:
        CSM = getCSM(Features1[FeatureName], Features2[FeatureName])
        Results['CSM%s'%FeatureName] = CSM

        getCSMSmithWatermanScores([Features1[FeatureName], Features2[FeatureName], Kappa, CSMTypes[FeatureName]], True)
        plt.savefig("%s_SSMs_%s.svg"%(fileprefix, FeatureName), dpi=200, bbox_inches='tight')

    sio.savemat("%s.mat"%fileprefix, Results)
