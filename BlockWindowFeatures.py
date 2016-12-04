
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
    BlockFeatures = {}
    OtherFeatures = {}

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
        BlockFeatures['MFCCs'] = np.zeros((NMFCCBlocks, MFCCSamplesPerBlock*NMFCC))
    if 'DPixels' in FeatureParams:
        DPixels = FeatureParams['DPixels']
        NPixels = DPixels*(DPixels-1)/2
        [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
        BlockFeatures['SSMs'] = SSMs = np.zeros((NMFCCBlocks, NPixels), dtype = np.float32)
        usingMFCC = True
    if 'GeodesicDelta' in FeatureParams:
        GeodesicDelta = FeatureParams['GeodesicDelta']
        usingMFCC = True
    if 'NGeodesic' in FeatureParams:
        NGeodesic = FeatureParams['NGeodesic']
        BlockFeatures['Geodesics'] = np.zeros((NMFCCBlocks, NGeodesic))
        usingMFCC = True
    if 'NJump' in FeatureParams:
        NJump = FeatureParams['NJump']
        BlockFeatures['Jumps'] = np.zeros((NMFCCBlocks, NJump), dtype = np.float32)
        usingMFCC = True
    if 'NCurv' in FeatureParams:
        NCurv = FeatureParams['NCurv']
        BlockFeatures['Curvs'] = np.zeros((NMFCCBlocks, NCurv), dtype = np.float32)
        usingMFCC = True
    if 'NTors' in FeatureParams:
        NTors = FeatureParams['NTors']
        BlockFeatures['Tors'] = np.zeros((NMFCCBlocks, NTors), dtype = np.float32)
        usingMFCC = True
    if 'D2Samples' in FeatureParams:
        D2Samples = FeatureParams['D2Samples']
        BlockFeatures['D2s'] = np.zeros((NMFCCBlocks, D2Samples), dtype = np.float32)
        usingMFCC = True
    if 'CurvSigma' in FeatureParams:
        CurvSigma = FeatureParams['CurvSigma']
        usingMFCC = True

    #Step 3: Compute Mel-Spaced log STFTs
    XMFCC = np.array([])
    if usingMFCC:
        XMFCC = getMFCCs(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)
    else:
        NMFCCBlocks = 0

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
            BlockFeatures['MFCCs'][i, :] = xnr.flatten()

        #Compute SSM and D2 histogram
        SSMRes = xn.shape[0]
        if DPixels > -1:
            SSMRes = DPixels
        if DPixels > -1 or D2Samples > -1:
            (DOrig, D) = getSSM(xn, SSMRes)
        if DPixels > -1:
            BlockFeatures['SSMs'][i, :] = D[I < J]
        if D2Samples > -1:
            [IO, JO] = np.meshgrid(np.arange(DOrig.shape[0]), np.arange(DOrig.shape[0]))
            BlockFeatures['D2s'][i, :] = np.histogram(DOrig[IO < JO], bins = D2Samples, range = (0, 2))[0]
            BlockFeatures['D2s'][i, :] = BlockFeatures['D2s'][i, :]/np.sum(BlockFeatures['D2s'][i, :]) #Normalize

        #Compute geodesic distance
        if NGeodesic > -1:
            jump = xn[1::, :] - xn[0:-1, :]
            jump = np.sqrt(np.sum(jump**2, 1))
            jump = np.concatenate(([0], jump))
            geodesic = np.cumsum(jump)
            geodesic = geodesic[GeodesicDelta*2::] - geodesic[0:-GeodesicDelta*2]
            BlockFeatures['Geodesics'][i, :] = signal.resample(geodesic, NGeodesic)

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
                BlockFeatures['Tors'][i, :] = signal.resample(tors, NTors)
            if MaxOrder > 1:
                curv = np.sqrt(np.sum(curvs[2]**2, 1))
                BlockFeatures['Curvs'][i, :] = signal.resample(curv, NCurv)
            jump = np.sqrt(np.sum(curvs[1]**2, 1))
            BlockFeatures['Jumps'][i, :] = signal.resample(jump, NJump)

    ###########################
    #  Chroma-Based Features  #
    ###########################
    #Step 1: Figure out which features are requested and allocate space
    usingChroma = False
    NChromaBlocks = 0
    ChromaBeatsPerBlock = 20
    ChromasPerBlock = 40
    NChromaBins = 12
    if 'ChromaBeatsPerBlock' in FeatureParams:
        ChromaBeatsPerBlock = FeatureParams['ChromaBeatsPerBlock']
        NChromaBlocks = NBeats - ChromaBeatsPerBlock
        usingChroma = True
    if 'ChromasPerBlock' in FeatureParams:
        ChromasPerBlock = FeatureParams['ChromasPerBlock']
        usingChroma = True
    if 'NChromaBins' in FeatureParams:
        NChromaBins = FeatureParams['NChromaBins']
    XChroma = np.array([])
    if usingChroma:
        BlockFeatures['Chromas'] = np.zeros((NChromaBlocks, ChromasPerBlock*NChromaBins))
        #XChroma = getCensFeatures(XAudio, Fs, hopSize)
        XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = NChromaBins)
        #librosa.display.specshow(XChroma, y_axis='chroma', x_axis='time')
        #plt.show()
        OtherFeatures['ChromaMean'] = np.mean(XChroma, 1)
    for i in range(NChromaBlocks):
        i1 = beats[i]
        i2 = beats[i+ChromaBeatsPerBlock]
        x = XChroma[:, i1:i2].T
        x = scipy.misc.imresize(x, (ChromasPerBlock, x.shape[1]))
        xnorm = np.sqrt(np.sum(x**2, 1))
        xnorm[xnorm == 0] = 1
        x = x/xnorm[:, None]
        BlockFeatures['Chromas'][i, :] = x.flatten()

    return (BlockFeatures, OtherFeatures)

def compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix):
    print "Getting features for %s..."%filename1
    (XAudio, Fs) = getAudio(filename1)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize)
    (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    print "Getting features for %s..."%filename2
    (XAudio, Fs) = getAudio(filename2)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias2, hopSize)
    (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    Results = {'filename1':filename1, 'filename2':filename2, 'TempoBias1':TempoBias1, 'TempoBias2':TempoBias2, 'hopSize':hopSize, 'FeatureParams':FeatureParams, 'CSMTypes':CSMTypes, 'Kappa':Kappa}
    plt.figure(figsize=(16, 48))

    #Do each feature individually
    for FeatureName in Features1:
        score = getCSMSmithWatermanScores([Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName]], True)
        plt.savefig("%s_CSMs_%s.svg"%(fileprefix, FeatureName), dpi=200, bbox_inches='tight')

    #Do OR Merging
    plt.clf()
    plt.figure(figsize=(16, 16 + 16*len(Features1.keys())))
    score = getCSMSmithWatermanScoresORMerge([Features1, O1, Features2, O2, Kappa, CSMTypes], True)
    plt.savefig("%s_CSM_ORMerged.svg"%fileprefix, dpi=200, bbox_inches='tight')

    sio.savemat("%s.mat"%fileprefix, Results)
