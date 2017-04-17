import numpy as np
import sys
import os
import glob
import scipy.io as sio
import subprocess
from CSMSSMTools import *
from BlockWindowFeatures import *

def doCovers1000Example(CliqueNum, hopSize):
    from MusicFeatures import getAudio, getMFCCs, getCensFeatures, getHPCPEssentia
    songs = glob.glob("Covers1000/%i/*.txt"%CliqueNum)
    for i in range(len(songs)):
        s = songs[i]
        num = int(os.path.split(s)[-1][0:-4])
        audiofile = glob.glob("Covers1000/%i/%i.*"%(CliqueNum, num))
        filename = s
        for f in audiofile:
            if not f[-3::] == "txt":
                filename = f
                break

        mfccfilename = "Covers1000/%i/%i_MFCC.mat"%(CliqueNum, num)
        censfilename = "Covers1000/%i/%i_CENS.mat"%(CliqueNum, num)
        hpcpfilename = "Covers1000/%i/%i_HPCP.mat"%(CliqueNum, num)
        beatsfilename = "Covers1000/%i/%i_Beats.mat"%(CliqueNum, num)

        if os.path.exists(mfccfilename) and os.path.exists(censfilename) and os.path.exists(hpcpfilename) and os.path.exists(beatsfilename):
            print "Skipping %s"%f
            continue

        print "Loading %s..."%filename

        (XAudio, Fs) = getAudio(filename)

        #Compute MFCCs
        winSize = Fs/2
        if os.path.exists(mfccfilename):
            print "Skipping MFCCs"
        else:
            print "Computing MFCCs..."
            XMFCC = getMFCCs(XAudio, Fs, winSize, hopSize, lifterexp = 0.6, NMFCC = 20)
            sio.savemat(mfccfilename, {"XMFCC":XMFCC, "winSize":winSize, "hopSize":hopSize, "Fs":Fs})

        #Compute CENs
        if os.path.exists(censfilename):
            print "Skipping CENS"
        else:
            print "Computing CENS..."
            XCENS = getCensFeatures(XAudio, Fs, hopSize)
            sio.savemat(censfilename, {"XCENS":XCENS, "hopSize":hopSize, "Fs":Fs})


        #Compute HPCPs
        if os.path.exists(hpcpfilename):
            print "Skipping HPCP"
        else:
            print "Computing HPCP..."
            XHPCP = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = 12)
            sio.savemat(hpcpfilename, {"XHPCP":XHPCP, "hopSize":hopSize, "Fs":Fs})

        #Do beat tracking
        if os.path.exists(beatsfilename):
            print "Skipping beats"
        else:
            beatsDict = {'Fs':Fs, 'hopSize':hopSize}
            print "Computing beats..."
            for TempoBias in [60, 120, 180]:
                (tempo, beats) = getBeats(XAudio, Fs, TempoBias, hopSize)
                beatsDict["beats%i"%TempoBias] = beats
                beatsDict["tempo%i"%TempoBias] = tempo
            sio.savemat(beatsfilename, beatsDict)

def precomputeFeatures():
    """
    Precompute all of the MFCC and HPCP features for the Covers1000 dataset
    """
    for i in range(1, 396):
        doCovers1000Example(i, 512)

#Get the list of songs
def getSongPrefixes():
    AllSongs = []
    for i in range(1, 396):
        songs = glob.glob("Covers1000/%i/*.txt"%i)
        songs = [s[0:-4] for s in songs]
        AllSongs += songs
    return AllSongs

def getCovers1000Features(fileprefix, TempoBias):
    beats = sio.loadmat("%s_Beats.mat"%fileprefix)
    beats1 = beats['beats%i'%TempoBias].flatten()
    tempo = beats['tempo%i'%TempoBias]
    MFCCs = sio.loadmat("%s_MFCC.mat"%fileprefix)
    XMFCC = MFCCs['XMFCC']
    Fs = MFCCs['Fs']
    hopSize = MFCCs['hopSize']
    XChroma = sio.loadmat("%s_HPCP.mat"%fileprefix)['XHPCP']
    (Features1, O1) = getBlockWindowFeatures((None, Fs, tempo, beats1, hopSize, FeatureParams), XMFCC, XChroma)
    return (Features1, O1)

def compareSongs1000(Features1, O1, Features2, O2, BeatsPerBlock, Kappa, FeatureParams):
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Do each feature individually
    FeatureCSMs = {}
    for FeatureName in Features1:
        print "Doing %s..."%FeatureName
        res =  getCSMSmithWatermanScores([Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName]], True)
        CSMs = {}
        CSMs['D'] = res['D']
        CSMs['CSM'] = res['CSM']
        CSMs['DBinary'] = 1-res['DBinary']
        CSMs['score'] = res['score']
        FeatureCSMs[FeatureName] = CSMs;

    #Do OR Merging
    print "Doing OR Merging..."
    res = getCSMSmithWatermanScoresORMerge([Features1, O1, Features2, O2, Kappa, CSMTypes], True)
    CSMs = {}
    CSMs['D'] = res['D']
    CSMs['CSM'] = res['D']
    CSMs['DBinary'] = CSMs['CSM']
    CSMs['score'] = res['score']
    CSMs['FeatureName'] = 'ORFusion'
    FeatureCSMs['ORFusion'] = CSMs

    #Do cross-similarity fusion
    print "Doing similarity network fusion..."
    K = 20
    NIters = 3
    res = getCSMSmithWatermanScoresEarlyFusionFull([Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes], True)
    CSMs = {}
    CSMs['D'] = res['D']
    CSMs['CSM'] = res['CSM']
    CSMs['DBinary'] = 1-res['DBinary']
    CSMs['score'] = res['score']
    FeatureCSMs['SNF'] = CSMs

    for f in FeatureCSMs:
        CSMs = FeatureCSMs[f]
        plt.subplot(121)
        plt.imshow(CSMs['CSM'], cmap = 'afmhot')
        plt.title(f)
        plt.subplot(122)
        plt.imshow(CSMs['D'], cmap = 'afmhot')
        plt.savefig("%s.svg"%f, bbox_inches = 'tight')


if __name__ == '__main__':
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    BeatsPerBlock = 20
    Kappa = 0.1
    AllSongs = getSongPrefixes()

    (Features1, O1) = getCovers1000Features(AllSongs[0], 120)
    (Features2, O2) = getCovers1000Features(AllSongs[1], 120)
    compareSongs1000(Features1, O1, Features2, O2, BeatsPerBlock, Kappa, FeatureParams)
