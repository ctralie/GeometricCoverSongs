import numpy as np
import sys
import os
import glob
import scipy.io as sio
import subprocess
import time
from sys import exit, argv
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

def getCovers1000Features(fileprefix, TempoBiases = [60, 120, 180]):
    beats = sio.loadmat("%s_Beats.mat"%fileprefix)
    MFCCs = sio.loadmat("%s_MFCC.mat"%fileprefix)
    XMFCC = MFCCs['XMFCC']
    Fs = MFCCs['Fs']
    hopSize = MFCCs['hopSize']
    XChroma = sio.loadmat("%s_HPCP.mat"%fileprefix)['XHPCP']
    tempos = []
    Features = []
    for TempoBias in TempoBiases:
        beats1 = beats['beats%i'%TempoBias].flatten()
        tempo = beats['tempo%i'%TempoBias]
        tempos.append(tempo)
        if len(tempos) > 1:
            if np.min(np.array(tempos[0:-1]) - tempo) == 0:
                print "Rendundant tempo"
                tempos.pop()
                continue
        (Features1, O1) = getBlockWindowFeatures((None, Fs, tempo, beats1, hopSize, FeatureParams), XMFCC, XChroma)
        Features.append((Features1, O1))
    return Features

def compareSongs1000(Features1List, Features2List, BeatsPerBlock, Kappa, FeatureParams, K = 20, NIters = 3):
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}
    #Do each feature individually
    Results = {'SNF':0}
    (Features1, O1) = Features1List[0]
    for FeatureName in Features1:
        Results[FeatureName] = 0
    #Compare all tempo levels
    for i in range(len(Features1List)):
        (Features1, O1) = Features1List[i]
        for j in range(len(Features2List)):
            (Features2, O2) = Features2List[j]
            for FeatureName in Features1:
                Results[FeatureName] =  max(Results[FeatureName], getCSMSmithWatermanScores([Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName]], False))

            #Do cross-similarity fusion
            res = getCSMSmithWatermanScoresEarlyFusionFull([Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes], False)['score']
            Results['SNF'] = max(res, Results['SNF'])
    return Results


if __name__ == '__main__':
    if len(argv) < 7:
        print argv
        print "Usage: python covers1000.py <start1> <end1> <start2> <end2> <Kappa> <BeatsPerBlock>"
        exit(0)
    AllSongs = getSongPrefixes()
    N = len(AllSongs)
    [s1, e1, s2, e2] = [int(a) for a in argv[1:5]]
    Kappa = float(argv[5])
    BeatsPerBlock = int(argv[6])

    FeatureParams = {'MFCCBeatsPerBlock':BeatsPerBlock, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':BeatsPerBlock, 'ChromasPerBlock':40}

    songs1 = AllSongs[s1:e1+1]
    songs2 = AllSongs[s2:e2+1]

    AllFeatures1 = []
    AllFeatures2 = []
    for i in range(len(songs1)):
        print "Getting features 1 %i of %i"%(i, len(songs1))
        AllFeatures1.append(getCovers1000Features(songs1[i]))
    for j in range(len(songs2)):
        print "Getting features 2 %i of %i"%(j, len(songs2))
        AllFeatures2.append(getCovers1000Features(songs2[j]))

    AllResults = {}
    tic = time.time()
    for i in range(len(songs1)):
        print "Comparing %i of %i"%(i+1, len(songs1))
        for j in range(len(songs2)):
            Results = compareSongs1000(AllFeatures1[i], AllFeatures2[j], BeatsPerBlock, Kappa, FeatureParams)
            for F in Results:
                if not F in AllResults:
                    AllResults[F] = np.zeros((len(songs1), len(songs2)))
                AllResults[F][i, j] = Results[F]
    print "Elapsed Time: ", time.time() - tic
    sio.savemat("Results_%i_%i_%i_%i.mat"%(s1, e1, s2, e2), AllResults)

if __name__ == '__main__2':
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    BeatsPerBlock = 20
    Kappa = 0.1
    AllSongs = getSongPrefixes()

    (Features1, O1) = getCovers1000Features(AllSongs[0], 120)
    (Features2, O2) = getCovers1000Features(AllSongs[1], 120)
    print compareSongs1000(Features1, O1, Features2, O2, BeatsPerBlock, Kappa, FeatureParams)
