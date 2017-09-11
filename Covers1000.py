"""
Programmer: Chris Tralie
Purpose: Code for doing experiments on covers 1000 dataset
"""
import numpy as np
import sys
import os
import glob
import scipy.io as sio
import subprocess
import time
from sys import exit, argv
from multiprocessing import Pool as PPool
from CSMSSMTools import *
from BlockWindowFeatures import *
from BatchCollection import *

def getAudioFeatures(hopSize, filename, mfccfilename, censfilename, hpcpfilename, beatsfilename):
    """
    Precompute and save MFCC, CENS, HPCP, and beats, before
    any blocked features are computed
    NOTE: Features saved at 32 bit precision to save space
    :param hopSize: STFT hop size for features
    :param filename: Path to audio file
    :param mfccfilename: Path to save MFCC features
    :param censfilename: Path to save CENS features
    :param hpcpfilename: Path to save HPCP features
    :param beatsfilename: Path to save beats (NOTE: 3 biases
        with dynamic programming, as well as a single madmom
        estimate, are computed)
    """
    from MusicFeatures import getAudio, getMFCCsLibrosa, getCensFeatures, getHPCPEssentia, getBeats
    if os.path.exists(mfccfilename) and os.path.exists(censfilename) and os.path.exists(hpcpfilename) and os.path.exists(beatsfilename):
        print "Skipping %s"%filename
        return

    print "Loading %s..."%filename
    (XAudio, Fs) = getAudio(filename)

    #Compute MFCCs
    winSize = Fs/2
    if os.path.exists(mfccfilename):
        print "Skipping MFCCs"
    else:
        print "Computing MFCCs..."
        XMFCC = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize, lifterexp = 0.6, NMFCC = 20)
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
        for TempoBias in [0, 60, 120, 180]:
            (tempo, beats) = getBeats(XAudio, Fs, TempoBias, hopSize, filename)
            beatsDict["beats%i"%TempoBias] = beats
            beatsDict["tempo%i"%TempoBias] = tempo
        sio.savemat(beatsfilename, beatsDict)

def getAudioFilename(filePrefix):
    audiofile = glob.glob("%s*"%filePrefix)
    filename = filePrefix
    for f in audiofile:
        if not f[-3::] == "txt":
            filename = f
            break
    return filename

def computeCovers100CliqueFeatures(args):
    """
    Compute the MFCC, CENS, HPCP, and beats for all songs
    in a clique.  Function is setup for parallel processing
    :param (CliqueNum, hopSize): Number of the clique, hopSize
        to use in STFT
    """
    (CliqueNum, hopSize) = args
    songs = glob.glob("Covers1000/%i/*.txt"%CliqueNum)
    for i in range(len(songs)):
        s = songs[i]
        num = int(os.path.split(s)[-1][0:-4])
        filename = getAudioFilename(s[0:-4])

        mfccfilename = "Covers1000/%i/%i_MFCC.mat"%(CliqueNum, num)
        censfilename = "Covers1000/%i/%i_CENS.mat"%(CliqueNum, num)
        hpcpfilename = "Covers1000/%i/%i_HPCP.mat"%(CliqueNum, num)
        beatsfilename = "Covers1000/%i/%i_Beats.mat"%(CliqueNum, num)

        getAudioFeatures(hopSize, filename, mfccfilename, censfilename, hpcpfilename, beatsfilename)

def getZappaFeatures(hopSize):
    """
    Get the MFCC, CENS, HPCP, and beats for all 8 Zappa
    covers
    :param hopSize: STFT hop size between windows
    """
    for i in range(1, 9):
        filename = "Covers1000/Zappa/%i.mp3"%i
        mfccfilename = "Covers1000/Zappa/%i_MFCC.mat"%i
        censfilename = "Covers1000/Zappa/%i_CENS.mat"%i
        hpcpfilename = "Covers1000/Zappa/%i_HPCP.mat"%i
        beatsfilename = "Covers1000/Zappa/%i_Beats.mat"%i
        getAudioFeatures(hopSize, filename, mfccfilename, censfilename, hpcpfilename, beatsfilename)

def precomputeCovers1000Features(hopSize, NThreads = 8):
    """
    Precompute all of the MFCC and HPCP features for the Covers1000 dataset
    """
    parpool = PPool(NThreads)
    cliques = range(1, 396)
    args = zip(cliques, [hopSize]*len(cliques))
    parpool.map(computeCovers100CliqueFeatures, args)
    """
    for i in range(1, 396):
        computeCovers100CliqueFeatures((i, hopSize))
    """

#Get the list of songs
def getSongPrefixes(verbose = False):
    AllSongs = []
    for i in range(1, 396):
        songs = glob.glob("Covers1000/%i/*.txt"%i)
        songs = sorted([s[0:-4] for s in songs])
        if verbose:
            print songs
            print sorted(songs)
            print "\n\n"
        AllSongs += songs
    return AllSongs

def getCovers1000Features(fileprefix, FeatureParams, TempoBiases = [60, 120, 180]):
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

def doZappaComparisons(Kappa, BeatsPerBlock):
    """
    Compare the 8 zappa covers to all songs in the covers
    1000 dataset, and save the results to "ResultsZappa.mat"
    :param Kappa: Nearest neighbor cutoff for SNF and
        binary CSMs
    :param BeatsPerBlock: BeatsPerBlock for HPCPs/MFCCs
    """
    FeatureParams = {'MFCCBeatsPerBlock':BeatsPerBlock, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':BeatsPerBlock, 'ChromasPerBlock':40}

    songs1 = ["Covers1000/Zappa/%i"%i for i in range(1, 9)]
    songs2 = getSongPrefixes() + songs1
    AllFeatures1 = []
    for i in range(len(songs1)):
        print "Getting features 1 %i of %i"%(i, len(songs1))
        AllFeatures1.append(getCovers1000Features(songs1[i], FeatureParams))

    AllResults = {}
    BatchSize = 8
    NBatches = len(songs2)/BatchSize
    for batch in range(len(songs2)/BatchSize):
        tic = time.time()
        for offset in range(BatchSize):
            j = offset + batch*BatchSize
            print "Doing j = %i"%j
            Features2 = getCovers1000Features(songs2[j], FeatureParams)
            for i in range(len(songs1)):
                print "Doing Zappa %i of %i Index %i"%(i+1, len(songs1), j)
                Results = compareSongs1000(AllFeatures1[i], Features2, BeatsPerBlock, Kappa, FeatureParams)
                for F in Results:
                    if not F in AllResults:
                        AllResults[F] = np.zeros((len(songs1), len(songs2)))
                    AllResults[F][i, j] = Results[F]
        sio.savemat("ResultsZappa.mat", AllResults)
        print "Batch %i Elapsed Time: "%batch, time.time() - tic

if __name__ == '__main__2':
    doZappaComparisons(0.1, 20)

if __name__ == '__main__2':
    hopSize = 512
    #Compute features for 1000 songs
    precomputeCovers1000Features(hopSize)
    #Compute Zappa Features
    getZappaFeatures(hopSize)

    #Make a text file with all of the audio filenames
    AllSongs = getSongPrefixes()
    fout = open("covers1000collection.txt", "w")
    for s in AllSongs:
        fout.write("%s\n"%getAudioFilename(s))
    fout.close()

if __name__ == '__main__':
    if len(argv) < 7:
        print argv
        print "Usage: python covers1000.py <doFeatures> <NPerBatch> <BatchNum> <Kappa> <BeatsPerBlock> <doMadmom>"
        exit(0)
    AllSongs = getSongPrefixes()
    [doFeatures, NPerBatch, BatchNum] = [int(a) for a in argv[1:4]]
    Kappa = float(argv[4])
    BeatsPerBlock = int(argv[5])
    doMadmom = int(argv[6])
    hopSize = 512

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    FeatureParams = {'MFCCBeatsPerBlock':BeatsPerBlock, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':BeatsPerBlock, 'ChromasPerBlock':BeatsPerBlock*2, 'NMFCC':20, 'lifterexp':0.6}

    scratchDir = "Covers1000Scratch"

    TempoLevels = [60, 120, 180]
    if doMadmom == 1:
        TempoLevels = [0]

    if doFeatures == 1:
        #If precomputing block features, ignore NPerBatch
        #And treat the batchnum as a song index
        filePrefix = AllSongs[BatchNum]
        print "filePrefix = ", filePrefix
        X = sio.loadmat("%s_MFCC.mat"%filePrefix)
        XMFCC = X['XMFCC']
        X = sio.loadmat("%s_HPCP.mat"%filePrefix)
        XHPCP = X['XHPCP']
        PFeatures = {'XMFCC':XMFCC, 'XChroma':XHPCP, 'NTempos':len(TempoLevels)}
        X = sio.loadmat("%s_Beats.mat"%filePrefix)
        for tidx in range(len(TempoLevels)):
            PFeatures['beats%i'%tidx] = X['beats%i'%TempoLevels[tidx]].flatten()
            PFeatures['tempos%i'%tidx] = X['tempo%i'%TempoLevels[tidx]]
        audiofilename = getAudioFilename(filePrefix)
        precomputeBatchFeatures((audiofilename, scratchDir, hopSize, Kappa, CSMTypes, FeatureParams, TempoLevels, PFeatures))
    else:
        #Compute features in a block
        allFiles = [getAudioFilename(s) for s in AllSongs]
        ranges = getBatchBlockRanges(1000, NPerBatch)
        compareBatchBlock((ranges[BatchNum], Kappa, CSMTypes, allFiles, scratchDir))
