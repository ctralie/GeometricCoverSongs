"""
Programmer: Chris Tralie
Purpose: To provide an interface for MIREX 2017
"""

import numpy as np
import scipy.io as sio
import os
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
from multiprocessing import Pool as PPool
from sys import exit, argv, stdout
import time

def getMatFilename(scratchDir, filename):
    prefix = filename.split("/")[-1]
    prefix = prefix[0:-4]
    return "%s/%s.mat"%(scratchDir, prefix)

def compareBlock(args):
    (idxs, hopSize, Kappa, FeatureParams, CSMTypes, allFiles, scratchDir) = args
    #Figure out block size thisM x thisN
    thisM = idxs[1] - idxs[0]
    thisN = idxs[3] - idxs[2]
    D = np.zeros((thisM, thisN))

    Features1 = []
    Features1Idx = {}
    Features2 = []
    tic = time.time()
    for i in range(thisM):
        idx = i + idxs[0]
        filename = getMatFilename(scratchDir, allFiles[idx])
        X = sio.loadmat(filename)
        Features1.append(getBlockWindowFeatures((None, X['Fs'], X['tempo'], X['beats'].flatten(), hopSize, FeatureParams), X['XMFCC'], X['XChroma']))
        Features1Idx[idx] = i
    for j in range(thisN):
        idx = j + idxs[2]
        if idx in Features1Idx:
            Features2.append(Features1[Features1Idx[idx]])
        else:
            filename = getMatFilename(scratchDir, allFiles[idx])
            X = sio.loadmat(filename)
            Features2.append(getBlockWindowFeatures((None, X['Fs'], X['tempo'], X['beats'].flatten(), hopSize, FeatureParams), X['XMFCC'], X['XChroma']))

    K = 20
    NIters = 3
    for i in range(len(Features1)):
        print("i = %i"%i)
        stdout.flush()
        (AllFeatures1, O1) = Features1[i]
        for j in range(len(Features2)):
            (AllFeatures2, O2) = Features2[j]
            res = getCSMSmithWatermanScoresEarlyFusionFull([AllFeatures1, O1, AllFeatures2, O2, Kappa, K, NIters, CSMTypes])
            D[i, j] = res['score']
    toc = time.time()
    print("Elapsed Time Block: ", toc-tic)
    stdout.flush()
    return D


if __name__ == '__main__':
    print argv
    if len(argv) < 5:
        print("Usage: python doMIREX.py <collection_list_file> <query_list_file> <working_directory> <output_file>")
        exit(0)
    #Open collection and query lists
    fin = open(argv[1], 'r')
    collectionFiles = [f.strip() for f in fin.readlines()]
    fin.close()

    fin = open(argv[2], 'r')
    queryFiles = [f.strip() for f in fin.readlines()]
    fin.close()

    #Take the union of the files in the query set and the collection
    #set and figure out how to index them from the original lists
    collectionSet = {}
    for i in range(len(collectionFiles)):
        collectionSet[collectionFiles[i]] = i
    allFiles = [] + collectionFiles
    query2All = {}
    for i in range(len(queryFiles)):
        if not queryFiles[i] in collectionSet:
            query2All[i] = len(allFiles)
            allFiles.append(queryFiles[i])
        else:
            query2All[i] = collectionSet[queryFiles[i]]

    print("There are %i files total"%len(allFiles))

    scratchDir = argv[3]
    filenameOut = argv[4]

    #Define parameters
    hopSize = 512
    Kappa = 0.1
    NMFCC = 20
    NChromaBins = 12
    lifterexp = 0.6
    FeatureParams = {'MFCCBeatsPerBlock':20, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Step 1: Precompute beat intervals, MFCC, and HPCP Features for each song
    for i in range(len(allFiles)):
        print("Computing features for file %i of %i..."%(i, len(allFiles)))
        filename = getMatFilename(scratchDir, allFiles[i])

        if os.path.exists(filename):
            print("Skipping...")
            continue

        print("Getting audio...")
        (XAudio, Fs) = getAudio(allFiles[i])
        print("Got audio")
        (tempo, beats) = getDegaraOnsets(XAudio, Fs, hopSize)
        winSize = (60.0/tempo)*Fs
        winSize = Fs/2

        print("Getting MFCCs...")
        XMFCC = getMFCCs(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)
        print("Finished MFCCs")

        print("Getting Chroma...")
        XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = NChromaBins)
        print("Finished Chroma")

        sio.savemat(filename, {"tempo":tempo, "beats":beats, "winSize":winSize, "Fs":Fs, "XMFCC":XMFCC, "XChroma":XChroma})

    #Setup parallel pool
    NThreads = 8
    if len(argv) > 5:
        NThreads = int(argv[5])
    parpool = PPool(NThreads)

    #Process blocks of similarity at a time
    N = len(allFiles)
    NPerBlock = 20
    NBlocks = int(np.ceil(N/float(NPerBlock)))
    ranges = []
    for i in range(NBlocks):
        for j in range(NBlocks):
            i2 = min(i+NPerBlock, N)
            j2 = min(j+NPerBlock, N)
            ranges.append([i, i2, j, j2])
    args = zip(ranges, [hopSize]*len(ranges), [Kappa]*len(ranges), [FeatureParams]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBlock, args)
    """
    for i in range(len(ranges)):
        compareBlock((ranges[i], hopSize, Kappa, FeatureParams, CSMTypes, allFiles, scratchDir))
    """

    #Assemble all blocks together
    D = np.zeros((N, N))
    for i in range(len(ranges)):
        [i1, i2, j1, j2] = ranges[i]
        D[i1:i2, j1:j2] = res[i]

    sio.savemat("D.mat", {"D":D})
