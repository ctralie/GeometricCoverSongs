"""
Programmer: Chris Tralie
Purpose: To provide an interface for the MIREX 2017
cover songs competition
"""

import numpy as np
import scipy.io as sio
import os
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
import SequenceAlignment._SequenceAlignment as SAC
from multiprocessing import Pool as PPool
from sys import exit, argv, stdout
import time

def getMatFilename(scratchDir, filename):
    prefix = filename.split("/")[-1]
    prefix = prefix[0:-4]
    return "%s/%s.mat"%(scratchDir, prefix)

def compareBlock(args):
    """
    Process a rectangular block of the all pairs score matrix
    between all of the songs.  Return score matrices for each
    individual type of feature, in addition to one for early
    similarity network fusion
    """
    (idxs, hopSize, Kappa, FeatureParams, CSMTypes, allFiles, scratchDir) = args
    #Figure out block size thisM x thisN
    thisM = idxs[1] - idxs[0]
    thisN = idxs[3] - idxs[2]
    D = np.zeros((thisM, thisN))

    AllFeatures = {}
    AllSSMWs = {}
    tic = time.time()
    allidxs = [i + idxs[0] for i in range(thisM)]
    allidxs += [j + idxs[2] for j in range(thisN)]
    allidxs = np.unique(np.array(allidxs))
    #Compute features and precompute Ws for SSM parts
    for idx in allidxs:
        filename = getMatFilename(scratchDir, allFiles[idx])
        X = sio.loadmat(filename)
        thisFeats = [] #Holds features at all tempo levels
        thisWs = [] #Holds the SSM Ws at all tempo levels
        k = 0
        while 'beats%i'%k in X:
            (Feats, O) = getBlockWindowFeatures((None, X['Fs'], X['tempo%i'%k], X['beats%i'%k].flatten(), hopSize, FeatureParams), X['XMFCC'], X['XChroma'])
            thisFeats.append((Feats, O))
            #Precompute the W for the SSM part for similarity network fusion
            Ws = {}
            for F in Feats:
                SSM = getCSMType(Feats[F], O, Feats[F], O, CSMTypes[F])
                K = int(0.5*Kappa*SSM.shape[0])
                Ws[F] = getW(SSM, K)
            thisWs.append(Ws)
            k += 1
        AllFeatures[idx] = thisFeats
        AllSSMWs[idx] = thisWs

    K = 20
    NIters = 3
    Ds = {'SNF':np.zeros((thisM, thisN))}
    for i in range(thisM):
        print("i = %i"%i)
        stdout.flush()
        AllFeatures1 = AllFeatures[i+idxs[0]]
        SSMWs1 = AllSSMWs[i+idxs[0]]
        for j in range(thisN):
            AllFeatures2 = AllFeatures[j+idxs[2]]
            SSMWs2 = AllSSMWs[i+idxs[2]]
            #Compare all tempo levels
            for a in range(len(AllFeatures1)):
                (Features1, O1) = AllFeatures1[a]
                SSMWsA = SSMWs1[a]
                for b in range(len(AllFeatures2)):
                    (Features2, O2) = AllFeatures2[b]
                    SSMWsB = SSMWs2[b]
                    Ws = []
                    OtherCSMs = {}
                    #Compute all W matrices
                    (M, N) = (0, 0)
                    for F in Features1:
                        CSMAB = getCSMType(Features1[F], O1, Features2[F], O2, CSMTypes[F])
                        OtherCSMs[F] = CSMAB
                        (M, N) = (CSMAB.shape[0], CSMAB.shape[1])
                        k1 = int(0.5*Kappa*M)
                        k2 = int(0.5*Kappa*N)
                        WCSMAB = getWCSM(CSMAB, k1, k2)
                        WSSMA = SSMWsA[F]
                        WSSMB = SSMWsB[F]
                        Ws.append(setupWCSMSSM(WSSMA, WSSMB, WCSMAB))
                    #Do Similarity Fusion
                    D = doSimilarityFusionWs(Ws, K, NIters, 1)
                    #Extract CSM Part
                    CSM = D[0:M, M::] + D[M::, 0:M].T
                    DBinary = CSMToBinaryMutual(np.exp(-CSM), Kappa)
                    score = SAC.swalignimpconstrained(DBinary)
                    Ds['SNF'][i, j] = max(score, Ds['SNF'][i, j])
                    #In addition to fusion, compute scores for individual
                    #features to be used with the fusion later
                    for Feature in OtherCSMs:
                        if not Feature in Ds:
                            Ds[Feature] = np.zeros((thisM, thisN))
                        DBinary = CSMToBinaryMutual(OtherCSMs[Feature], Kappa)
                        score = SAC.swalignimpconstrained(DBinary)
                        Ds[Feature][i, j] = max(Ds[Feature][i, j], score)
    toc = time.time()
    print("Elapsed Time Block: ", toc-tic)
    stdout.flush()
    return Ds

def precomputeFeatures(allFiles, scratchDir, hopSize, lifterexp, TempoLevels = [60, 120, 180]):
    for i in range(len(allFiles)):
        print("Computing features for file %i of %i..."%(i, len(allFiles)))
        filename = getMatFilename(scratchDir, allFiles[i])

        if os.path.exists(filename):
            print("Skipping...")
            continue

        (XAudio, Fs) = getAudioLibrosa(allFiles[i])
        ret = {'Fs':Fs, 'hopSize':hopSize}

        #Get beats at different tempo levels
        tempos = []
        for k in range(len(TempoLevels)):
            level = TempoLevels[k]
            (tempo, beats) = getBeats(XAudio, Fs, level, hopSize)
            novelTempo = True
            for t in tempos:
                [smaller, larger] = [min(tempo, t), max(tempo, t)]
                if float(larger)/smaller < 1.1:
                    novelTempo = False
                    break
            if novelTempo:
                ret['beats%i'%len(tempos)] = beats
                ret['tempo%i'%len(tempos)] = tempo
                tempos.append(tempo)
        print("%i Unique Tempos"%len(tempos))

        winSize = (60.0/tempo)*Fs
        winSize = Fs/2
        ret['winSize'] = winSize

        XMFCC = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)
        ret['XMFCC'] = XMFCC

        XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = NChromaBins)
        ret['XChroma'] = XChroma

        sio.savemat(filename, ret)

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
    precomputeFeatures(allFiles, scratchDir, hopSize, lifterexp)

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
        i1 = i*NPerBlock
        for j in range(NBlocks):
            j1 = j*NPerBlock
            i2 = min(i1+NPerBlock, N)
            j2 = min(j1+NPerBlock, N)
            ranges.append([i1, i2, j1, j2])
    args = zip(ranges, [hopSize]*len(ranges), [Kappa]*len(ranges), [FeatureParams]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBlock, args)
    """
    for i in range(len(ranges)):
        compareBlock((ranges[i], hopSize, Kappa, FeatureParams, CSMTypes, allFiles, scratchDir))
    """

    #Assemble all blocks together
    Ds = {}
    for i in range(len(ranges)):
        [i1, i2, j1, j2] = ranges[i]
        for Feature in res[i]:
            if not Feature in Ds:
                Ds[Feature] = np.zeros((N, N))
            Ds[Feature][i1:i2, j1:j2] = res[i][Feature]

    #Save distance matrix in case there's a problem
    #with the text output
    sio.savemat("%s/D.mat"%scratchDir, Ds)
