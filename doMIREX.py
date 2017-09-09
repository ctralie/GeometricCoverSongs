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
    DsFilename = "%s/D%i_%i_%i_%i.mat"%(scratchDir, idxs[0], idxs[1], idxs[2], idxs[3])
    if os.path.exists(DsFilename):
        return sio.loadmat(DsFilename)
    #Figure out block size thisM x thisN
    thisM = idxs[1] - idxs[0]
    thisN = idxs[3] - idxs[2]
    D = np.zeros((thisM, thisN))

    AllFeatures = {}
    tic = time.time()
    allidxs = [i + idxs[0] for i in range(thisM)]
    allidxs += [j + idxs[2] for j in range(thisN)]
    allidxs = np.unique(np.array(allidxs))
    #Preload features and Ws for SSM parts
    ticfeatures = time.time()
    count = 1
    for idx in allidxs:
        filename = getMatFilename(scratchDir, allFiles[idx])
        AllFeatures[idx] = sio.loadmat(filename)
    tocfeatures = time.time()
    print("Elapsed Time Loading Features: ", tocfeatures-ticfeatures)
    stdout.flush()

    K = 20
    NIters = 3
    Ds = {'SNF':np.zeros((thisM, thisN))}
    for Feature in CSMTypes.keys():
        Ds[Feature] = np.zeros((thisM, thisN))
    for i in range(thisM):
        print("i = %i"%i)
        stdout.flush()
        thisi = i + idxs[0]
        Features1 = AllFeatures[thisi]
        for j in range(thisN):
            thisj = j + idxs[2]
            if thisj < thisi:
                #Only compute upper triangular part since it's symmetric
                continue
            Features2 = AllFeatures[thisj]
            #Compare all tempo levels
            for a in range(Features1['NTempos']):
                O1 = {'ChromaMean':Features1['ChromaMean%i'%a].flatten()}
                for b in range(Features2['NTempos']):
                    O2 = {'ChromaMean':Features2['ChromaMean%i'%b].flatten()}
                    Ws = []
                    OtherCSMs = {}
                    #Compute all W matrices
                    (M, N) = (0, 0)
                    for F in CSMTypes.keys():
                        CSMAB = getCSMType(Features1['%s%i'%(F, a)], O1, Features2['%s%i'%(F, b)], O2, CSMTypes[F])
                        OtherCSMs[F] = CSMAB
                        (M, N) = (CSMAB.shape[0], CSMAB.shape[1])
                        k1 = int(0.5*Kappa*M)
                        k2 = int(0.5*Kappa*N)
                        WCSMAB = getWCSM(CSMAB, k1, k2)
                        WSSMA = Features1['W%s%i'%(F, a)]
                        WSSMB = Features2['W%s%i'%(F, b)]
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
                        DBinary = CSMToBinaryMutual(OtherCSMs[Feature], Kappa)
                        score = SAC.swalignimpconstrained(DBinary)
                        Ds[Feature][i, j] = max(Ds[Feature][i, j], score)
    toc = time.time()
    print("Elapsed Time Block: ", toc-tic)
    stdout.flush()
    sio.savemat(DsFilename, Ds)
    return Ds

def precomputeFeatures(args):
    (audiofilename, scratchDir, hopSize, lifterexp, Kappa, FeatureParams, TempoLevels) = args
    tic = time.time()
    print("Computing features for %s..."%audiofilename)
    filename = getMatFilename(scratchDir, audiofilename)

    if os.path.exists(filename):
        print("Skipping...")
        return

    (XAudio, Fs) = getAudioScipy(audiofilename)
    winSize = Fs/2
    XMFCC = getMFCCs(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)

    XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = 12)

    #Computed blocked features at different tempo levels
    tempos = []
    ret = {'hopSize':hopSize, 'winSize':winSize, 'lifterexp':lifterexp}
    for level in TempoLevels:
        if level == 0:
            #Use madmom
            (tempo, beats) = getRNNDBNOnsets(audiofilename, Fs, hopSize)
        else:
            #Use dynamic programming beat tracker biased with a level
            (tempo, beats) = getBeats(XAudio, Fs, level, hopSize)
        novelTempo = True
        for t in tempos:
            [smaller, larger] = [min(tempo, t), max(tempo, t)]
            if float(larger)/smaller < 1.1:
                novelTempo = False
                break
        if not novelTempo:
            continue
        tidx = len(tempos)
        ret['beats%i'%tidx] = beats
        ret['tempo%i'%tidx] = tempo
        tempos.append(tempo)

        (Feats, O) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams), XMFCC, XChroma)
        ret['ChromaMean%i'%tidx] = O['ChromaMean']

        #Precompute the W for the SSM part for similarity network fusion
        for F in Feats:
            ret['%s%i'%(F, tidx)] = Feats[F]
            SSM = getCSMType(Feats[F], O, Feats[F], O, CSMTypes[F])
            K = int(0.5*Kappa*SSM.shape[0])
            ret['W%s%i'%(F, tidx)] = getW(SSM, K)

    ret['NTempos'] = len(tempos)
    print("%i Unique Tempos"%len(tempos))
    print("Elapsed Time: ", time.time() - tic)
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
    lifterexp = 0.6
    #TempoLevels = [60, 120, 180]
    TempoLevels = [0]
    FeatureParams = {'MFCCBeatsPerBlock':20, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Setup parallel pool
    NThreads = 8
    if len(argv) > 5:
        NThreads = int(argv[5])
    parpool = PPool(NThreads)

    #Step 1: Precompute beat intervals, MFCC, and HPCP Features for each song
    NF = len(allFiles)
    args = zip(allFiles, [scratchDir]*NF, [hopSize]*NF, [lifterexp]*NF, [Kappa]*NF, [FeatureParams]*NF, [TempoLevels]*NF)
    """
    for i in range(NF):
        precomputeFeatures((allFiles[i], scratchDir, hopSize, lifterexp, Kappa, FeatureParams, TempoLevels))
    """
    parpool.map(precomputeFeatures, args)

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
        for Feature in CSMTypes.keys() + ['SNF']:
            if not Feature in Ds:
                Ds[Feature] = np.zeros((N, N))
            Ds[Feature][i1:i2, j1:j2] = res[i][Feature]

    #Fill in lower triangular part
    for Feature in Ds.keys():
        Ds[Feature] = Ds[Feature] + Ds[Feature].T
        np.fill_diagonal(Ds[Feature], 0.5*np.diagonal(Ds[Feature], 0))
        plt.imshow(Ds[Feature], cmap = 'afmhot', interpolation = 'none')
        plt.show()

    #Save full distance matrix in case there's a problem
    #with the text output
    sio.savemat("%s/D.mat"%scratchDir, Ds)
