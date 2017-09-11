"""
Programmer: Chris Tralie
Purpose: Code for processing batches features and song comparisons.
Used for Covers80, Covers1000, and MIREX
"""

import numpy as np
import scipy.io as sio
import os
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
import SequenceAlignment._SequenceAlignment as SAC
from sys import stdout
import time

def getMatFilename(scratchDir, filename):
    prefix = filename.split("/")[-1]
    prefix = prefix[0:-4]
    return "%s/%s.mat"%(scratchDir, prefix)

def compareBatchBlock(args):
    """
    Process a rectangular block of the all pairs score matrix
    between all of the songs.  Return score matrices for each
    individual type of feature, in addition to one for early
    similarity network fusion
    :param idxs: [start1, end1, start2, end2] range of rectangular
        block of songs to compare
    :param Kappa: Percent nearest neighbors to use both for
        binary cross-similarity and similarity network fusion
    :param CSMTypes: Dictionary of types of features and
        associated cross-similarity comparisons to do
    :param allFiles: List of all files that are being compared
        from which this block is drawn
    :param scratchDir: Path to directory for storing block results
    """
    (idxs, Kappa, CSMTypes, allFiles, scratchDir) = args
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

def getBatchBeats(TempoLevels, audiofilename, XAudio, Fs, hopSize, ret, consolidateTempos = True):
    """
    Compute all beat intervals and tempos for an audio file
    :param TempoLevels: An array of tempo biases.  0 means use madmom.
    The rest are biases for dynamic programming beat tracking
    :param audiofilename: Path to the audio file
    :param XAudio: Numpy array of audio samples
    :param Fs: Sample Rate
    :param hopSize: Hop size for beat onset functions
    :param ret: Dictionary to store the results
    """
    tempos = []
    for level in TempoLevels:
        (tempo, beats) = getBeats(XAudio, Fs, level, hopSize, audiofilename)
        novelTempo = True
        for t in tempos:
            [smaller, larger] = [min(tempo, t), max(tempo, t)]
            if float(larger)/smaller < 1.1:
                novelTempo = False
                break
        if not novelTempo and consolidateTempos:
            continue
        tidx = len(tempos)
        ret['beats%i'%tidx] = beats
        ret['tempos%i'%tidx] = tempo
        tempos.append(tempo)
    return tempos

def precomputeBatchFeatures(args):
    """
    Precompute all of the features for a file, including
    MFCCs, HPCPs, all beats/tempos at different levels,
    blocked features at different tempo levels, and the
    self-similarity W matrix for each feature.  Save all of
    this to a .mat file
    :param audiofilename: Path to audio file
    :param scratchDir: Path to directory to which to store features
    :param hopSize: Hop size of STFT windows for features
    :param lifterexp: MFCC lifter parameter
    :param Kappa: Kappa for W matrix for similarity fusion
    :param CSMTypes: Dictionary of types of features and
        associated cross-similarity comparisons to do
    :param FeatureParams: Dictionary of parameters for computing
                        features using BlockWindowFeatures.py
    :param TempoLevels: An array of tempo biases.  If this array
        contains a 0, compute Madmom tempos.  Otherwise, do
        dynamic programming beat tracking with that bias
    :param PFeatures: Precomputed features
    """
    (audiofilename, scratchDir, hopSize, Kappa, CSMTypes, FeatureParams, TempoLevels, PFeatures) = args
    tic = time.time()
    filename = getMatFilename(scratchDir, audiofilename)
    if os.path.exists(filename):
        print("Skipping...")
        return
    print("Computing features for %s..."%audiofilename)

    Fs = 22050
    XAudio = np.array([])
    lifterexp = 0.6
    if 'XMFCC' in PFeatures:
        XMFCC = PFeatures['XMFCC']
    else:
        if XAudio.size == 0:
            (XAudio, Fs) = getAudio(audiofilename)
            print "Fs = ", Fs
        NMFCC = 20
        if 'NMFCC' in FeatureParams:
            NMFCC = FeatureParams['NMFCC']
        lifterexp = 0.6
        if 'lifterexp' in FeatureParams:
            lifterexp = FeatureParams['lifterexp']
        winSize = Fs/2
        XMFCC = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)

    if 'XChroma' in PFeatures:
        XChroma = PFeatures['XChroma']
    else:
        if XAudio.size == 0:
            (XAudio, Fs) = getAudio(audiofilename)
            print "Fs = ", Fs
        XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = 12)

    #Computed blocked features at different tempo levels
    winSize = Fs/2
    ret = {'hopSize':hopSize, 'winSize':winSize, 'lifterexp':lifterexp}
    tempos = []
    if 'NTempos' in PFeatures:
        #If tempos have been precomputed, load them in
        NTempos = PFeatures['NTempos']
        for i in range(NTempos):
            ret['tempos%i'%i] = PFeatures['tempos%i'%i]
            ret['beats%i'%i] = PFeatures['beats%i'%i]
            tempos.append(ret['tempos%i'%i])
    else:
        #Otherwise, compute tempos / beat intervals
        tempos = getBatchBeats(TempoLevels, audiofilename, XAudio, Fs, hopSize, ret)
    for tidx in range(len(tempos)):
        tempo = ret['tempos%i'%tidx]
        beats = ret['beats%i'%tidx]
        print "XMFCC.shape = ", XMFCC.shape
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
    print("Elapsed Time: %g"%(time.time() - tic))
    sio.savemat(filename, ret)

def assembleBatchBlocks(FeatureTypes, res, ranges, N):
    """
    Code for assembling a bunch of rectangular block comparisons
    into one large matrix
    :param FeatureTypes: The types of features
    :param res: List of blocks returned from compareBatchBlock
    :param ranges: An array of block ranges, parallel with res
    :param N: The total batch is NxN
    :return Ds: A dictionary of NxN all pairs matrices, one for
        each feature type
    """
    #Assemble all blocks together
    Ds = {}
    for i in range(len(ranges)):
        [i1, i2, j1, j2] = ranges[i]
        for Feature in FeatureTypes:
            if not Feature in Ds:
                Ds[Feature] = np.zeros((N, N))
            Ds[Feature][i1:i2, j1:j2] = res[i][Feature]

    #Fill in lower triangular part
    for Feature in Ds.keys():
        Ds[Feature] = Ds[Feature] + Ds[Feature].T
        np.fill_diagonal(Ds[Feature], 0.5*np.diagonal(Ds[Feature], 0))

    return Ds

def getBatchBlockRanges(N, NPerBlock):
    """
    Get the row and column index ranges of all blocks in an
    all pairs similarity experiment between N songs
    :param N: The N songs in the experiment for an NxN matrix
    :param NPerBlock: The number of elements in a square block
    :returns ranges: An array of ranges [[starti, endi, startj, endj]]
        comprising each block.  Clipped to boundaries of NxN
        where appropriate.
    """
    K = int(np.ceil(N/float(NPerBlock)))
    ranges = []
    for i in range(K):
        i1 = i*NPerBlock
        for j in range(K):
            j1 = j*NPerBlock
            i2 = min(i1+NPerBlock, N)
            j2 = min(j1+NPerBlock, N)
            ranges.append([i1, i2, j1, j2])
    return ranges
