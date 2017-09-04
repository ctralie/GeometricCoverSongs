#Code that was primarily used to test EchoNest Features

import numpy as np
import sys
import scipy.io as sio
from scipy.interpolate import interp1d
import time
import cv2
import pickle
import matplotlib.pyplot as plt
from multiprocessing import Pool
from CSMSSMTools import *
import SequenceAlignment.SequenceAlignment as SA
import SequenceAlignment._SequenceAlignment as SAC


#############################################################################
##                  Code for beat averaging features
#############################################################################
#Helper function for getBeatSyncFeatures: Computes the average of the segment
#synchronous features within a beat interval [b1, b2]
def getBeatAvgFeatures(b1, b2, a, segs, segments):
    C = np.zeros(12)
    T = np.zeros(12)
    norm = 0.0
    b = a+1
    while b < len(segs):
        #Make sure this segment still overlaps with the beat interval
        left = max(segs[a], b1)
        right = min(segs[b], b2)
        if right - left <= 0:
            break
        n = (right - left)/(b2 - b1)
        norm += n
        C += np.array(segments[a]['pitches'])*n
        T += np.array(segments[a]['timbre'])*n
        a += 1
        b += 1
    return (C, T, norm)

#Average the segments within each beat interval (properly accounts for overlapping intervals)
def getBeatSyncFeatures(data):
    N = len(data['beats'])
    M = len(data['segments'])
    bts = [b['start'] for b in data['beats']]
    bts += [bts[-1] + data['beats'][-1]['duration']]
    segs = [s['start'] for s in data['segments']]
    segs += [segs[-1] + data['segments'][-1]['duration']]
    Chroma = np.zeros((12, N))
    Timbre = np.zeros((12, N))
    norms = np.zeros(N) #For debugging to make sure the entire interval of each beat is covered
    idxs = np.argsort(bts + segs) #This could be done with linear merging
    N = N + 1
    M = M + 1
    for i in range(len(idxs)-1):
        if not idxs[i] < N:
            continue
        #Narrow down to a beat interval
        i2 = i+1
        while i2 < N+M and (not idxs[i2] < N):
            i2 += 1
        if i2 == N+M:
            break
        #Now beat interval is from idxs[i] to idxs[i2]
        #Find where to start searching for segment intervals to the left of the beat
        a = max(i-1, 0)
        while a >= 0:
            if idxs[a] >= N or a == 0:
                break
            a -= 1
        if a > 0:
            a = idxs[a] - N
        b1 = bts[idxs[i]]
        b2 = bts[idxs[i2]]
        (Chroma[:, idxs[i]], Timbre[:, idxs[i]], norms[idxs[i]]) = getBeatAvgFeatures(b1, b2, a, segs, data['segments'])
    return (Chroma, Timbre)

#Return the raw segment chroma/timbre vectors
def getSegmentFeatures(data):
    M = len(data['segments'])
    ts = [s['start'] for s in data['segments']]
    Chroma = np.zeros((12, M))
    Timbre = np.zeros((12, M))
    for i in range(M):
        s = data['segments'][i]
        Chroma[:, i] = np.array(s['pitches'])
        Timbre[:, i] = np.array(s['timbre'])
    return (Chroma, Timbre)

def getInterpolatedFeatures(data, dt):
    tsbts = [s['start'] for s in data['beats']]
    tsseg = [s['start'] for s in data['segments']]
    tsnew = np.arange(0, min(tsseg[-1], tsbts[-1]), dt)
    N = len(tsnew)
    (ChromaOrig, TimbreOrig) = getSegmentFeatures(data)
    Chroma = np.zeros((12, N))
    Timbre = np.zeros((12, N))
    for i in range(12):
        f = interp1d(tsseg, ChromaOrig[i, :], kind = 'cubic')
        Chroma[i, :] = f(tsnew)
        f = interp1d(tsseg, TimbreOrig[i, :], kind = 'cubic')
        Timbre[i, :] = f(tsnew)
    return (Chroma, Timbre, tsnew)


#############################################################################
## Code for dealing with delay of raw timbral
#############################################################################
def getTimbralBeatAverageDelay(args):
    (filename, BeatsPerBlock) = args
    data = pickle.load(open("covers32k/%s.txt"%filename))
    (Chroma, Timbre) = getBeatSyncFeatures(data)
    #Skip chroma for now, only deal with timbre
    X = Timbre.T
    ND = X.shape[0] - BeatsPerBlock + 1
    dim = X.shape[1]*BeatsPerBlock
    Y = np.zeros((ND, dim), dtype = 'float32')
    YNorm = np.zeros((ND, dim), dtype = 'float32')
    for i in range(ND):
        x = np.array(X[i:i+BeatsPerBlock, :])
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm
        Y[i, :] = x.flatten()
        YNorm[i, :] = xnorm.flatten()
    return YNorm

#############################################################################
## Code for dealing with self-similarity matrices
#############################################################################

#Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
#computing of all of the beat-synchronous self-similarity matrices
#This version averages the Timbre and Pitch features within each beat
def getSSMsBeatAverage(args):
    (filename, BeatsPerBlock, DPixels) = args
    [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
    NPixels = DPixels*(DPixels-1)/2
    data = pickle.load(open("covers32k/%s.txt"%filename))
    (Chroma, Timbre) = getBeatSyncFeatures(data)
    #Skip chroma for now, only deal with timbre
    X = Timbre.T
    ND = X.shape[0] - BeatsPerBlock + 1
    Y = np.zeros((ND, NPixels), dtype = 'float32')
    YNorm = np.zeros((ND, NPixels), dtype = 'float32')
    for i in range(ND):
        x = np.array(X[i:i+BeatsPerBlock, :])
        D = getSSM(x, DPixels)
        Y[i, :] = D[I < J]
        #Mean-center x
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm
        D = getSSM(xn, DPixels)
        YNorm[i, :] = D[I < J]
    return YNorm

def makeDelay(x, NDelay):
    N = x.shape[0]
    M = N - NDelay + 1
    dim = x.shape[1]
    y = np.zeros((M, dim*NDelay))
    for i in range(NDelay):
        y[:, i*dim:(i+1)*dim] = x[i:i+M, :]
    return y

#Helper fucntion for "runCovers80Experiment" that can be used for multiprocess
#computing of all of the beat-synchronous self-similarity matrices
#This function interpolates the segment data onto a finer grid
def getSSMsInterp(args):
    (filename, BeatsPerBlock, DPixels, upfac) = args
    [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
    NPixels = DPixels*(DPixels-1)/2
    data = pickle.load(open("covers32k/%s.txt"%filename))
    tsbts = np.array([s['start'] for s in data['beats']])
    #Make the new sampling interval the mean of the average beat length
    #divided by the upsample factor
    dt = np.mean(tsbts[1:] - tsbts[0:-1])/upfac
    (Chroma, Timbre, tsnew) = getInterpolatedFeatures(data, dt)
    #Figure out the indices for each beat interval
    NBeats = len(tsbts)
    beatidxs = np.zeros(NBeats)
    idx = 0
    for i in range(NBeats):
        while idx < len(tsnew) and tsnew[idx] < tsbts[i]:
            idx += 1
        beatidxs[i] = idx

    #Skip chroma for now, only deal with timbre
    X = Timbre.T
    ND = NBeats - BeatsPerBlock
    Y = np.zeros((ND, NPixels), dtype = 'float32')
    YNorm = np.zeros((ND, NPixels), dtype = 'float32')
    for i in range(ND):
        t1 = data['beats'][i]['start']
        t2 = data['beats'][i+BeatsPerBlock]['start']
        x = np.array(X[beatidxs[i]:beatidxs[i+BeatsPerBlock], :])
        x = makeDelay(x, upfac) #Do a delay embedding with the delay window covering a beat
        D = getSSM(x, DPixels)
        Y[i, :] = D[I < J]
        #Mean/center x
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm
        D = getSSM(xn, DPixels)
        YNorm[i, :] = D[I < J]
    return YNorm

#############################################################################
## Code for running the experiments
#############################################################################
#Do batch tests on covers 80 using the EchoNest features
def runCovers80Experiment(BeatsPerBlock, Kappa):
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    N = len(files1)

    #Set up the parallel pool
    parpool = Pool(processes = 8)

    #Precompute all SSMs (can be stored in memory since dimensions are small)
    Z = zip(files1, [BeatsPerBlock]*len(files1), [BeatsPerBlock]*len(files1))
    SSMs1 = parpool.map(getSSMsBeatAverage, Z)
    Z = zip(files2, [BeatsPerBlock]*len(files2), [BeatsPerBlock]*len(files2))
    SSMs2 = parpool.map(getSSMsBeatAverage, Z)

    Scores = np.zeros((N, N))
    for i in range(N):
        print "Comparing song %i of %i"%(i, N)
        Z = zip([SSMs1[i]]*N, SSMs2, [Kappa]*N)
        Scores[i, :] = parpool.map(getCSMSmithWatermanScores, Z)
    return Scores

#Instead of looking in set 2 to compare to set 1, report mean rank,
#mean reciprocal rank, and median rank of identified track
#as well as top-01, top-10, top-25, top-50, and top-100
def runCovers80ExperimentAllSongs(BeatsPerBlock, PixelDim, Kappa, topsidx = [1, 25, 50, 100]):
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    NSongs = len(files1) #Should be 80
    files = files1 + files2
    N = len(files)

    #Set up the parallel pool
    parpool = Pool(processes = 8)

    #Precompute all SSMs (can be stored in memory since dimensions are small)
    Z = zip(files, [BeatsPerBlock]*N, [PixelDim]*N)
    SSMs = parpool.map(getSSMsBeatAverage, Z)
    Scores = np.zeros((N, N))
    for i in range(N):
        print "Comparing song %i of %i"%(i, N)
        Z = zip([SSMs[i]]*N, SSMs, [Kappa]*N)
        Scores[i, :] = parpool.map(getCSMSmithWatermanScores, Z)

    sio.savemat("AllScores.mat", {"Scores":Scores})

    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf)
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    for i in range(N):
        cover = (i+NSongs)%N #The index of the correct song
        print "%i, %i"%(i, cover)
        for k in range(N):
            if idx[i, k] == cover:
                ranks[i] = k+1
                break
    print ranks
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print "MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR)
    tops = np.zeros(len(topsidx))
    for i in range(len(tops)):
        tops[i] = np.sum(ranks <= topsidx[i])
        print "Top-%i: %i"%(topsidx[i], tops[i])
    return (Scores, MR, MRR, MDR, tops)


def writeArray(fout, x):
    for i in range(x.size):
        fout.write("%.3g"%x[i])
        if i < x.size-1:
            fout.write(",")
        else:
            fout.write("\n")

#Code to save features to files
def saveFeatures(BeatsPerBlock, PixelDim):
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    N = len(files1)

    #Set up the parallel pool
    parpool = Pool(processes = 8)

    #Precompute all SSMs (can be stored in memory since dimensions are small)
    Z = zip(files1, [BeatsPerBlock]*len(files1), [PixelDim]*len(files1))
    SSMs1 = parpool.map(getSSMsBeatAverage, Z)
    Z = zip(files2, [BeatsPerBlock]*len(files2), [PixelDim]*len(files2))
    SSMs2 = parpool.map(getSSMsBeatAverage, Z)

    fout1 = open("Covers80Files1_%i_%i.txt"%(BeatsPerBlock, PixelDim), 'w')
    fout1.write("%i\n"%(PixelDim*(PixelDim-1)/2))
    for i in range(len(SSMs1)):
        SSMs = SSMs1[i].flatten()
        fout1.write(files1[i] + "\n")
        writeArray(fout1, SSMs)
    fout1.close()

    fout2 = open("Covers80Files2_%i_%i.txt"%(BeatsPerBlock, PixelDim), 'w')
    fout2.write("%i\n"%(PixelDim*(PixelDim-1)/2))
    for i in range(len(SSMs2)):
        SSMs = SSMs2[i].flatten()
        fout2.write(files2[i] + "\n")
        writeArray(fout2, SSMs)
    fout2.close()

#############################################################################
## Entry points for running the experiments
#############################################################################
if __name__ == '__main__1':
    BeatsPerBlock = 30
    PixelDim = 10
    Kappa = 0.1
    (Scores, MR, MRR, MDR, tops) = runCovers80ExperimentAllSongs(BeatsPerBlock, PixelDim, Kappa, topsidx = [1, 25, 50, 100])

if __name__ == '__main__1':
    BeatsPerBlock = 30
    PixelDim = 10
    Kappa = 0.1
    saveFeatures(BeatsPerBlock, PixelDim)

if __name__ == '__main__1':
    BeatsPerBlocks = range(10, 80)
    Kappas = [0.05, 0.1, 0.15, 0.2]
    S = np.zeros((len(BeatsPerBlocks), len(Kappas)))
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            BeatsPerBlock = BeatsPerBlocks[i]
            Kappa = Kappas[j]
            Scores = runCovers80Experiment(BeatsPerBlock, Kappa)
            idx = np.argmax(Scores, 1)
            NumCorrect = np.sum(idx == np.arange(Scores.shape[0]))
            S[i, j] = NumCorrect
            print "Doing BeatsPerBlock = %i, Kappa = %g, NumCorrect = %i"%(BeatsPerBlock, Kappa, NumCorrect)
            sio.savemat("S.mat", {"S":S})

#Test smith waterman
if __name__ == "__main__":
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()

    idx = 1
    print files1[idx]
    print files2[idx]
#    SSMs1 = getSSMsInterp( (files1[idx], 20, 100, 10) )
#    SSMs2 = getSSMsInterp( (files2[idx], 20, 100, 10) )
    SSMs1 = getSSMsBeatAverage( (files1[idx], 20, 20) )
    SSMs2 = getSSMsBeatAverage( (files2[idx], 20, 20) )
    CSM = getCSM(SSMs1, SSMs2)
    CSMB = CSMToBinaryMutual(CSM, 0.1)
    (maxD, D) = SA.swalignimpconstrained(CSMB)
    sio.savemat("Abracadabra20x20SSM.mat", {"CSM":CSM, "CSMB":CSMB, "SW":D})
    maxD2 = SAC.swalignimpconstrained(CSMB)
    print "maxD = ", maxD, ", maxD2 = ", maxD2
    plt.subplot(221)
    plt.imshow(CSM, interpolation = 'none')
    plt.subplot(222)
    plt.imshow(CSMB, interpolation = 'none')
    plt.subplot(223)
    plt.imshow(D, interpolation = 'none')
    plt.show()

#Testing parallel processing
if __name__ == '__main__2':
    parpool = Pool(processes = 8)
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    files1 = files1[0:5]
    Z = zip(files1, [20]*len(files1), [20]*len(files1))
    S = parpool.map(getSSMsBeatAverage, Z)
    for s in S:
        plt.imshow(s)
        plt.show()

#For eyeballing whether beat-averaged features look right
if __name__ == '__main__3':
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    for f in files1[0:5]:
        data = pickle.load(open("covers32k/%s.txt"%f))
        (Chroma, Timbre) = getBeatSyncFeatures(data)
        (ChromaS, TimbreS) = getSegmentFeatures(data)
        plt.subplot(211)
        print np.sum(np.isnan(Timbre))
        plt.imshow(np.isnan(Timbre), aspect = 'auto', interpolation = 'none')
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(TimbreS, aspect = 'auto', interpolation = 'none')
        plt.colorbar()
        plt.show()
