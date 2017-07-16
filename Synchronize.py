import numpy as np
import os
import glob
import scipy.io as sio
import scipy.misc
import time
import matplotlib.pyplot as plt
from CSMSSMTools import *
from BlockWindowFeatures import *
from Covers80Experiments import *
import json
import pyrubberband as pyrb
import subprocess

def getGreedyPerm(D):
    """
    Purpose: Naive O(N^2) algorithm to do the greedy permutation
    param: D (NxN distance matrix for points)
    return: (permutation (N-length array of indices),
            lambdas (N-length array of insertion radii))
    """
    N = D.shape[0]
    #By default, takes the first point in the list to be the
    #first point in the permutation, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return (perm, lambdas)

def syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = ""):
    XFinal = np.array([[0, 0]])
    beatsFinal = [] #The final beat locations based on hop size
    scoresFinal = []
    for i in range(path.shape[0]):
        [j, k] = [path[i, 0], path[i, 1]]
        scoresFinal.append(CSM[j, k])
        t1 = beats1[j]*hopSize
        t2 = beats1[j+BeatsPerBlock]*hopSize
        s1 = beats2[k]*hopSize
        s2 = beats2[k+BeatsPerBlock]*hopSize
        x1 = XAudio1[t1:t2]
        x2 = XAudio2[s1:s2]
        #Figure out the time factor by which to stretch x2 so it aligns
        #with x1
        fac = float(len(x1))/len(x2)
        print "fac = ", fac
        x2 = pyrb.time_stretch(x2, Fs, 1.0/fac)
        print "len(x1) = %i, len(x2) = %i"%(len(x1), len(x2))
        N = min(len(x1), len(x2))
        x1 = x1[0:N]
        x2 = x2[0:N]
        X = np.zeros((N, 2))
        X[:, 0] = x1
        X[:, 1] = x2
        if len(fileprefix) > 0:
            filename = "%s_%i.mp3"%(fileprefix, i)
            sio.wavfile.write("temp.wav", Fs, X)
            subprocess.call(["avconv", "-i", "temp.wav", filename])
        beat1 = beats1[j+1]*hopSize-t1
        beatsFinal.append(XFinal.shape[0])
        XFinal = np.concatenate((XFinal, X[0:beat1, :]))
    return (XFinal, beatsFinal, scoresFinal)

def synchronize(filename1, filename2, hopSize, TempoBiases, FeatureParams, CSMTypes, Kappa, fileprefix, doPlot = True, outputSnippets = True):
    print "Loading %s..."%filename1
    (XAudio1, Fs) = getAudio(filename1)
    print "Loading %s..."%filename2
    (XAudio2, Fs) = getAudio(filename2)

    maxScore = 0.0
    maxRes = {}

    for TempoBias1 in TempoBiases:
        for TempoBias2 in TempoBiases:
            print "Doing TempoBias1 = %i, TempoBias2 = %i..."%(TempoBias1, TempoBias2)
            (tempo, beats1) = getBeats(XAudio1, Fs, TempoBias1, hopSize)
            (Features1, O1) = getBlockWindowFeatures((XAudio1, Fs, tempo, beats1, hopSize, FeatureParams))
            (tempo, beats2) = getBeats(XAudio2, Fs, TempoBias2, hopSize)
            (Features2, O2) = getBlockWindowFeatures((XAudio2, Fs, tempo, beats2, hopSize, FeatureParams))
            print "Doing similarity fusion"
            K = 20
            NIters = 3
            res = getCSMSmithWatermanScoresEarlyFusionFull([Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes], True)
            print "score = ", res['maxD']
            if res['maxD'] > maxScore:
                print "New maximum score!"
                maxScore = res['maxD']
                maxRes = res
                res['beats1'] = beats1
                res['beats2'] = beats2
                res['TempoBias1'] = TempoBias1
                res['TempoBias2'] = TempoBias2
    res = maxRes
    print "TempoBias1 = %i, TempoBias2 = %i"%(res['TempoBias1'], res['TempoBias2'])
    beats1 = res['beats1']
    beats2 = res['beats2']
    CSM = res['CSM']
    CSM = CSM/np.max(CSM) #Normalize so highest score is 1

    if doPlot:
        plt.clf()
        plt.figure(figsize=(20, 8))
        plt.subplot(121)
        plt.imshow(CSM, cmap = 'afmhot')
        plt.hold(True)
        path = np.array(res['path'])
        plt.plot(path[:, 1], path[:, 0], '.')
        plt.subplot(122)
        plt.plot(path[:, 0], path[:, 1])
        plt.savefig("%sBlocksAligned.svg"%fileprefix, bbox_inches = 'tight')

    #Now extract signal snippets that are in correspondence, beat by beat
    BeatsPerBlock = FeatureParams['MFCCBeatsPerBlock']
    path = np.flipud(path)
    (XFinal, beatsFinal, scoresFinal) = syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = "")
    #Write out true positives synced
    sio.wavfile.write("temp.wav", Fs, XFinal)
    subprocess.call(["avconv", "-i", "temp.wav", "%sTrue.mp3"%fileprefix])
    #Write out true positives beat times and scores
    [beatsFinal, scoresFinal] = [np.array(beatsFinal), np.array(scoresFinal)]
    sio.savemat("%sTrue.mat"%fileprefix, {"beats":beatsFinal, "scores":scoresFinal, "BeatsPerBlock":BeatsPerBlock, "hopSize":hopSize})

    #Now save negative examples (same number as positive blocks)
    NBlocks = path.shape[0]
    x = CSM.flatten()
    idx = np.argsort(x)
    idx = idx[0:5*CSM.shape[0]]
    idxy = np.unravel_index(idx, CSM.shape)
    idx = np.zeros((idx.size, 2), dtype = np.int64)
    idx[:, 0] = idxy[0]
    idx[:, 1] = idxy[1]
    D = getCSM(idx, idx)
    #Do furthest point sampling on negative locations
    (perm, lambdas) = getGreedyPerm(D)
    path = idx[perm[0:NBlocks], :]
    if doPlot:
        plt.clf()
        plt.imshow(CSM, interpolation = 'nearest', cmap = 'afmhot')
        plt.hold(True)
        plt.plot(path[:, 1], path[:, 0], '.')
        plt.savefig("%sBlocksMisaligned.svg"%fileprefix, bbox_inches = 'tight')
    #Output negative example audio synced
    (XFinal, beatsFinal, scoresFinal) = syncBlocks(path, CSM, beats1, beats2, Fs, hopSize, XAudio1, XAudio2, BeatsPerBlock, fileprefix = "%sFalse"%fileprefix)
    sio.savemat("%sFalse.mat"%fileprefix, {"scores":scoresFinal, "BeatsPerBlock":BeatsPerBlock, "hopSize":hopSize})

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBiases = [60, 120, 180]

#    filename1 = "CSMViewer/MJ.mp3"
#    filename2 = "CSMViewer/AAF.mp3"
#    fileprefix = "SmoothCriminal"
#    artist1 = "Michael Jackson"
#    artist2 = "Alien Ant Farm"
#    songName = "Smooth Criminal"

#    filename1 = "CSMViewer/Eurythmics.mp3"
#    filename2 = "CSMViewer/MarilynManson.mp3"
#    artist1 = "Eurythmics"
#    artist2 = "Marilyn Manson"
#    fileprefix = "Synced/1"
#    songName = "Sweet Dreams"
#    TempoBiases = [120]

#    filename1 = "CSMViewer/sting.mp3"
#    filename2 = "CSMViewer/gaga.mp3"
#    fileprefix = "Synced/2"
#    TempoBiases = [60, 120, 180]

    filename1 = "CSMViewer/BadCompany.mp3"
    filename2 = "CSMViewer/BadCompanyFive.mp3"
    artist1 = "Bad Company"
    artist2 = "Five Finger Discount"
    fileprefix = "badcompany"
    songName = "Bad Company"
    TempoBiases = [120]


#    filename1 = "CSMViewer/261_1.ebm"
#    filename2 = "CSMViewer/261_2.ebm"
#    fileprefix = "261"

    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'SSMsDiffusion':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'CurvsSS':'Euclidean', 'TorsSS':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}


    synchronize(filename1, filename2, hopSize, TempoBiases, FeatureParams, CSMTypes, Kappa, fileprefix)
