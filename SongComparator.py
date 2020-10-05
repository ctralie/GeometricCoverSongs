"""
Programmer: Chris Tralie
Purpose: To have some code that makes it easy to compare two songs
in this pipeline and to get verbose output and figures about all
of the different features / techniques
"""
import numpy as np
import sys
import scipy.io as sio
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from CSMSSMTools import *
from BlockWindowFeatures import *
from Onsets import *
import subprocess

def plotSongLabels(song1, song2, dim1 = 1, dim2 = 3):
    for k in range(dim1*dim2):
        plt.subplot(dim1, dim2, k+1)
        plt.xlabel("%s Beat Index"%song2)
        plt.ylabel("%s Beat Index"%song1)

def makeColorbar(dim1 = 1, dim2 = 3, k = 3):
    plt.subplot(dim1, dim2, k)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad = 0.05)
    plt.colorbar(cax = cax)

def makeISMIRPlot(AllDs, fileprefix, song1name, song2name):
    plt.clf()
    NSubplots = len(AllDs)
    plt.figure(figsize=(NSubplots*4.5, 3.5))
    for i in range(NSubplots):
        plt.subplot(1, NSubplots, i+1)
        (FeatureName, D) = AllDs[i]
        plt.imshow(D, interpolation = 'nearest', cmap = 'afmhot')
        plt.title("%s Score = %g"%(FeatureName, np.max(D)))
        makeColorbar(1, NSubplots, i+1)
    plotSongLabels(song1name, song2name, 1, NSubplots)
    plt.savefig("%s.svg"%fileprefix, bbox_inches = 'tight')

def compareTwoFeatureSets(Results, Features1, O1, Features2, O2, CSMTypes, Kappa, fileprefix, NIters = 3, K = 20, song1name = 'Song 1', song2name = 'Song 2'):
    plt.figure(figsize=(18, 5))
    #Do each feature individually
    AllDs = []
    for FeatureName in Features1:
        plt.clf()
        res = getCSMSmithWatermanScores(Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName], True)
        AllDs.append((FeatureName, res['D']))
        plotSongLabels(song1name, song2name)
        makeColorbar()
        plt.subplot(131)
        plt.title("CSM %s"%FeatureName)
        plt.savefig("%s_CSMs_%s.svg"%(fileprefix, FeatureName), dpi=200, bbox_inches='tight')

    #Do OR Merging
    plt.clf()
    res = getCSMSmithWatermanScoresORMerge(Features1, O1, Features2, O2, Kappa, CSMTypes, True)
    plt.subplot(131)
    plt.imshow(1-res['DBinary'], interpolation = 'nearest', cmap = 'gray')
    plt.title("CSM Binary OR Fused, $\kappa$=%g"%Kappa)
    plt.subplot(132)
    plt.imshow(res['D'], interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Smith Waterman Score = %g"%res['maxD'])
    plotSongLabels(song1name, song2name)
    plt.savefig("%s_CSMs_ORMerged.svg"%fileprefix, dpi=200, bbox_inches='tight')

    #Do cross-similarity fusion
    plt.clf()
    res = getCSMSmithWatermanScoresEarlyFusionFull(Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes, True)
    plt.clf()
    Results['CSMFused'] = res['CSM']
    plt.subplot(131)
    C = res['CSM']
    plt.imshow(np.max(C) - C, cmap = 'afmhot', interpolation = 'nearest')
    plt.title('W Similarity Network Fusion')
    plt.subplot(132)
    plt.imshow(1-res['DBinary'], interpolation = 'nearest', cmap = 'gray')
    plt.title("CSM Binary, $\kappa$=%g"%Kappa)
    plt.subplot(133)
    plt.imshow(res['D'], interpolation = 'nearest', cmap = 'afmhot')
    plt.title("Smith Waterman Score = %g"%res['maxD'])
    plotSongLabels(song1name, song2name)
    makeColorbar()
    plt.savefig("%s_CSMs_Fused.svg"%fileprefix, dpi=200, bbox_inches='tight')
    AllDs.append(('SNF', res['D']))
    makeISMIRPlot(AllDs, fileprefix, song1name, song2name)

    sio.savemat("%s.mat"%fileprefix, Results)

def compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix, song1name = 'Song 1', song2name = 'Song 2'):
    from AudioIO import getAudioLibrosa
    from Onsets import getBeats
    print("Getting features for %s..."%filename1)
    (XAudio, Fs) = getAudioLibrosa(filename1)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize, filename2)
    print("Tempo 1: %.3g bpm"%tempo)
    (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    print("Getting features for %s..."%filename2)
    (XAudio, Fs) = getAudioLibrosa(filename2)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias2, hopSize, filename2)
    print("Tempo 2: %.3g bpm"%tempo)
    (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    print("Feature Types: ", Features1.keys())

    Results = {'filename1':filename1, 'filename2':filename2, 'TempoBias1':TempoBias1, 'TempoBias2':TempoBias2, 'hopSize':hopSize, 'FeatureParams':FeatureParams, 'CSMTypes':CSMTypes, 'Kappa':Kappa}

    compareTwoFeatureSets(Results, Features1, O1, Features2, O2, CSMTypes, Kappa, fileprefix, song1name = song1name, song2name = song2name)

#Modify the main function below to try on songs of your choice
if __name__ == '__main__':
    #Fraction of nearest neighbors in binary cross-similarity matrix
    Kappa = 0.1 
    hopSize = 512
    #Tempo bias for each song in the dynamic programming beat tracker
    TempoBias1 = 180
    TempoBias2 = 180
    
    #Setup filenames, artist names, and song name
    from Covers80 import getCovers80ArtistName, getCovers80SongName
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    
    index = 4
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    fileprefix = "Covers80_%i"%index

    artist1 = getCovers80ArtistName(files1[index])
    artist2 = getCovers80ArtistName(files2[index])
    print("artist1 = %s"%artist1)
    songName = getCovers80SongName(files1[index])
    
    #Parameters for the blocked features
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}
    
    #Run comparison and make plots
    compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix, artist1, artist2)
