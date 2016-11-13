#Programmer: Chris Tralie
#Purpose: To replicate my experiments from ISMIR2015 in Python with librosa

import numpy as np
from BlockWindowFeatures import *
from MusicFeatures import *
from multiprocessing import Pool as PPool

def getEvalStatistics(ScoresParam, N, NSongs, topsidx):
    Scores = np.array(ScoresParam)
    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf)
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    for i in range(N):
        cover = (i+NSongs)%N #The index of the correct song
        print("%i, %i"%(i, cover))
        for k in range(N):
            if idx[i, k] == cover:
                ranks[i] = k+1
                break
    print(ranks)
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print("MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR))
    tops = np.zeros(len(topsidx))
    for i in range(len(tops)):
        tops[i] = np.sum(ranks <= topsidx[i])
        print("Top-%i: %i"%(topsidx[i], tops[i]))
    #TODO: Add covers80 score
    return (MR, MRR, MDR, tops)

#############################################################################
## Code for running the experiments
#############################################################################

#Returns a dictionary of the form {'FeatureName':[Array of Features at tempo level 1, Array of Features at tempo level 2, ...., Array of Features at tempo level N]}
def getCovers80Features(FeatureParams, hopSize, TempoBiases):
    fin = open('covers32k/list1.list', 'r')
    files1 = ["covers32k/" + f.strip() + ".ogg" for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = ["covers32k/" + f.strip() + ".ogg" for f in fin.readlines()]
    fin.close()
    NSongs = len(files1) #Should be 80
    files = files1 + files2
    N = len(files)

    #Set up the parallel pool
    #parpool = Pool(processes = 8)
    #Precompute all SSMs for all tempo biases (can be stored in memory since dimensions are small)
    AllFeatures = {}
    for filename in files:
        (XAudio, Fs) = getAudio(filename)
        print "Getting features for %s..."%filename
        for k in range(len(TempoBiases)):
            (tempo, beats) = getBeats(XAudio, Fs, TempoBiases[k], hopSize)
            #(XAudio, Fs, hopSize, beats, tempo, BeatsPerBlock, FeatureParams)
            Features = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))
            for FeatureName in Features:
                if not FeatureName in AllFeatures:
                    AllFeatures[FeatureName] = []
                    for k in range(len(TempoBiases)):
                        AllFeatures[FeatureName].append([])
                AllFeatures[FeatureName][k].append(Features[FeatureName])
    return AllFeatures

def doCovers80Experiments(FeatureParams, hopSize, TempoBiases, Kappa, CSMTypes, matfilename):
    #What types of cross-similarity should be used to compare different blocks for different feature types

    AllFeatures = getCovers80Features(FeatureParams, hopSize, TempoBiases)
    Results = {'Params':FeatureParams, 'hopSize':hopSize, 'TempoBiases':TempoBiases, 'Kappa':Kappa, 'CSMTypes':CSMTypes}
    for FeatureName in AllFeatures:
        CSMType = 'Euclidean'
        if FeatureName in CSMTypes:
            CSMType = CSMTypes[FeatureName]
        Scores = getScores(AllFeatures[FeatureName], CSMType, Kappa)
        Results[FeatureName] = Scores
        print("\n\nScores %s"%FeatureName)
        getEvalStatistics(Scores, N, NSongs, [1, 25, 50, 100])
        sio.savemat(matfilename, Results)


#############################################################################
## Entry points for running the experiments
#############################################################################

if __name__ == '__main__2':
    Kappa = 0.1
    hopSize = 512
    TempoBiases = [60, 120, 180]

    FeatureParams = {'DPixels':50, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':40, 'D2Samples':40, 'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':50, 'GeodesicDelta':10}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D'}
    doCovers80Experiments(FeatureParams, hopSize, TempoBiases, Kappa, CSMTypes, "Results.mat")

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 120
    TempoBias2 = 120

    index = 75
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'

    FeatureParams = {'DPixels':50, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':20, 'D2Samples':40, 'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':50, 'GeodesicDelta':10, 'NGeodesic':400}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D'}

    compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, "Covers80%i"%index)
