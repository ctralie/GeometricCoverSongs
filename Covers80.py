"""
Programmer: Chris Tralie
Purpose: To run experiments on the Covers80 dataset and
report the results
"""
import numpy as np
import scipy.io as sio
import os
from sys import exit, argv
from BlockWindowFeatures import *
from EvalStatistics import *
from BatchCollection import *
from SimilarityFusion import *
from multiprocessing import Pool as PPool

def getCovers80ArtistName(filename):
    artistname = filename.split("/")[-1].split("+")[0]
    artistname = [s.capitalize() for s in artistname.split("_")]
    s = artistname[0]
    for i in range(1, len(artistname)):
        s = s + " " + artistname[i]
    return s

def getCovers80SongName(filename):
    songname = filename.split("/")[0]
    songname = [s.capitalize() for s in songname.split("_")]
    s = songname[0]
    for i in range(1, len(songname)):
        s = s + " " + songname[i]
    return s

def getCovers80Files():
    fin = open()

if __name__ == '__main__':
    #Setup parameters
    scratchDir = "ScratchCovers80"
    hopSize = 512
    TempoLevels = [60, 120, 180]
    Kappa = 0.1
    BeatsPerBlock = 20
    filePrefix = "Covers80_%g_%i"%(Kappa, BeatsPerBlock)
    if os.path.exists("%s.mat"%filePrefix):
        print("Already done covers80 with BeatsPerBlock = %i, Kappa = %g"%(BeatsPerBlock, Kappa))
        exit(0)

    FeatureParams = {'MFCCBeatsPerBlock':BeatsPerBlock, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':BeatsPerBlock, 'ChromasPerBlock':BeatsPerBlock*2, 'NMFCC':20, 'lifterexp':0.6}

    #What types of cross-similarity should be used to compare different blocks for different feature types
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Open collection and query lists
    fin = open("covers80collection.txt", 'r')
    allFiles = [f.strip() for f in fin.readlines()]
    fin.close()

    #Setup parallel pool
    NThreads = 8
    parpool = PPool(NThreads)

    #Precompute beat intervals, MFCC, and HPCP Features for each song
    NF = len(allFiles)
    args = zip(allFiles, [scratchDir]*NF, [hopSize]*NF, [Kappa]*NF, [CSMTypes]*NF, [FeatureParams]*NF, [TempoLevels]*NF, [{}]*NF)
    parpool.map(precomputeBatchFeatures, args)

    #Process blocks of similarity at a time
    N = len(allFiles)
    NPerBlock = 20
    ranges = getBatchBlockRanges(N, NPerBlock)
    args = zip(ranges, [Kappa]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBatchBlock, args)
    Ds = assembleBatchBlocks(list(CSMTypes) + ['SNF'], res, ranges, N)

    #Perform late fusion
    Scores = [1.0/Ds[F] for F in Ds.keys()]
    Ds['Late'] = doSimilarityFusion(Scores, 20, 20, 1)

    #Write results to disk
    sio.savemat("%s.mat"%filePrefix, Ds)
    fout = open("Covers80Results_%g_%s.html"%(Kappa, BeatsPerBlock), "w")
    fout.write("""
    <table border = "1" cellpadding = "10">
    <tr><td><h3>Name</h3></td><td><h3>Mean Rank</h3></td><td><h3>Mean Reciprocal Rank</h3></td><td><h3>Median #Rank</h3></td><td><h3>Top-01</h3></td><td><h3>Top-10</h3></td><td><h3>Covers80</h3></td></tr>""")
    for FeatureName in ['MFCCs', 'SSMs', 'Chromas', 'SNF', 'Late']:
        S = Ds[FeatureName]
        getCovers80EvalStatistics(S,  [1, 10], fout, name = FeatureName)
    fout.close()
