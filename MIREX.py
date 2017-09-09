"""
Programmer: Chris Tralie
Purpose: An entry point for the MIREX 2017 cover song ID task
"""
import numpy as np
import scipy.io as sio
from multiprocessing import Pool as PPool
from BatchCollection import *
from sys import exit, argv

if __name__ == '__main__':
    if len(argv) < 5:
        print("Usage: python doMIREX.py <collection_list_file> <query_list_file> <working_directory> <output_file> <num threads in parallel pool>")
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
    #TempoLevels = [60, 120, 180]
    TempoLevels = [0] #Madmom only
    FeatureParams = {'MFCCBeatsPerBlock':20, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40, 'NMFCC':20, 'lifterexp':0.6}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Setup parallel pool
    NThreads = 8
    if len(argv) > 5:
        NThreads = int(argv[5])
    parpool = PPool(NThreads)

    #Precompute beat intervals, MFCC, and HPCP Features for each song
    NF = len(allFiles)
    args = zip(allFiles, [scratchDir]*NF, [hopSize]*NF, [Kappa]*NF, [CSMTypes]*NF, [FeatureParams]*NF, [TempoLevels]*NF, [{}]*NF)

    """
    for i in range(NF):
        precomputeBatchFeatures((allFiles[i], scratchDir, hopSize, Kappa, CSMTypes, FeatureParams, TempoLevels, {}))
    """
    parpool.map(precomputeBatchFeatures, args)

    #Process blocks of similarity at a time
    N = len(allFiles)
    NPerBlock = 20
    ranges = getBatchBlockRanges(N, NPerBlock)
    args = zip(ranges, [Kappa]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBatchBlock, args)
    Ds = assembleBatchBlocks(CSMTypes.keys() + ['SNF'], res, ranges, N)

    #Perform late fusion
    Scores = [1.0/Ds[F] for F in Ds.keys()]
    Ds['Late'] = doSimilarityFusion(Scores, 20, 20, 1)
    D = Ds['Late']

    #Save full distance matrix in case there's a problem
    #with the text output
    sio.savemat("%s/D.mat"%scratchDir, Ds)

    #Save the results to a text file
    fout = open(filenameOut, "w")
    fout.write("Early+Late SNF Chris Tralie 2017\n")
    for i in range(len(allFiles)):
        f = allFiles[i]
        fout.write("%i\t%s\n"%(i+1, f))
    fout.write("Q/R")
    for i in range(len(allFiles)):
        fout.write("\t%i"%(i+1))
    for i in range(len(queryFiles)):
        idx = query2All[i]
        fout.write("\n%i"%(idx+1))
        for j in range(len(allFiles)):
            fout.write("\t%g"%(D[idx, j]))
    fout.close()
