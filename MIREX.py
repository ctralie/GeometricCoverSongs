"""
Programmer: Chris Tralie
Purpose: An entry point for the MIREX 2017 cover song ID task
"""
import numpy as np
import scipy.io as sio
from multiprocessing import Pool as PPool
from BatchCollection import *
from sys import exit, argv
import logging as logger
logger.basicConfig(level=logger.DEBUG)

if __name__ == '__main__':
    if len(argv) < 5:
        print("Usage: python doMIREX.py <collection_list_file> <query_list_file> <working_directory> <output_file> <num threads in parallel pool>")
        exit(0)
    #Open collection and query lists
    logger.info("Reading Collections File")
    fin = open(argv[1], 'r')
    collectionFiles = [f.strip() for f in fin.readlines()]
    fin.close()

    logger.info("Reading Query File")
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

    logger.info("There are %i music items in Collections file"%len(allFiles))

    scratchDir = argv[3]
    filenameOut = argv[4]

    #Define parameters
    hopSize = 512
    Kappa = 0.1
    TempoLevels = [60, 120, 180]
    #TempoLevels = [0] #Madmom only
    FeatureParams = {'MFCCBeatsPerBlock':20, 'DPixels':50, 'MFCCSamplesPerBlock':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40, 'NMFCC':20, 'lifterexp':0.6}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}

    #Setup parallel pool
    NThreads = 8
    logger.info("Process will run with {} Threads".format(NThreads))
    if len(argv) > 5:
        NThreads = int(argv[5])
    parpool = PPool(NThreads)

    #Precompute beat intervals, MFCC, and HPCP Features for each song
    logger.info("Precompute the features for each song")
    NF = len(allFiles)
    args = zip(allFiles, [scratchDir]*NF, [hopSize]*NF, [Kappa]*NF, [CSMTypes]*NF, [FeatureParams]*NF, [TempoLevels]*NF, [{}]*NF)

    """
    for i in range(NF):
        precomputeBatchFeatures((allFiles[i], scratchDir, hopSize, Kappa, CSMTypes, FeatureParams, TempoLevels, {}))
    """
    parpool.map(precomputeBatchFeatures, args)

    #Process blocks of similarity at a time
    logger.info("Perform Similarity Analysis")
    N = len(allFiles)
    NPerBlock = 20
    ranges = getBatchBlockRanges(N, NPerBlock)
    logger.debug('Ranges {}'.format(ranges))
    args = zip(ranges, [Kappa]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBatchBlock, args)
    Ds = assembleBatchBlocks(list(CSMTypes) + ['SNF'], res, ranges, N)

    #Perform late fusion
    logger.info("Performing Late Fusion")
    Scores = [1.0/(1.0+Ds[F]) for F in Ds.keys()]
    Ds['Late'] = doSimilarityFusion(Scores, 20, 20, 1)
    D = np.exp(-Ds['Late']) #Turn similarity score into a distance

    #Save full distance matrix in case there's a problem
    #with the text output
    logger.info("Save the Matrix form of results")
    sio.savemat("%s/D.mat"%scratchDir, Ds)

    #Save the results to a text file
    logger.info("Generating Final Results")
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
