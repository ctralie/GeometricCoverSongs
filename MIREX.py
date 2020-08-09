"""
Programmer: Chris Tralie
Purpose: An entry point for the MIREX 2017 cover song ID task
"""
import numpy as np
import scipy.io as sio
from multiprocessing import Pool as PPool
from BatchCollection import *
from sys import exit, argv
from configparser import ConfigParser
from ast import literal_eval
import logging as logger
logger.basicConfig(level=logger.DEBUG)

if __name__ == '__main__':
    config = ConfigParser().read('config.ini')

    #Open collection and query lists
    logger.info("Reading Collections File")
    collections_file = config.get('PARAMETERS','collectionFilePath')
    fin = open(collections_file, 'r')
    collectionFiles = [f.strip() for f in fin.readlines()]
    fin.close()

    logger.info("Reading Query File")
    query_file = config.get('PARAMETERS', 'queryFilePath')
    fin = open(query_file, 'r')
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

    scratchDir = config.get('PARAMETERS', 'scratchDirectoryName')
    filenameOut = config.get('PARAMETERS', 'outputFileName')

    #Define parameters
    hopSize = int(config.get('HYPERPARAMETERS', 'hopSize'))
    Kappa = int(config.get('HYPERPARAMETERS', 'Kappa'))
    TempoLevels = literal_eval(config.get('HYPERPARAMETERS', 'TempoLevels')) #TempoLevels = [0] #Madmom only
    logger.info('TempoLevels: {} Length {}'.format(TempoLevels, len(TempoLevels)))

    MFCCBeatsPerBlock = int(config.get('HYPERPARAMETERS','MFCCBeatsPerBlock'))
    DPixels = int(config.get('HYPERPARAMETERS', 'DPixels'))
    MFCCSamplesPerBlock = int(config.get('HYPERPARAMETERS', 'MFCCSamplesPerBlock'))
    ChromaBeatsPerBlock = int(config.get('HYPERPARAMETERS', 'ChromaBeatsPerBlock'))
    ChromasPerBlock = int(config.get('HYPERPARAMETERS', 'ChromasPerBlock'))
    NMFCC = int(config.get('HYPERPARAMETERS', 'NMFCC'))
    lifterexp = float(config.get('HYPERPARAMETERS', 'lifterexp'))

    FeatureParams = {'MFCCBeatsPerBlock':MFCCBeatsPerBlock,
                     'DPixels':DPixels,
                     'MFCCSamplesPerBlock':MFCCSamplesPerBlock,
                     'ChromaBeatsPerBlock':ChromaBeatsPerBlock,
                     'ChromasPerBlock':ChromasPerBlock,
                     'NMFCC':NMFCC,
                     'lifterexp':lifterexp}

    CSMTypes = {'MFCCs':'Euclidean',
                'SSMs':'Euclidean',
                'Chromas':'CosineOTI'}

    #Setup parallel pool
    NThreads = int(config.get('PARAMETERS', 'numberOfThreads'))
    logger.info("Process will run with {} Threads".format(NThreads))
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
    NPerBlock = int(config.get('PARAMETERS', 'numberPerBlock'))
    ranges = getBatchBlockRanges(N, NPerBlock)
    args = zip(ranges, [Kappa]*len(ranges), [CSMTypes]*len(ranges), [allFiles]*len(ranges), [scratchDir]*len(ranges))
    res = parpool.map(compareBatchBlock, args)
    Ds = assembleBatchBlocks(list(CSMTypes) + ['SNF'], res, ranges, N)

    #Perform late fusion
    logger.info("Performing Late Fusion")
    numberOfNearestNeighbor = int(config.get('HYPERPARAMETERS', 'numberOfNearestNeighbor'))
    numberOfIter = int(config.get('HYPERPARAMETERS', 'numberOfIter'))
    Scores = [1.0/(1.0+Ds[F]) for F in Ds.keys()]
    Ds['Late'] = doSimilarityFusion(Scores, numberOfNearestNeighbor, numberOfIter, 1)
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
