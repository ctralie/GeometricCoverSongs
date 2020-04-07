#Programmer: Chris Tralie
#Purpose: To extract cover song alignments for use in the GUI
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../SequenceAlignment")
import os
import glob
import scipy.io as sio
import skimage
import skimage.io
import time
import matplotlib.pyplot as plt
from CSMSSMTools import *
from BlockWindowFeatures import *
from pyMIRBasic.Onsets import *
from pyMIRBasic.AudioIO import *
import json
import argparse
import base64

def getBase64File(filename):
    fin = open(filename, "rb")
    b = fin.read()
    b = base64.b64encode(b)
    fin.close()
    return b

def getBase64PNGImage(D, cmapstr):
    c = plt.get_cmap(cmapstr)
    D = np.round(255.0*D/np.max(D))
    C = c(np.array(D, dtype=np.int32))
    C = np.array(np.round(C*255), dtype=np.uint8)
    skimage.io.imsave("temp.png", C)
    b = getBase64File("temp.png")
    os.remove("temp.png")
    return b

#http://stackoverflow.com/questions/1447287/format-floats-with-standard-json-module
class PrettyFloat(float):
    def __repr__(self):
        return '%.4g' % self
def pretty_floats(obj):
    if isinstance(obj, float):
        return PrettyFloat(obj)
    elif isinstance(obj, dict):
        return dict((k, pretty_floats(v)) for k, v in obj.items())
    elif isinstance(obj, (list, tuple)):
        return map(pretty_floats, obj)
    return obj

def compareTwoSongsJSON(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, outfilename, song1name = 'Song 1', song2name = 'Song 2'):
    print("Getting features for %s..."%filename1)
    (XAudio, Fs) = getAudioLibrosa(filename1)
    (tempo, beats1) = getBeats(XAudio, Fs, TempoBias1, hopSize, filename1)
    (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats1, hopSize, FeatureParams))

    print("Getting features for %s..."%filename2)
    (XAudio, Fs) = getAudioLibrosa(filename2)
    (tempo, beats2) = getBeats(XAudio, Fs, TempoBias2, hopSize, filename2)
    (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats2, hopSize, FeatureParams))

    print("Feature Types: ", Features1.keys())

    beats1 = beats1*hopSize/float(Fs)
    beats2 = beats2*hopSize/float(Fs)
    Results = {'song1name':song1name, 'song2name':song2name, 'hopSize':hopSize, 'FeatureParams':FeatureParams, 'Kappa':Kappa, 'CSMTypes':CSMTypes, 'beats1':pretty_floats(beats1.tolist()), 'beats2':pretty_floats(beats2.tolist())}
    #Do each feature individually
    FeatureCSMs = {}
    for FeatureName in Features1:
        print("Doing %s..."%FeatureName)
        res =  getCSMSmithWatermanScores(Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName], True)
        CSMs = {}
        CSMs['D'] = getBase64PNGImage(res['D'], 'afmhot')
        CSMs['CSM'] = getBase64PNGImage(res['CSM'], 'afmhot')
        CSMs['DBinary'] = getBase64PNGImage(1-res['DBinary'], 'gray')
        CSMs['score'] = res['score']
        FeatureCSMs[FeatureName] = CSMs
    
    #Do OR Merging
    print("Doing OR Merging...")
    res = getCSMSmithWatermanScoresORMerge(Features1, O1, Features2, O2, Kappa, CSMTypes, True)
    CSMs = {}
    CSMs['D'] = getBase64PNGImage(res['D'], 'afmhot')
    CSMs['CSM'] = getBase64PNGImage(1-res['DBinary'], 'gray')
    CSMs['DBinary'] = CSMs['CSM']
    CSMs['score'] = res['score']
    CSMs['FeatureName'] = 'ORFusion'
    FeatureCSMs['ORFusion'] = CSMs

    #Do cross-similarity fusion
    print("Doing similarity network fusion...")
    K = 20
    NIters = 3
    res = getCSMSmithWatermanScoresEarlyFusionFull(Features1, O1, Features2, O2, Kappa, K, NIters, CSMTypes, True)
    CSMs = {}
    CSMs['D'] = getBase64PNGImage(res['D'], 'afmhot')
    CSMs['CSM'] = getBase64PNGImage(res['CSM'], 'afmhot')
    CSMs['DBinary'] = getBase64PNGImage(1-res['DBinary'], 'gray')
    CSMs['score'] = res['score']
    FeatureCSMs['SNF'] = CSMs

    Results['FeatureCSMs'] = FeatureCSMs
    print("Saving results...")
    Results['beats1'] = list(Results['beats1'])
    Results['beats2'] = list(Results['beats2']) 
    for feature in ['MFCCs', 'SSMs', 'Chromas', 'ORFusion', 'SNF']:
        for imgtype in ['D', 'CSM', 'DBinary']:
            Results['FeatureCSMs'][feature][imgtype] = "data:image/png;base64, " + Results['FeatureCSMs'][feature][imgtype].decode('ascii')

    #Add music as base64 files
    path, ext = os.path.splitext(filename1)
    Results['file1'] = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename1).decode('ascii')
    path, ext = os.path.splitext(filename2)
    Results['file2'] = "data:audio/%s;base64, "%ext[1::] + getBase64File(filename2).decode('ascii')
    fout = open(outfilename, "w")
    fout.write(json.dumps(Results))
    fout.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename1', type=str, required=True, help="Path to audio file for first song")
    parser.add_argument('--filename2', type=str, required=True, help="Path to audio file for first song")
    parser.add_argument('--jsonfilename', type=str, required=True, help="Output name of json file")
    parser.add_argument('--artist1', type=str, default="Artist 1", help="Name of the artist of the first song")
    parser.add_argument('--artist2', type=str, default="Artist 2", help="Name of the artist of the second song")
    parser.add_argument('--tempobias1', type=int, default=120, help="Tempo bias of the beat tracker for the first song")
    parser.add_argument('--tempobias2', type=int, default=120, help="Tempo bias of the beat tracker for the second song")
    parser.add_argument('--hopsize', type=int, default=512, help="Hop size to use for the features")
    parser.add_argument('--kappa', type=float, default=0.1, help="Nearest neighbor threshold")
    opt = parser.parse_args()

    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'SSMsDiffusion':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'CurvsSS':'Euclidean', 'TorsSS':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    compareTwoSongsJSON(opt.filename1, opt.tempobias1, opt.filename2, opt.tempobias2, opt.hopsize, FeatureParams, CSMTypes, opt.kappa, opt.jsonfilename, opt.artist1, opt.artist2)
