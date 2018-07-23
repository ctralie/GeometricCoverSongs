#Programmer: Chris Tralie
#Purpose: To extract cover song alignments for use in the GUI
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../SequenceAlignment")
import os
import glob
import scipy.io as sio
import scipy.misc
import time
import matplotlib.pyplot as plt
from CSMSSMTools import *
from BlockWindowFeatures import *
from pyMIRBasic.Onsets import *
from pyMIRBasic.AudioIO import *
import json


def getBase64File(filename):
    fin = open(filename, "rb")
    b = fin.read()
    b = b.encode("base64")
    fin.close()
    return b

def getBase64PNGImage(D, cmapstr):
    c = plt.get_cmap(cmapstr)
    D = np.round(255.0*D/np.max(D))
    C = c(np.array(D, dtype=np.int32))
    scipy.misc.imsave("temp.png", C)
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
        FeatureCSMs[FeatureName] = CSMs;

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
    #Add music as base64 files
    Results['file1'] = getBase64File(filename1)
    Results['file2'] = getBase64File(filename2)
    fout = open(outfilename, "w")
    fout.write(json.dumps(Results))
    fout.close()

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 120
    TempoBias2 = 120

    #File information (change this to try songs of your choosing)
    """
    filename1 = "MJ.mp3"
    filename2 = "AAF.mp3"
    fileprefix = "SmoothCriminal" #Save a JSON file with this prefix
    artist1 = "Michael Jackson"
    artist2 = "Alien Ant Farm"
    songName = "Smooth Criminal"
    """
    
    """
    filename1 = "Eurythmics.mp3"
    filename2 = "MarilynManson.mp3"
    artist1 = "Eurythmics"
    artist2 = "Marilyn Manson"
    fileprefix = "sweetdreams"
    songName = "Sweet Dreams"
    """
    
    """
    filename1 = "DejLoafTryMe.webm"
    filename2 = "BlackBearTryMe.m4a"
    artist1 = "Dej Loaf"
    artist2 = "BlackBear"
    fileprefix = "tryme"
    songName = "Try Me"
    """
    
    """
    filename1 = "WakaNoHands.webm"
    filename2 = "DannyVolaNoHands.m4a"
    artist1 = "Waka Flocka Flame"
    artist2 = "Danny Vola"
    fileprefix = "nohands"
    songName = "No Hands"
    """
    
    """
    #filename1 = "VivaldiLaFolia.webm"
    #filename2 = "VivaldiLaFolia.webm"
    filename1 = "LaFolia1.mp3"
    filename2 = "LaFolia2.mp3"
    artist1 = "Vivaldi"
    artist2 = "Vivaldi"
    fileprefix = "LaFolia"
    songName = "La Folia"
    TempoBias1 = 0
    TempoBias2 = 0
    """

    """
    filename1 = "BadCompany.mp3"
    filename2 = "BadCompanyFive.mp3"
    artist1 = "Bad Company"
    artist2 = "Five Finger Discount"
    fileprefix = "badcompany"
    songName = "Bad Company"
    """
    
    """
    filename1 = "BlurredLines.mp3"
    filename2 = "GotToGiveItUp.mp3"
    artist1 = "Robin Thicke"
    artist2 = "Marvin Gaye"
    fileprefix = "blurred"
    songName = "Blurred Lines"
    """
    
    """
    filename1 = "TotoOrig.webm"
    filename2 = "TotoCover.mp3"
    fileprefix = "Toto" #Save a JSON file with this prefix
    artist1 = "Toto"
    artist2 = "Leo Moracchioli feat. Rabea & Hannah"
    songName = "Africa"
    """
    
    """
    filename1 = "ELPOrig.webm"
    artist1 = "Emerson, Lake & Palmer"
    #filename2 = "ELPMarchingBand.m4a"
    #artist2 = "Tom Wallace"
    filename2 = "ELPCover2.webm"
    artist2 = "Paul Gilbert"
    fileprefix = "KarnEvil9"
    songName = "Karn Evil 9"
    """

    filename1 = "MJBad.mp3"
    filename2 = "SmoothCriminalMJ.mp3"
    fileprefix = "MJCross" #Save a JSON file with this prefix
    artist1 = "Michael Jackson - Bad"
    artist2 = "Michael Jackson - Smooth Criminal"
    songName = "MJ"

    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'SSMsDiffusion':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'CurvsSS':'Euclidean', 'TorsSS':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    compareTwoSongsJSON(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, "%s.json"%fileprefix, artist1, artist2)
