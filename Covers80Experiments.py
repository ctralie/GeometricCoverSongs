#Programmer: Chris Tralie
#Purpose: To replicate my experiments from ISMIR2015 in Python with librosa

import numpy as np
from BlockWindowFeatures import *
from MusicFeatures import *
from EvalStatistics import *
from multiprocessing import Pool as PPool

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
    files = files1 + files2

    #Set up the parallel pool
    #parpool = Pool(processes = 8)
    #Precompute all SSMs for all tempo biases (can be stored in memory since dimensions are small)
    AllFeatures = {}
    OtherFeatures = []
    for k in range(len(TempoBiases)):
        OtherFeatures.append([])
    for filename in files:
        (XAudio, Fs) = getAudio(filename)
        print "Getting features for %s..."%filename
        for k in range(len(TempoBiases)):
            (tempo, beats) = getBeats(XAudio, Fs, TempoBiases[k], hopSize)
            #(XAudio, Fs, hopSize, beats, tempo, BeatsPerBlock, FeatureParams)
            (Features, Other) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))
            for FeatureName in Features:
                if not FeatureName in AllFeatures:
                    AllFeatures[FeatureName] = []
                    for a in range(len(TempoBiases)):
                        AllFeatures[FeatureName].append([])
                AllFeatures[FeatureName][k].append(Features[FeatureName])
            OtherFeatures[k].append(Other)
    return (AllFeatures, OtherFeatures, files)

def doCovers80Experiments(FeatureParams, hopSize, TempoBiases, Kappa, CSMTypes, matfilename, fout):
    NSongs = 80
    N = NSongs*2

    (AllFeatures, OtherFeatures, files) = getCovers80Features(FeatureParams, hopSize, TempoBiases)

    #Setup files that will hold cross-similarity images
    for i in range(NSongs):
        fh = open("CSMResults/%i.html"%i, "w")
        fh.write("<html><body><h1>%s</h1><HR><BR>"%files[i])
        fh.close()

    Results = {'Params':FeatureParams, 'hopSize':hopSize, 'TempoBiases':TempoBiases, 'Kappa':Kappa, 'CSMTypes':CSMTypes}

    print "Scoring ", AllFeatures.keys()
    for FeatureName in AllFeatures:
        CSMType = 'Euclidean' #Euclidean comparison by default
        if FeatureName in CSMTypes:
            CSMType = CSMTypes[FeatureName]
        (Scores, BestTempos) = getScores(AllFeatures[FeatureName], OtherFeatures, CSMType, Kappa)
        Results[FeatureName] = Scores
        Results["%sTempos"%FeatureName] = BestTempos
        print("\n\nScores %s"%FeatureName)
        getCovers80EvalStatistics(Scores, N, NSongs, [1, 25, 50, 100], fout, FeatureName)
        sio.savemat(matfilename, Results)

        #Output the cross-similarity matrices for this feature
        for i in range(NSongs):
            [i1, i2] = BestTempos[i, i, :]
            F1 = AllFeatures[FeatureName][i1][i]
            O1 = OtherFeatures[i1][i]
            F2 = AllFeatures[FeatureName][i2][i+NSongs]
            O2 = OtherFeatures[i2][i+NSongs]
            plt.close("all")
            plt.figure(figsize=(48, 16))
            getCSMSmithWatermanScores([F1, O1, F2, O2, Kappa, CSMType], doPlot = True)
            plt.savefig("CSMResults/%i%s.svg"%(i, FeatureName), dpi=200, bbox_inches='tight')
            fh = open("CSMResults/%i.html"%i, "a")
            fh.write("<h2><a name = \"%s\">%s: %s (Tempo Level %i, %i)</a></h2>"%(FeatureName, FeatureName, CSMType, i1, i2))
            fh.write("<img src = \"%i%s.svg\"><BR>"%(i, FeatureName))
            fh.close()

    #Ouput table showing which features got songs correct or not
    csmout = open("CSMResults/index.html", "w")
    csmout.write("<html><body><table>\n<tr><td>Cover Song</td>")
    print "Processing ", AllFeatures.keys()
    for FeatureName in AllFeatures:
        csmout.write("<td>%s</td>"%FeatureName)
    csmout.write("</tr>\n")
    for i in range(NSongs):
        csmout.write("<tr><td>%s</td>"%files[i])
        for FeatureName in AllFeatures:
            Scores = Results[FeatureName]
            idx = np.argmax(Scores[i, NSongs::])
            if idx == i:
                csmout.write("<td><a href = \"%i.html#%s\"><font color = green>Correct</font></a></td>"%(i, FeatureName))
            else:
                csmout.write("<td><a href = \"%i.html#%s\"><font color = red>Incorrect</font></a></td>"%(i, FeatureName))
        csmout.write("</tr>\n")
    csmout.close()


#############################################################################
## Entry points for running the experiments
#############################################################################

if __name__ == '__main__2':
    Kappa = 0.1
    hopSize = 512
    TempoBiases = [60, 120, 180]

    #FeatureParams = {'DPixels':50, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':40, 'D2Samples':40, 'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':50, 'GeodesicDelta':10, 'NGeodesics':400, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    FeatureParams = {'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    #What types of cross-similarity should be used to compare different blocks for different feature types
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    fout = open("results.html", "a")
    doCovers80Experiments(FeatureParams, hopSize, TempoBiases, Kappa, CSMTypes, "Results.mat", fout)
    fout.close()

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180

    index = 16
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    fileprefix = "Covers80%i"%index

    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'
    #fileprefix = "BlurredLines"

    FeatureParams = {'DPixels':200, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':20, 'D2Samples':40, 'MFCCSamplesPerBlock':200, 'GeodesicDelta':10, 'NGeodesic':400, 'lifterexp':0.6, 'MFCCBeatsPerBlock':12, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    #FeatureParams = {'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix)

if __name__ == '__main__':
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180

    index = 16
    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()
    filename1 = "covers32k/" + files1[index] + ".mp3"
    filename2 = "covers32k/" + files2[index] + ".mp3"
    fileprefix = "Covers80%i"%index

    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'
    #fileprefix = "BlurredLines"

    FeatureParams = {'DPixels':200, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':20, 'D2Samples':40, 'MFCCSamplesPerBlock':200, 'GeodesicDelta':10, 'NGeodesic':400, 'lifterexp':0.6, 'MFCCBeatsPerBlock':12, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    #FeatureParams = {'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    print "Getting features for %s..."%filename1
    (XAudio, Fs) = getAudio(filename1)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias1, hopSize)
    (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    print "Getting features for %s..."%filename2
    (XAudio, Fs) = getAudio(filename2)
    (tempo, beats) = getBeats(XAudio, Fs, TempoBias2, hopSize)
    (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats, hopSize, FeatureParams))

    Results = {'filename1':filename1, 'filename2':filename2, 'TempoBias1':TempoBias1, 'TempoBias2':TempoBias2, 'hopSize':hopSize, 'FeatureParams':FeatureParams, 'CSMTypes':CSMTypes, 'Kappa':Kappa}
    plt.figure(figsize=(16, 48))
    for FeatureName in Features1:
        CSM = getCSMType(Features1[FeatureName], O1, Features2[FeatureName], O2, CSMTypes[FeatureName])
        plt.subplot(131)
        plt.imshow(CSM)
        CSM = np.argsort(CSM, 1) + np.argsort(CSM, 0)
        plt.subplot(132)
        plt.imshow(CSM)
        #score = getCSMSmithWatermanScores([Features1[FeatureName], O1, Features2[FeatureName], O2, Kappa, CSMTypes[FeatureName]], True)
        plt.savefig("%s_CSMs_%s.svg"%(fileprefix, FeatureName), dpi=200, bbox_inches='tight')

    sio.savemat("%s.mat"%fileprefix, Results)
