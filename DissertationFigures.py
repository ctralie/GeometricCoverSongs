from BlockWindowFeatures import *
from Covers80Experiments import *
import scipy.io.wavfile
import librosa

def plotCSM(CSM, artist1, artist2, songName):
    [I2, J2] = np.meshgrid(np.arange(CSM.shape[1]), np.arange(CSM.shape[0]))
    CSM2 = np.array(CSM)
    CSM2[np.abs(I2 - J2) > 300] = np.inf
    idx = np.unravel_index(np.argmin(CSM2), CSM2.shape)
    print idx
    plt.imshow(CSM, cmap = 'afmhot', interpolation = 'nearest')
    plt.hold(True)
    plt.scatter(idx[1], idx[0], 50)
    plt.xlabel(artist2 + " Block Index")
    plt.ylabel(artist1 + " Block Index")
    plt.title("CSM " + songName)
    return idx

def getSampleSSMs():
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180
    DPixels = 400
    BeatsPerBlock = 8
    p = np.arange(DPixels)
    [I, J] = np.meshgrid(p, p)

    FeatureParams = {'MFCCBeatsPerBlock':BeatsPerBlock, 'MFCCSamplesPerBlock':200, 'DPixels':DPixels, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'CurvsSS':'Euclidean', 'TorsSS':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}

    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()

    cmap = 'Spectral'

    #67 is a good male/female example
    for index in [21]:
        fileprefix = "Covers80%i"%index
        filename1 = "covers32k/" + files1[index] + ".mp3"
        filename2 = "covers32k/" + files2[index] + ".mp3"
        artist1 = getArtistName(files1[index])
        artist2 = getArtistName(files2[index])
        songName = getSongName(files1[index])

        print "Getting features for %s..."%filename1
        (XAudio1, Fs1) = getAudio(filename1)
        (tempo, beats1) = getBeats(XAudio1, Fs1, TempoBias1, hopSize)
        (Features1, O1) = getBlockWindowFeatures((XAudio1, Fs1, tempo, beats1, hopSize, FeatureParams))
        bRatio1 = float(Fs1)/hopSize

        print "Getting features for %s..."%filename2
        (XAudio2, Fs2) = getAudio(filename2)
        (tempo, beats2) = getBeats(XAudio2, Fs2, TempoBias2, hopSize)
        (Features2, O2) = getBlockWindowFeatures((XAudio2, Fs2, tempo, beats2, hopSize, FeatureParams))
        bRatio2 = float(Fs2)/hopSize

        #Make SSM CSM
        plt.figure()
        CSM = getCSM(Features1['SSMs'], Features2['SSMs'])
        idx = plotCSM(CSM, artist1, artist2, songName)
        plt.savefig("DissertationFigures/CSM%i_SSM.svg"%index, bbox_inches = 'tight')

        D1 = np.zeros((DPixels, DPixels))
        D1[I < J] = Features1['SSMs'][idx[0]]
        D1 = D1 + D1.T
        t1l = beats1[idx[0]]/bRatio1
        t1r = beats1[idx[0]+BeatsPerBlock]/bRatio1
        s1 = beats1[idx[0]]*hopSize
        s2 = beats1[idx[0]+BeatsPerBlock]*hopSize
        x1 = XAudio1[s1:s2]
        scipy.io.wavfile.write("DissertationFigures/%i_1.wav"%index, Fs1, x1)
        
        D2 = np.zeros((DPixels, DPixels))
        D2[I < J] = Features2['SSMs'][idx[1]]
        D2 = D2 + D2.T
        t2l = beats2[idx[1]]/bRatio2
        t2r = beats2[idx[1]+BeatsPerBlock]/bRatio2
        s1 = beats2[idx[1]]*hopSize
        s2 = beats2[idx[1]+BeatsPerBlock]*hopSize
        x2 = XAudio2[s1:s2]
        scipy.io.wavfile.write("DissertationFigures/%i_2.wav"%index, Fs2, x2)
        
        #Plot spectrograms
        plt.clf()
        plt.figure(figsize=(12, 5))
        plt.subplot(211)
        S1 = librosa.logamplitude(np.abs(librosa.stft(x1)))
        librosa.display.specshow(S1, x_axis='time', y_axis='log')
        plt.subplot(212)
        S2 = librosa.logamplitude(np.abs(librosa.stft(x2)))
        librosa.display.specshow(S2, x_axis='time', y_axis='log')
        plt.savefig("DissertationFigures/Spectrograms%i.svg"%index, bbox_inches='tight')
        
        
        #Plot SSMs
        plt.clf()
        plt.subplot(121)
        plt.title(artist1)
        plt.imshow(D1, interpolation = 'nearest', cmap = cmap, extent = (t1l, t1r, t1l, t1r))
        plt.xlabel("Time (sec)")
        plt.ylabel("Time (sec)")
        plt.subplot(122)
        plt.title(artist2)
        plt.imshow(D2, interpolation = 'nearest', cmap = cmap, extent = (t2l, t2r, t2l, t2r))
        plt.xlabel("Time (sec)")
        plt.ylabel("Time (sec)")
        plt.savefig("DissertationFigures/SSMs%i.svg"%index, bbox_inches = 'tight')
        
        #Make HPCP CSM
        off1 = 400
        off2 = 700
        F1 = Features1['Chromas'][off1:off1+200]
        F2 = Features2['Chromas'][off2:off2+200]
        CSM = getCSMType(F1, O1, F2, O2, 'CosineOTI')
        idx = plotCSM(CSM, artist1, artist2, songName)
        plt.savefig("DissertationFigures/CSM%i_HPCP.svg"%index, bbox_inches = 'tight')
        
        #Plot HPCP Blocks
        plt.clf()
        HPCP1 = Features1['Chromas'][idx[0] + off1]
        HPCP2 = Features2['Chromas'][idx[1] + off2]
        HPCP1 = np.reshape(HPCP1, [len(HPCP1)/12, 12])
        HPCP2 = np.reshape(HPCP2, [len(HPCP2)/12, 12])
        plt.subplot(211)
        librosa.display.specshow(HPCP1.T, y_axis = 'chroma')
        plt.title("HPCP %s"%artist1)
        plt.subplot(212)
        librosa.display.specshow(HPCP2.T, y_axis = 'chroma')
        plt.title("HPCP %s"%artist2)
        plt.savefig("DissertationFigures/HPCP_%i.svg"%index, bbox_inches = 'tight')
        

def getFalseCoversPair():
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180
    index1 = 6
    index2 = 62

    fin = open('covers32k/list1.list', 'r')
    files1 = [f.strip() for f in fin.readlines()]
    fin.close()
    fin = open('covers32k/list2.list', 'r')
    files2 = [f.strip() for f in fin.readlines()]
    fin.close()

    filename1 = "covers32k/" + files1[index1] + ".mp3"
    filename2 = "covers32k/" + files2[index2] + ".mp3"
    fileprefix = "Covers80_%i_%i"%(index1, index2)

    artist1 = getArtistName(files1[index1])
    artist2 = getArtistName(files2[index2])
    songName1 = getSongName(files1[index1])
    songName2 = getSongName(files2[index2])

    #filename1 = 'MIREX_CSIBSF/GotToGiveItUp.mp3'
    #filename2 = 'MIREX_CSIBSF/BlurredLines.mp3'
    #fileprefix = "BlurredLines"

    #FeatureParams = {'DPixels':200, 'NCurv':400, 'NJump':400, 'NTors':400, 'D2Samples':50, 'CurvSigma':20, 'D2Samples':40, 'MFCCSamplesPerBlock':200, 'GeodesicDelta':10, 'NGeodesic':400, 'lifterexp':0.6, 'MFCCBeatsPerBlock':12, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}
    #FeatureParams = {'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40, 'DPixels':200, 'MFCCBeatsPerBlock':20}

    CurvSigmas = [10, 60]
    FeatureParams = {'MFCCBeatsPerBlock':20, 'MFCCSamplesPerBlock':200, 'DPixels':50, 'ChromaBeatsPerBlock':20, 'ChromasPerBlock':40}

    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'SSMsDiffusion':'Euclidean', 'Geodesics':'Euclidean', 'Jumps':'Euclidean', 'Curvs':'Euclidean', 'Tors':'Euclidean', 'CurvsSS':'Euclidean', 'TorsSS':'Euclidean', 'D2s':'EMD1D', 'Chromas':'CosineOTI'}
    for sigma in CurvSigmas:
        CSMTypes['Jumps%g'%sigma] = 'Euclidean'
        CSMTypes['Curvs%g'%sigma] = 'Euclidean'
        CSMTypes['Tors%g'%sigma] = 'Euclidean'

    compareTwoSongs(filename1, TempoBias1, filename2, TempoBias2, hopSize, FeatureParams, CSMTypes, Kappa, fileprefix, songName1, songName2)

if __name__ == '__main__':
    getSampleSSMs()
    #getFalseCoversPair()
