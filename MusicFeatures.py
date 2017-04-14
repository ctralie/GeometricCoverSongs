import numpy as np
import matplotlib.pyplot as plt
import sys
import librosa
import scipy.misc
from CSMSSMTools import *

def getAudio(filename):
    XAudio, Fs = librosa.load(filename)
    XAudio = librosa.core.to_mono(XAudio)
    return (XAudio, Fs)

def getBeats(XAudio, Fs, TempoBias, hopSize):
    if TempoBias == -1:
        return getDegaraOnsets(XAudio, Fs, hopSize)
    try:
        (tempo, beats) = librosa.beat.beat_track(XAudio, Fs, start_bpm = TempoBias, hop_length = hopSize)
    except:
        print("Falling back to Degara for beat tracking...")
        (tempo, beats) = getDegaraOnsets(XAudio, Fs, hopSize)
    return (tempo, beats)

#Call Essentia's implementation of Degara's technique
def getDegaraOnsets(XAudio, Fs, hopSize):
    from essentia import Pool, array
    import essentia.standard as ess
    X = array(XAudio)
    b = ess.BeatTrackerDegara()
    beats = b(X)
    tempo = 60/np.mean(beats[1::] - beats[0:-1])
    beats = np.array(np.round(beats*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getMFCCs(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
    S = librosa.core.stft(XAudio, winSize, hopSize)
    M = librosa.filters.mel(Fs, winSize, n_mels = NBands, fmax = fmax)

    #if usecmp:
    #    #Hynek's magical equal-loudness-curve formula
    #    fsq = M**2
    #    ftmp = fsq + 1.6e5
    #    eql = ((fsq/ftmp)**2)*((fsq + 1.44e6)/(fsq + 9.61e6))

    X = M.dot(np.abs(S))
    X = librosa.core.logamplitude(X)
    X = np.dot(librosa.filters.dct(NMFCC, X.shape[0]), X) #Make MFCC
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    return X

#Norm-preserving square root (as in "chrompwr.m" by Ellis)
def sqrtCompress(X):
    Norms = np.sqrt(np.sum(X**2, 0))
    Norms[Norms == 0] = 1
    Y = (X/Norms[None, :])**0.5
    NewNorms = np.sqrt(np.sum(Y**2, 0))
    NewNorms[NewNorms == 0] = 1
    Y = Y*(Norms[None, :]/NewNorms[None, :])
    return Y

def getHPCPEssentia(XAudio, Fs, winSize, hopSize, squareRoot = False, NChromaBins = 36):
    import essentia
    from essentia import Pool, array
    import essentia.standard as ess
    spectrum = ess.Spectrum()
    window = ess.Windowing(size=winSize, type='hann')
    spectralPeaks = ess.SpectralPeaks()
    hpcp = ess.HPCP(size = NChromaBins)
    H = []
    for frame in ess.FrameGenerator(XAudio, frameSize=winSize, hopSize=hopSize, startFromZero = True):
        S = spectrum(window(frame))
        freqs, mags = spectralPeaks(S)
        H.append(hpcp(freqs, mags))
    H = np.array(H)
    H = H.T
    if squareRoot:
        H = sqrtCompress(H)
    return H

def getCensFeatures(XAudio, Fs, hopSize, squareRoot = False):
    X = librosa.feature.chroma_cens(y=XAudio, sr=Fs, hop_length = hopSize)
    if squareRoot:
        X = sqrtCompress(X)
    return X


if __name__ == '__main__':
    import librosa
    XAudio, Fs = librosa.load("piano-chrom.wav")
    XAudio = librosa.core.to_mono(XAudio)
    w = int(np.floor(Fs/4)*2)

    hopSize = 512#8192
    winSize = hopSize*4#8192#16384

    H = getHPCPEssentia(XAudio, Fs, winSize, hopSize)

    Cens = getCensFeatures(XAudio, Fs, hopSize, squareRoot = True)
    #Cens = librosa.feature.chroma_cens(y=XAudio, sr=Fs)
    #N = int(np.round(len(XAudio)/Fs))*3
    #Cens = scipy.misc.imresize(Cens, (12, N))
    #print Cens.shape

    plt.subplot(211)
    librosa.display.specshow(H.T, y_axis='chroma', x_axis='time')
    plt.subplot(212)
    librosa.display.specshow(Cens, y_axis='chroma', x_axis='time')
    plt.show()
