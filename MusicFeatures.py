"""
Programmer: Chris Tralie
Purpose: To provide wrappers around various libraries
(librosa, Essentia, madmom) to compute various features,
in addition to some of my own implementations of these
features.  Features include MFCC, HPCP, CENS, and beat
onsets / tempos
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy.misc
import subprocess
from scipy.io import wavfile
from scipy.signal import spectrogram
from scipy.interpolate import interp1d
from CSMSSMTools import *
import os

def getAudioScipy(filename):
    """
    Wrap around scipy to load audio.  Since scipy only
    loads wav files, call avconv through a subprocess to
    convert any non-wav files to a temporary wav file,
    which is removed after loading:
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    toload = filename
    tempfilename = ""
    if not filename[-3::] == 'wav':
        tempfilename = "%s.wav"%filename[0:-4]
        if os.path.exists(tempfilename):
            os.remove(tempfilename)
        subprocess.call(["avconv", "-i", filename, "-ar", "44100", tempfilename])
        toload = tempfilename
    Fs, XAudio = wavfile.read(toload)
    #Convert shorts to floats
    XAudio = np.array(XAudio, dtype = np.float32) / (2.0**16)
    if len(XAudio.shape) > 1:
        XAudio = np.mean(XAudio, 1)
    if len(tempfilename) > 0:
        os.remove(tempfilename)
    return (XAudio, Fs)

def getAudio(filename):
    """
    Use librosa to load audio
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    import librosa
    XAudio, Fs = librosa.load(filename)
    XAudio = librosa.core.to_mono(XAudio)
    return (XAudio, Fs)

def getMultiFeatureOnsets(XAudio, Fs, hopSize):
    """
    Call Essentia's implemtation of multi feature
    beat tracking
    :param XAudio: Numpy array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    from essentia import Pool, array
    import essentia.standard as ess
    X = array(XAudio)
    b = ess.BeatTrackerMultiFeature()
    beats = b(X)
    print("Beat confidence: ", beats[1])
    beats = beats[0]
    tempo = 60/np.mean(beats[1::] - beats[0:-1])
    beats = np.array(np.round(beats*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getDegaraOnsets(XAudio, Fs, hopSize):
    """
    Call Essentia's implementation of Degara's technique
    :param XAudio: Numpy array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    from essentia import Pool, array
    import essentia.standard as ess
    X = array(XAudio)
    b = ess.BeatTrackerDegara()
    beats = b(X)
    tempo = 60/np.mean(beats[1::] - beats[0:-1])
    beats = np.array(np.round(beats*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getRNNDBNOnsets(filename, Fs, hopSize):
    """
    Call Madmom's implementation of RNN + DBN beat tracking
    :param filename: Path to audio file
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    print("Computing madmom beats...")
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(filename)
    b = proc(act)
    tempo = 60/np.mean(b[1::] - b[0:-1])
    beats = np.array(np.round(b*Fs/hopSize), dtype=np.int64)
    return (tempo, beats)

def getBeats(XAudio, Fs, TempoBias, hopSize, filename = ""):
    """
    Get beat intervals using dynamic programming beat
    tracking with a tempo bias, or if a tempo bias
    isn't specified, use the madmom RNN+DBN implementation
    :param XAudio: Flat numpy array of audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size of each onset function value
    :param filename: Path to audio file
    :returns (tempo, beats): Average tempo, numpy array
        of beat intervals in seconds
    """
    if TempoBias == 0:
        if len(filename) == 0:
            return getDegaraOnsets(XAudio, Fs, hopSize)
        return getRNNDBNOnsets(filename, Fs, hopSize)
    try:
        import librosa
        return librosa.beat.beat_track(XAudio, Fs, start_bpm = TempoBias, hop_length = hopSize)
    except:
        print("Falling back to Degara for beat tracking...")
        if len(filename) == 0:
            return getDegaraOnsets(XAudio, Fs, hopSize)
        return getRNNDBNOnsets(filename, Fs, hopSize)

def getMelFilterbank(Fs, winSize, NSpectrumSamples, NBands = 40, fmin = 0.0, fmax = 8000):
    """
    Return a mel-spaced triangular filterbank
    :param Fs: Audio sample rate
    :param winSize: Window size of associated STFT
    :param NSpectrumSamples: Number of samples in
        assocated spectrogram (related to winSize)
    :param NBands: Number of bands to use
    :param fmin: Minimum frequency
    :param fmax: Maximum frequency
    :returns melfbank: An NBands x NSpectrumSamples matrix
        with each filter per row
    """
    melbounds = np.array([fmin, fmax])
    melbounds = 1125*np.log(1 + melbounds/700.0)
    mel = np.linspace(melbounds[0], melbounds[1], NBands+2)
    binfreqs = 700*(np.exp(mel/1125.0) - 1)
    binbins = np.floor(((winSize-1)/float(Fs))*binfreqs) #Floor to the nearest bin
    binbins = np.array(binbins, dtype=np.int64)

    #Step 2: Create mel triangular filterbank
    melfbank = np.zeros((NBands, NSpectrumSamples))
    for i in range(1, NBands+1):
        thisbin = binbins[i]
        lbin = binbins[i-1]
        rbin = thisbin + (thisbin - lbin)
        rbin = binbins[i+1]
        melfbank[i-1, lbin:thisbin+1] = np.linspace(0, 1, 1 + (thisbin - lbin))
        melfbank[i-1, thisbin:rbin+1] = np.linspace(1, 0, 1 + (rbin - thisbin))
    melfbank = melfbank/np.sum(melfbank, 1)[:, None]
    return melfbank

def getDCTBasis(NDCT, N):
    """
    Return a DCT Type-III basis
    :param NDCT: Number of DCT basis elements
    :param N: Number of samples in signal
    :returns B: An NDCT x N matrix of DCT basis
    """
    ts = np.arange(1, 2*N, 2)*np.pi/(2.0*N)
    fs = np.arange(1, NDCT)
    B = np.zeros((NDCT, N))
    B[1::, :] = np.cos(fs[:, None]*ts[None, :])*np.sqrt(2.0/N)
    B[0, :] = 1.0/np.sqrt(N)
    return B

def getMFCCs(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
    """
    Get MFCC features, my own implementation
    :param XAudio: A flat array of audio samples
    :param Fs: Sample rate
    :param winSize: Window size to use for STFT
    :param hopSize: Hop size to use for STFT (default 512)
    :param NBands: Number of mel bands to use
    :param fmax: Maximum frequency
    :param NMFCC: Number of MFCC coefficients to return
    :param lifterexp: Lifter exponential
    :return X: An (NMFCC x NWindows) array of MFCC samples
    """
    f, t, S = spectrogram(XAudio, nperseg=winSize, noverlap=winSize-hopSize, window='blackman')
    M = getMelFilterbank(Fs, winSize, S.shape[0], NBands, fmax = fmax)

    #Convert STFT to Mel scale
    X = M.dot(np.abs(S))
    #Get log amplitude
    amin = 1e-10
    X = 10*np.log10(np.maximum(amin, X))
    #Do DCT
    B = getDCTBasis(NMFCC, X.shape[0])
    X = np.dot(B, X)
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32)
    return X

def getMFCCsLibrosa(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
    """
    Get MFCC features using librosa functions
    :param XAudio: A flat array of audio samples
    :param Fs: Sample rate
    :param winSize: Window size to use for STFT
    :param hopSize: Hop size to use for STFT (default 512)
    :param NBands: Number of mel bands to use
    :param fmax: Maximum frequency
    :param NMFCC: Number of MFCC coefficients to return
    :param lifterexp: Lifter exponential
    :return X: An (NMFCC x NWindows) array of MFCC samples
    """
    print("Getting MFCCs Librosa...")
    import librosa
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
    X = np.array(X, dtype = np.float32)
    return X


#Norm-preserving square root (as in "chrompwr.m" by Ellis)
def sqrtCompress(X):
    """
    Square root compress chroma bin values
    :param X: An (NBins x NWindows) array of chroma
    :returns Y: An (NBins x NWindows) sqrt normalized
        chroma matrix
    """
    Norms = np.sqrt(np.sum(X**2, 0))
    Norms[Norms == 0] = 1
    Y = (X/Norms[None, :])**0.5
    NewNorms = np.sqrt(np.sum(Y**2, 0))
    NewNorms[NewNorms == 0] = 1
    Y = Y*(Norms[None, :]/NewNorms[None, :])
    return Y

def getHPCPEssentia(XAudio, Fs, winSize, hopSize, squareRoot = False, NChromaBins = 36):
    """
    Wrap around the essentia library to compute HPCP features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param winSize: Window size of each STFT window
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :param NChromaBins: How many chroma bins (default 36)
    :returns H: An (NChromaBins x NWindows) matrix of all
        chroma windows
    """
    import essentia
    from essentia import Pool, array
    import essentia.standard as ess
    print("Getting HPCP Essentia...")
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

def getHPCPJVB(XAudio, Fs, winSize, hopSize, NChromaBins = 36):
    """
    Use Jan Van Balen's HPCP library
    """
    from hpcp_demo.HPCP import hpcp
    return hpcp(XAudio, Fs, winSize, hopSize, bins_per_octave = NChromaBins).T



def getCensFeatures(XAudio, Fs, hopSize, squareRoot = False):
    """
    Wrap around librosa to compute CENs features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :returns X: A (12 x NWindows) matrix of all
        chroma windows
    """
    import librosa
    X = librosa.feature.chroma_cens(y=XAudio, sr=Fs, hop_length = hopSize)
    if squareRoot:
        X = sqrtCompress(X)
    X = np.array(X, dtype = np.float32)
    return X


if __name__ == '__main__2':
    import librosa
    Fs = 44100
    winSize = 22050
    NSpectrumSamples = 22050/2+1
    NBands = 40
    B1 = getMelFilterbank(Fs, winSize, NSpectrumSamples, NBands, fmax = 8000)
    B2 = librosa.filters.mel(Fs, winSize, n_mels = NBands, fmax = 8000)
    plt.subplot(211)
    plt.imshow(B1, cmap = 'afmhot', aspect = 'auto', interpolation = 'none')
    plt.subplot(212)
    plt.imshow(B2, cmap = 'afmhot', aspect = 'auto', interpolation = 'none')
    plt.show()

if __name__ == '__main__':
    import librosa
    XAudio, Fs = librosa.load("piano-chrom.wav")
    XAudio = librosa.core.to_mono(XAudio)
    w = int(np.floor(Fs/4)*2)

    hopSize = 512#8192
    winSize = hopSize*4#8192#16384
    NChromaBins = 12

    H = getHPCPEssentia(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins)
    H2 = getHPCPJVB(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins)
    print H2.shape

    Cens = getCensFeatures(XAudio, Fs, hopSize, squareRoot = True)
    #Cens = librosa.feature.chroma_cens(y=XAudio, sr=Fs)
    #N = int(np.round(len(XAudio)/Fs))*3
    #Cens = scipy.misc.imresize(Cens, (12, N))
    #print Cens.shape

    plt.subplot(311)
    plt.imshow(H, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("HPCP Essentia")
    plt.subplot(312)
    plt.imshow(H2, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("HPCP JVB")
    plt.subplot(313)
    plt.imshow(Cens, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("CENS")
    plt.show()
