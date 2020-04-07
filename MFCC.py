"""
Mel-Frequency Cepstral Coefficients (MFCCs).  Including a wrapper
around librosa's implementation
TODO: Look at what has been done in Essentia recently
"""
import numpy as np
from scipy.signal import spectrogram

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
    M = getMelFilterbank(Fs, winSize, S.size, NBands, fmax = fmax)
    #Convert STFT to Mel scale
    XMel = M.dot(np.abs(S))
    #Get log amplitude
    amin = 1e-10
    XMel = 10*np.log10(np.maximum(amin, XMel))
    #Do DCT
    B = getDCTBasis(NMFCC, XMel.shape[0])
    XMFCC = np.dot(B, XMel)
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    XMFCC = coeffs[:, None]*XMFCC
    XMFCC = np.array(XMFCC, dtype = np.float32)
    return {'XMFCC':XMFCC, 'XMel':XMel}

def getMFCCsLowMem(XAudio, Fs, winSize, hopSize = 512, NBands = 40, fmax = 8000, NMFCC = 20, lifterexp = 0):
    """
    Get MFCC features, my own implementation.  Do one STFT window
        at a time
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
    NWin = int(np.floor((len(XAudio)-winSize)/float(hopSize))) + 1
    f, t, S = spectrogram(XAudio[0:winSize], nperseg=winSize, window='blackman')
    M = getMelFilterbank(Fs, winSize, S.shape[0], NBands, fmax = fmax)
    B = np.array(getDCTBasis(NMFCC, NBands), dtype = np.float32)
    XMel = np.zeros((NBands, NWin), dtype = np.float32)
    XMFCC = np.zeros((NMFCC, NWin), dtype = np.float32)
    amin = 1e-10
    #Do STFT window by window, convert to mel scale, and do DCT
    for i in range(NWin):
        f, t, S = spectrogram(XAudio[i*hopSize:i*hopSize+winSize], nperseg=winSize, window='blackman')
        XMel[:, i] = M.dot(np.abs(S)).flatten()
        XMel[:, i] = 10*np.log10(np.maximum(amin, XMel[:, i]))
        XMFCC[:, i] = B.dot(XMel[:, i])
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    XMFCC = coeffs[:, None]*XMFCC
    return {'XMFCC':XMFCC, 'XMel':XMel}

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
    import librosa
    X = librosa.feature.mfcc(XAudio, Fs, n_mfcc=NMFCC)
    #Do liftering
    coeffs = np.arange(NMFCC)**lifterexp
    coeffs[0] = 1
    X = coeffs[:, None]*X
    X = np.array(X, dtype = np.float32)
    return X

if __name__ == '__main__':
    """
    Compare my filterbank to librosa's filterbank
    """
    import matplotlib.pyplot as plt
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
