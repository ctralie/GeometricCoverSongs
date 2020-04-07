#Chroma / HPCPs
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
import scipy.sparse as sparse

HPCP_PRECISION = 1e-5
STFT_MIN = 1e-10

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

def getHPCPEssentia(XAudio, Fs, winSize, hopSize, squareRoot=False, NChromaBins=36, NHarmonics = 0):
    """
    Wrap around the essentia library to compute HPCP features
    :param XAudio: A flat array of raw audio samples
    :param Fs: Sample rate
    :param winSize: Window size of each STFT window
    :param hopSize: Hop size between STFT windows
    :param squareRoot: Do square root compression?
    :param NChromaBins: How many chroma bins (default 36)
    :returns H: An (NChromaBins x NWindows) matrix of all \
        chroma windows
    """
    import essentia
    from essentia import Pool, array
    import essentia.standard as ess
    spectrum = ess.Spectrum()
    window = ess.Windowing(size=winSize, type='hann')
    spectralPeaks = ess.SpectralPeaks()
    hpcp = ess.HPCP(size=NChromaBins, harmonics=NHarmonics)
    H = []
    for frame in ess.FrameGenerator(array(XAudio), frameSize=winSize, hopSize=hopSize, startFromZero=True):
        S = spectrum(window(frame))
        freqs, mags = spectralPeaks(S)
        H.append(hpcp(freqs, mags))
    H = np.array(H)
    H = H.T
    if squareRoot:
        H = sqrtCompress(H)
    return H

def get1DPeaks(X, doParabolic=True, MaxPeaks = -1):
    """
    Find peaks in intermediate locations using parabolic interpolation
    :param X: A 1D array in which to find interpolated peaks
    :param doParabolic: Whether to use parabolic interpolation to get refined \
        peak estimates (default True)
    :param MaxPeaks: The maximum number of peaks to consider\
        (default -1, consider all peaks)
    :return (bins, freqs): p is signed interval to the left/right of the max
        at which the true peak resides, and b is the peak value
    """
    idx = np.arange(1, X.size-1)
    idx = idx[(X[idx-1] < X[idx])*(X[idx+1] < X[idx])]
    vals = X[idx]
    if doParabolic:
        #Reference:
        # https://ccrma.stanford.edu/~jos/parshl/Peak_Detection_Steps_3.html
        alpha = X[idx-1]
        beta = X[idx]
        gamma = X[idx+1]
        p = 0.5*(alpha - gamma)/(alpha-2*beta+gamma)
        idx = np.array(idx, dtype = np.float64) + p
        vals = beta - 0.25*(alpha - gamma)*p
    else:
        idx = np.array(idx, dtype = np.float64)
    if MaxPeaks > 0:
        if len(vals) > MaxPeaks:
            idxx = np.argsort(-vals)
            vals = vals[idxx[0:MaxPeaks]]
            idx = idx[idxx[0:MaxPeaks]]
    return (idx, vals)        

def unitMaxNorm(x):
    m = np.max(x)
    if m < HPCP_PRECISION:
        m = 1.0
    return x/m

def getHPCP(XAudio, Fs, winSize, hopSize, NChromaBins = 36, minFreq = 40, maxFreq = 5000, 
            bandSplitFreq = 500, refFreq = 440, NHarmonics = 0, windowSize = 1,
            MaxPeaks = 100, doParabolic = True, dodB = False, squareMags = True):
    """
    My implementation of HPCP
    :param XAudio: The raw audio
    :param Fs: The sample rate
    :param winSize: The window size of each HPCP window in samples
    :param hopSize: The hop size between windows
    :param NChromaBins: The number of semitones for each HPCP window (default 36)
    :param minFreq: Minimum frequency to consider (default 40hz)
    :param maxFreq: Maximum frequency to consider (default 5000hz)
    :param bandSplitFreq: The frequency separating low and high bands (default 500hz)
    :param refFreq: Reference frequency (440hz default)
    :param NHarmonics: The number of harmonics to contribute to each semitone (default 0)
    :param windowSize: Size in semitones of window used for weighting
    :param MaxPeaks: The maximum number of peaks to include per window
    :param doParabolic: Do parabolic interpolation when finding peaks
    :param dodB: Whether to use dB instead of linear magnitudes (default False)
    :param squareMags: Whether to square the linear magnitudes of the contributions
        from the spectrogram
    """
    #Squared cosine weight type

    NWin = int(np.floor((len(XAudio)-winSize)/float(hopSize))) + 1
    binFrac,_,S = spectrogram(XAudio[0:winSize], nperseg=winSize, window='blackman')
    #Setup center frequencies of HPCP
    NBins = int(NChromaBins*np.ceil(np.log2(float(maxFreq)/minFreq)))
    freqs = np.zeros(NBins)
    binIdx = -1*np.ones(NBins)
    for i in range(NChromaBins):
        f = refFreq*2.0**(float(i)/NChromaBins)
        while f > minFreq*2:
            f /= 2.0
        k = i
        while f <= maxFreq:
            freqs[k] = f
            binIdx[k] = i
            k += NChromaBins
            f *= 2.0
    freqs = freqs[binIdx >= 0]
    binIdx = binIdx[binIdx >= 0]
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    binIdx = binIdx[idx]
    freqsNorm = freqs/Fs #Normalize to be fraction of sampling frequency
    maxFreqIdx = int(np.ceil(winSize*float(maxFreq)/Fs))
    minFreqIdx = int(np.floor(winSize*float(minFreq)/Fs))

    #Do STFT window by window
    H = []
    for i in range(NWin):
        #Compute spectrogram and pull out relevant portions
        _,_,S = spectrogram(XAudio[i*hopSize:i*hopSize+winSize], nperseg=winSize, window='blackman')
        S = S.flatten()
        S = S[0:maxFreqIdx]
        
        if dodB:
            #Convert to dB
            S = np.maximum(S, STFT_MIN)
            S = np.log(S)
        
        #Do parabolic interpolation on each peak
        (pidxs, pvals) = get1DPeaks(S, doParabolic=doParabolic, MaxPeaks=MaxPeaks)
        pidxs /= float(winSize) #Normalize to be fraction of sampling frequency
        
        #Figure out number of semitones from each unrolled bin
        ratios = pidxs[:, None]/freqsNorm[None, :]
        ratios[ratios == 0] = 1
        delta = np.abs(np.log2(ratios))*NChromaBins
        
        #Weight by squared cosine window
        weights = (np.cos((delta/windowSize)*np.pi/2)**2)*(delta <= windowSize)
        pvals = pvals[:, None]*weights
        if squareMags:
            pvals = pvals**2
        hpcpUnrolled = np.sum(pvals, 0)
        
        #Make hpcp low and hpcp high
        hpcplow = hpcpUnrolled[freqs <= minFreq]
        binIdxLow = binIdx[freqs <= minFreq]
        hpcplow = sparse.coo_matrix((hpcplow, (np.zeros(binIdxLow.size), binIdxLow)), 
            shape=(1, NChromaBins)).todense()
        hpcphigh = hpcpUnrolled[freqs > minFreq]
        binIdxHigh = binIdx[freqs > minFreq]
        hpcphigh = sparse.coo_matrix((hpcphigh, (np.zeros(binIdxHigh.size), binIdxHigh)), 
            shape=(1, NChromaBins)).todense()
        
        #unitMax normalization of low and high individually, then sum
        hpcp = unitMaxNorm(hpcplow) + unitMaxNorm(hpcphigh)
        hpcp = unitMaxNorm(hpcp)
        hpcp = np.array(hpcp).flatten()
        H.append(hpcp.tolist())
    H = np.array(H)
    H = H.T
    return H


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
    np.random.seed(10)
    x = np.random.randn(100)
    (idx, vals) = get1DPeaks(x)
    plt.plot(np.arange(len(x)), x)
    plt.scatter(idx, vals)
    plt.show()

if __name__ == '__main__':
    """
    Compare my HPCP features to Essentia's HPCP Features
    """
    import time
    from AudioIO import getAudio
    import cProfile
    XAudio, Fs = getAudio("piano-chrom.wav")
    #XAudio, Fs = getAudio("MJ.wav")
    #XAudio = XAudio[Fs*10:Fs*15]

    hopSize = 512#8192
    winSize = 8192#16384
    NChromaBins = 12

    tic = time.time()
    H = getHPCPEssentia(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins)
    print("Elapsed time Essentia: %g"%(time.time() - tic))
    tic = time.time()
    cProfile.run('H2 = getHPCP(XAudio, Fs, winSize, hopSize, NChromaBins = NChromaBins, windowSize = 1, MaxPeaks = 10)')
    print("Elapsed time mine: %g"%(time.time() - tic))
    M = min(H.shape[1], H2.shape[1])
    H = H[:, 0:M]
    H2 = H2[:, 0:M]

    Cens = getCensFeatures(XAudio, Fs, hopSize, squareRoot = True)

    plt.subplot(311)
    plt.imshow(H, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("HPCP Essentia")
    plt.subplot(312)
    plt.imshow(H2, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("My HPCP")
    plt.subplot(313)
    plt.imshow(Cens, cmap = 'afmhot', interpolation = 'none', aspect = 'auto')
    plt.title("Cens")
    plt.show()
