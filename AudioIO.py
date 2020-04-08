#Audio loading
import numpy as np
import subprocess
from scipy.io import wavfile

def getAudio(filename):
    """
    Wrap around scipy to load audio.  Since scipy only
    loads wav files, call avconv through a subprocess to
    convert any non-wav files to a temporary wav file,
    which is removed after loading:
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    import os
    toload = filename
    tempfilename = ""
    if not filename[-3::] == 'wav':
        tempfilename = '%s.wav'%filename[0:-4]
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

def getAudioLibrosa(filename):
    r"""
    Use librosa to load audio
    :param filename: Path to audio file
    :return (XAudio, Fs): Audio in samples, sample rate
    """
    import librosa
    XAudio, Fs = librosa.load(filename)
    XAudio = librosa.core.to_mono(XAudio)
    return (XAudio, Fs)

