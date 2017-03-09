from BlockWindowFeatures import *

def plotCSM(CSM, beats1, beats2, bratio):
    [I2, J2] = np.meshgrid(np.arange(CSM.shape[1]), np.arange(CSM.shape[0]))
    CSM2 = np.array(CSM)
    CSM2[np.abs(I2 - J2) > 20] = np.inf
    idx = np.unravel_index(np.argmin(CSM2), CSM2.shape)
    print idx
    plt.clf()
    #plt.imshow(CSM, cmap = 'afmhot')
    b1 = [beats1[0]/bratio, beats1[CSM.shape[0]-1]/bratio]
    b2 = [beats2[0]/bratio, beats2[CSM.shape[1]-1]/bratio]
    plt.imshow(CSM, extent = (b2[0], b2[1], b1[1], b1[0]), cmap = 'afmhot')
    plt.hold(True)
    plt.scatter((b2[1] - b2[0])*float(idx[1])/CSM.shape[1], (b1[1] - b1[0])*float(idx[0])/CSM.shape[1], 50)
    return idx

def getSampleSSMs():
    Kappa = 0.1
    hopSize = 512
    TempoBias1 = 180
    TempoBias2 = 180
    DPixels = 400
    BeatsPerBlock = 20
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

    #67 is a good male/female example
    for index in [15, 37, 67, 75]:
        fileprefix = "Covers80%i"%index
        filename1 = "covers32k/" + files1[index] + ".mp3"
        filename2 = "covers32k/" + files2[index] + ".mp3"

        print "Getting features for %s..."%filename1
        (XAudio, Fs) = getAudio(filename1)
        (tempo, beats1) = getBeats(XAudio, Fs, TempoBias1, hopSize)
        (Features1, O1) = getBlockWindowFeatures((XAudio, Fs, tempo, beats1, hopSize, FeatureParams))
        bratio = float(Fs)/hopSize

        print "Getting features for %s..."%filename2
        (XAudio, Fs) = getAudio(filename2)
        (tempo, beats2) = getBeats(XAudio, Fs, TempoBias2, hopSize)
        (Features2, O2) = getBlockWindowFeatures((XAudio, Fs, tempo, beats2, hopSize, FeatureParams))

        #Make SSM CSM
        CSM = getCSM(Features1['SSMs'], Features2['SSMs'])
        idx = plotCSM(CSM, beats1, beats2, bratio)
        plt.savefig("CSM%i_SSM.svg"%index, bbox_inches = 'tight')

        D1 = np.zeros((DPixels, DPixels))
        D1[I < J] = Features1['SSMs'][idx[0]]
        D1 = D1 + D1.T
        D2 = np.zeros((DPixels, DPixels))
        D2[I < J] = Features2['SSMs'][idx[1]]
        D2 = D2 + D2.T
        plt.clf()
        plt.imshow(D1, interpolation = 'none', cmap = 'afmhot')
        plt.savefig("SSM%i_1.svg"%index, bbox_inches = 'tight')
        plt.clf()
        plt.imshow(D2, interpolation = 'none', cmap = 'afmhot')
        plt.savefig("SSM%i_2.svg"%index, bbox_inches = 'tight')

        #Make MFCC CSM
        CSM = getCSM(Features1['MFCCs'], Features2['MFCCs'])
        idx = plotCSM(CSM, beats1, beats2, bratio)
        plt.savefig("CSM%i_MFCC.svg"%index, bbox_inches = 'tight')

        #Make HPCP SSM
        CSM = getCSMCosineOTI(Features1['Chromas'], Features2['Chromas'], O1['ChromaMean'], O2['ChromaMean'])
        idx = plotCSM(CSM, beats1, beats2, bratio)
        plt.savefig("CSM%i_HPCP.svg"%index, bbox_inches = 'tight')

if __name__ == '__main__':
    getSampleSSMs()
