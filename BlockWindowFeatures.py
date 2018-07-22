"""
Programmer: Chris Tralie
Purpose: To provide highly customizable code to compute features
within blocks
"""
import numpy as np
import sys
import scipy.io as sio
import scipy.misc
from scipy.interpolate import interp1d
from scipy import signal
import time
import matplotlib.pyplot as plt
from CSMSSMTools import *
from CurvatureTools import *
from SpectralMethods import *
import subprocess

def getBlockWindowFeatures(args, XMFCCParam = np.array([]), XChromaParam = np.array([]), do32Bit = True):
    print("Getting Blocked Features...")
    #NOTE: Need to specify hopSize as as parameter so that beat
    #onsets align with MFCC and chroma windows
    #Unpack parameters
    (XAudio, Fs, tempo, beats, hopSize, FeatureParams) = args
    NBeats = len(beats)-1
    winSize = int(np.round((60.0/tempo)*Fs))
    BlockFeatures = {}
    OtherFeatures = {}

    #########################
    #  MFCC-Based Features  #
    #########################
    #Step 1: Determine which features have been specified and allocate space
    usingMFCC = False
    [MFCCSamplesPerBlock, DPixels, NGeodesic, NJump, NCurv, NTors, NJumpSS, NCurvSS, NTorsSS, D2Samples, DiffusionKappa, tDiffusion] = [-1]*12
    #Default parameters
    GeodesicDelta = 10
    CurvSigmas = [40]
    NMFCC = 20
    MFCCBeatsPerBlock = 20
    sigmasSS = np.linspace(1, 40, 10) #Scale space sigmas
    NMFCCBlocks = 0
    lifterexp = 0.6
    if 'NMFCC' in FeatureParams:
        NMFCC = FeatureParams['NMFCC']
        usingMFCC = True
    if 'lifterexp' in FeatureParams:
        lifterexp = FeatureParams['lifterexp']
        usingMFCC = True
    if 'MFCCBeatsPerBlock' in FeatureParams:
        MFCCBeatsPerBlock = FeatureParams['MFCCBeatsPerBlock']
        usingMFCC = True

    NMFCCBlocks = int(NBeats - MFCCBeatsPerBlock)

    if 'MFCCSamplesPerBlock' in FeatureParams:
        MFCCSamplesPerBlock = FeatureParams['MFCCSamplesPerBlock']
        BlockFeatures['MFCCs'] = np.zeros((NMFCCBlocks, MFCCSamplesPerBlock*NMFCC))
    if 'DPixels' in FeatureParams:
        DPixels = FeatureParams['DPixels']
        NPixels = int(DPixels*(DPixels-1)/2)
        [I, J] = np.meshgrid(np.arange(DPixels), np.arange(DPixels))
        BlockFeatures['SSMs'] = np.zeros((NMFCCBlocks, NPixels), dtype = np.float32)
        if 'DiffusionKappa' in FeatureParams:
            DiffusionKappa = FeatureParams['DiffusionKappa']
            BlockFeatures['SSMsDiffusion'] = np.zeros((NMFCCBlocks, NPixels), dtype = np.float32)
        usingMFCC = True
    if 'tDiffusion' in FeatureParams:
        tDiffusion = FeatureParams['tDiffusion']
    if 'sigmasSS' in FeatureParams:
        sigmasSS = FeatureParams['sigmasSS']
        usingMFCC = True
    if 'CurvSigmas' in FeatureParams:
        CurvSigmas = FeatureParams['CurvSigmas']
        usingMFCC = True

    #Geodesic/jump/curvature/torsion
    if 'GeodesicDelta' in FeatureParams:
        GeodesicDelta = FeatureParams['GeodesicDelta']
        usingMFCC = True
    if 'NGeodesic' in FeatureParams:
        NGeodesic = FeatureParams['NGeodesic']
        BlockFeatures['Geodesics'] = np.zeros((NMFCCBlocks, NGeodesic))
        usingMFCC = True
    if 'NJump' in FeatureParams:
        NJump = FeatureParams['NJump']
        for sigma in CurvSigmas:
            BlockFeatures['Jumps%g'%sigma] = np.zeros((NMFCCBlocks, NJump), dtype = np.float32)
        usingMFCC = True
    if 'NCurv' in FeatureParams:
        NCurv = FeatureParams['NCurv']
        for sigma in CurvSigmas:
            BlockFeatures['Curvs%g'%sigma] = np.zeros((NMFCCBlocks, NCurv), dtype = np.float32)
        usingMFCC = True
    if 'NTors' in FeatureParams:
        NTors = FeatureParams['NTors']
        for sigma in CurvSigmas:
            BlockFeatures['Tors%g'%sigma] = np.zeros((NMFCCBlocks, NTors), dtype = np.float32)
        usingMFCC = True

    #Scale space stuff
    if 'NCurvSS' in FeatureParams:
        NCurvSS = FeatureParams['NCurvSS']
        BlockFeatures['CurvsSS'] = np.zeros((NMFCCBlocks, NCurvSS*len(sigmasSS)), dtype = np.float32)
        usingMFCC = True
    if 'NTorsSS' in FeatureParams:
        NTorsSS = FeatureParams['NTorsSS']
        BlockFeatures['TorsSS'] = np.zeros((NMFCCBlocks, NTorsSS*len(sigmasSS)), dtype = np.float32)
        usingMFCC = True
    if 'NJumpSS' in FeatureParams:
        NJumpSS = FeatureParams['NJumpSS']
        BlockFeatures['JumpsSS'] = np.zeros((NMFCCBlocks, NJumpSS*len(sigmasSS)), dtype = np.float32)
        usingMFCC = True


    if 'D2Samples' in FeatureParams:
        D2Samples = FeatureParams['D2Samples']
        BlockFeatures['D2s'] = np.zeros((NMFCCBlocks, D2Samples), dtype = np.float32)
        usingMFCC = True

    #Step 3: Compute Mel-Spaced log STFTs
    XMFCC = np.array([])
    if usingMFCC:
        if XMFCCParam.size == 0:
            from pyMIRBasic.MFCC import getMFCCsLibrosa
            XMFCC = getMFCCsLibrosa(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)
            #XMFCC = getMFCCsLowMem(XAudio, Fs, winSize, hopSize, lifterexp = lifterexp, NMFCC = NMFCC)['XMFCC']
        else:
            XMFCC = XMFCCParam
    else:
        NMFCCBlocks = 0

    #Step 4: Compute MFCC-based features in z-normalized blocks
    for i in range(NMFCCBlocks):
        i1 = beats[i]
        i2 = beats[i+MFCCBeatsPerBlock]
        x = XMFCC[:, i1:i2].T
        #Mean-center x
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm

        #Straight block-windowed MFCC
        if MFCCSamplesPerBlock > -1:
            xnr = scipy.misc.imresize(xn, (MFCCSamplesPerBlock, xn.shape[1]))
            BlockFeatures['MFCCs'][i, :] = xnr.flatten()

        #Compute SSM and D2 histogram
        SSMRes = xn.shape[0]
        if DPixels > -1:
            SSMRes = DPixels
        if DPixels > -1 or D2Samples > -1:
            (DOrig, D) = getSSM(xn, SSMRes)
        if DPixels > -1:
            BlockFeatures['SSMs'][i, :] = D[I < J]
            if DiffusionKappa > -1:
                xDiffusion = getDiffusionMap(DOrig, DiffusionKappa, tDiffusion)
                (_, SSMDiffusion) = getSSM(xDiffusion, SSMRes)
                BlockFeatures['SSMsDiffusion'][i, :] = SSMDiffusion[I < J]

        if D2Samples > -1:
            [IO, JO] = np.meshgrid(np.arange(DOrig.shape[0]), np.arange(DOrig.shape[0]))
            BlockFeatures['D2s'][i, :] = np.histogram(DOrig[IO < JO], bins = D2Samples, range = (0, 2))[0]
            BlockFeatures['D2s'][i, :] = BlockFeatures['D2s'][i, :]/np.sum(BlockFeatures['D2s'][i, :]) #Normalize

        #Compute geodesic distance
        if NGeodesic > -1:
            jump = xn[1::, :] - xn[0:-1, :]
            jump = np.sqrt(np.sum(jump**2, 1))
            jump = np.concatenate(([0], jump))
            geodesic = np.cumsum(jump)
            geodesic = geodesic[GeodesicDelta*2::] - geodesic[0:-GeodesicDelta*2]
            BlockFeatures['Geodesics'][i, :] = signal.resample(geodesic, NGeodesic)

        #Compute velocity/curvature/torsion
        MaxOrder = 0
        if NTors > -1:
            MaxOrder = 3
        elif NCurv > -1:
            MaxOrder = 2
        elif NJump > -1:
            MaxOrder = 1
        if MaxOrder > 0:
            for sigma in CurvSigmas:
                curvs = getCurvVectors(xn, MaxOrder, sigma)
                if MaxOrder > 2 and NTors > -1:
                    tors = np.sqrt(np.sum(curvs[3]**2, 1))
                    BlockFeatures['Tors%g'%sigma][i, :] = signal.resample(tors, NTors)
                if MaxOrder > 1 and NCurv > -1:
                    curv = np.sqrt(np.sum(curvs[2]**2, 1))
                    BlockFeatures['Curvs%g'%sigma][i, :] = signal.resample(curv, NCurv)
                if NJump > -1:
                    jump = np.sqrt(np.sum(curvs[1]**2, 1))
                    BlockFeatures['Jumps%g'%sigma][i, :] = signal.resample(jump, NJump)

        #Compute curvature/torsion scale space
        MaxOrder = 0
        if NTorsSS > -1:
            MaxOrder = 3
        elif NCurvSS > -1:
            MaxOrder = 2
        elif NJumpSS > -1:
            MaxOrder = 1
        if MaxOrder > 0:
            SSImages = getMultiresCurvatureImages(xn, MaxOrder, sigmasSS)
            if len(SSImages) >= 3 and NTorsSS > -1:
                TSS = SSImages[2]
                TSS = scipy.misc.imresize(TSS, (len(sigmasSS), NTorsSS))
                BlockFeatures['TorsSS'][i, :] = TSS.flatten()
            if len(SSImages) >= 2 and NCurvSS > -1:
                CSS = SSImages[1]
                CSS = scipy.misc.imresize(CSS, (len(sigmasSS), NCurvSS))
                #plt.imshow(CSS, interpolation = 'none', aspect = 'auto')
                #plt.show()
                BlockFeatures['CurvsSS'][i, :] = CSS.flatten()
            if len(SSImages) >= 1 and NJumpSS > -1:
                JSS = SSImages[0]
                JSS = scipy.misc.imresize(JSS, (len(sigmasSS), NJumpSS))
                BlockFeatures['JumpsSS'][i, :] = JSS.flatten()


    ###########################
    #  Chroma-Based Features  #
    ###########################
    #Step 1: Figure out which features are requested and allocate space
    usingChroma = False
    NChromaBlocks = 0
    ChromaBeatsPerBlock = 20
    ChromasPerBlock = 40
    NChromaBins = 12
    FTM2D = False #2D Fourier Magnitude coefficients
    if 'ChromaBeatsPerBlock' in FeatureParams:
        ChromaBeatsPerBlock = FeatureParams['ChromaBeatsPerBlock']
        NChromaBlocks = NBeats - ChromaBeatsPerBlock
        usingChroma = True
    if 'ChromasPerBlock' in FeatureParams:
        ChromasPerBlock = FeatureParams['ChromasPerBlock']
        usingChroma = True
    if 'ChromasFTM2D' in FeatureParams:
        FTM2D = FeatureParams['ChromasFTM2D']

    XChroma = np.array([])
    if usingChroma:
        BlockFeatures['Chromas'] = np.zeros((NChromaBlocks, ChromasPerBlock*NChromaBins))
        if FTM2D:
            print("")
            BlockFeatures['ChromasFTM2D'] = np.zeros((NChromaBlocks, ChromasPerBlock*NChromaBins))
        if XChromaParam.size == 0:
            from pyMIRBasic.Chroma import getHPCPEssentia
            #XChroma = getCensFeatures(XAudio, Fs, hopSize)
            tic = time.time()
            XChroma = getHPCP(XAudio, Fs, hopSize*4, hopSize, NChromaBins = NChromaBins)
            #XChroma = getHPCPEssentia(XAudio, Fs, hopSize*4, hopSize, NChromaBins = NChromaBins)
            print("Elapsed Time Chroma: %g"%(time.time() - tic))
        else:
            XChroma = XChromaParam
        print("XChroma.shape = ", XChroma.shape)
        OtherFeatures['ChromaMean'] = np.mean(XChroma, 1)
    for i in range(NChromaBlocks):
        i1 = beats[i]
        i2 = beats[i+ChromaBeatsPerBlock]
        x = XChroma[:, i1:i2].T
        x = scipy.misc.imresize(x, (ChromasPerBlock, x.shape[1]))
        BlockFeatures['Chromas'][i, :] = x.flatten()
        if FTM2D:
            xf = np.fft.fft(x, axis = 1)
            xf = np.abs(xf)
            xf[:, 0] = 0 #Ignore DC
            BlockFeatures['ChromasFTM2D'][i, :] = xf.flatten()
            continue
            plt.subplot(211)
            plt.imshow(x.T, cmap = 'afmhot', aspect = 'auto', interpolation = 'none')
            plt.subplot(212)
            plt.imshow(xf.T, cmap = 'afmhot', aspect = 'auto', interpolation = 'none')
            plt.savefig("2DFTM%i.png"%i, bbox_inches = 'tight')
    if do32Bit:
        for F in BlockFeatures:
            BlockFeatures[F] = np.array(BlockFeatures[F], dtype = np.float32)
    return (BlockFeatures, OtherFeatures)
