import numpy as np
import scipy.io as sio
import time
from CSMSSMTools import *
from BlockWindowFeatures import *

def getSHSIDDict():
    """
    Get the dictionary of IDs to index numbers in
    the features file
    """
    m = {}
    fin = open("SHSDataset/Chromas/msd_keys_mapping.cly")
    for l in fin.readlines():
        l = l.rstrip()
        f = l.split(",")
        m[f[0]] = int(f[1])
    fin.close()
    return m

def getSHSCliques():
    """
    Return a dictionary of cliques of index numbers
    """
    m = getSHSIDDict()
    fin = open("SHSDataset/Chromas/shs_nodup.txt")
    cliques = {}
    currClique = ""
    for l in fin.readlines():
        l = l.rstrip()
        if l[0] == '%':
            currClique = l[1::]
            cliques[currClique] = []
        else:
            cliques[currClique].append(m[l])
    fin.close()
    return cliques

def getSHSInfo():
    database = {}
    fin = open("SHSDataset/MFCC/info.cly")
    fin.readline()
    while True:
        ID = fin.readline()
        if not ID:
            break
        ID = int(ID)
        artist = fin.readline()
        songname = fin.readline()
        year = int(fin.readline())
        database[ID] = {'artist':artist, 'songname':songname, 'year':year}
    fin.close()
    return database

def loadSHSChromas(IDs):
    """
    Load all of the 12-dim chroma features
    """
    fin = open("SHSDataset/Chromas/btchromas.cly")
    fin.readline() #First line is 'chroma'
    chromas = {}
    while True:
        ID = fin.readline()
        if not ID:
            break
        ID = int(ID)
        if ID%1000 == 0:
            print "Loaded chromas for %i songs..."%ID
        if not ID in IDs:
            fin.readline()
            continue
        x = fin.readline().rstrip()
        x = np.array([float(a) for a in x.split(",")])
        x = np.reshape(x, (len(x)/12, 12))
        chromas[ID] = x
    fin.close()
    return chromas

def loadSHSMFCCs(IDs):
    """
    Load all of the 12-dim MFCC features
    """
    IDDict = getSHSIDDict()
    fin = open("SHSDataset/MFCC/bt_aligned_mfccs_shs.txt")
    mfccs = {}
    count = 0
    while True:
        ID = fin.readline().rstrip()
        if not ID:
            break
        ID = IDDict[ID]
        if count%1000 == 0:
            print "Loaded mfccs for %i songs..."%count
        if not ID in IDs:
            fin.readline()
            count += 1
            continue
        x = fin.readline().rstrip()
        x = x.split(",")
        if len(x[-1]) == 0:
            x = x[0:-1]
        x = np.array([float(a) for a in x])
        x = np.reshape(x, (len(x)/12, 12))
        mfccs[ID] = x
        count += 1
    fin.close()
    return mfccs

def getBeatsPerSong():
    C = loadSHSChromas(np.arange(20000))
    BeatsPerSong = np.zeros((len(C)))
    for i in range(len(BeatsPerSong)):
        BeatsPerSong[i] = len(C[i])
    sio.savemat("SHSDataset/BeatsPerSong.mat", {"BeatsPerSong":BeatsPerSong})

def getSHSSubset(N, maxPerClique, minBeats = 100, maxBeats = 1000):
    """
    Get a subset of the SHS dataset with N songs
    formed of cliques of at most size "maxPerClique"
    """
    BeatsPerSong = sio.loadmat("SHSDataset/BeatsPerSong.mat")['BeatsPerSong'].flatten()
    cliques = getSHSCliques()
    keys = cliques.keys()
    idx = np.random.permutation(len(cliques))
    n = 0
    i = 0
    IDs = []
    Ks = []
    while n < N:
        clique = cliques[keys[idx[i]]]
        K = len(clique)
        if K > 4:
            i += 1
            continue
        withinBeatRange = True
        for s in cliques[keys[idx[i]]]:
            if BeatsPerSong[s] < minBeats or BeatsPerSong[s] > maxBeats:
                withinBeatRange = False
                break
        if not withinBeatRange:
            i += 1
            continue
        n += K
        IDs += clique
        Ks += [K]
        i += 1
    return (IDs, Ks)

def getSHSBlockFeatures(c, m, BeatsPerBlock):
    """
    Get normalized blocked chroma, mfcc, and SSM mfcc features
    """
    N = m.shape[0]
    NBlocks = N - BeatsPerBlock + 1
    DPixels = BeatsPerBlock*(BeatsPerBlock-1)/2

    print "N = %i, NBlocks = %i, BeatsPerBlock = %i"%(N, NBlocks, BeatsPerBlock)

    cRet = np.zeros((NBlocks, 12*BeatsPerBlock))
    mRet = np.zeros((NBlocks, 12*BeatsPerBlock))
    dRet = np.zeros((NBlocks, DPixels))

    [I, J] = np.meshgrid(np.arange(BeatsPerBlock), np.arange(BeatsPerBlock))
    for i in range(NBlocks):
        #MFCC Block
        x = m[i:i+BeatsPerBlock, :]
        x = x - np.mean(x, 0)
        #Normalize x
        xnorm = np.sqrt(np.sum(x**2, 1))[:, None]
        xnorm[xnorm == 0] = 1
        xn = x / xnorm
        mRet[i, :] = xn.flatten()
        D = getCSM(xn, xn)
        dRet[i, :] = D[I < J]

        #Chroma Block
        x = c[i:i+BeatsPerBlock, :]
        xnorm = np.sqrt(np.sum(x**2, 1))
        xnorm[xnorm == 0] = 1
        x = x/xnorm[:, None]
        cRet[i, :] = x.flatten()
    BlockFeatures = {'Chromas':cRet, 'SSMs':dRet, 'MFCCs':mRet}
    #BlockFeatures = {'Chromas':cRet, 'SSMs':dRet}
    OtherFeatures = {'ChromaMean':np.mean(c, 0)}
    return (BlockFeatures, OtherFeatures)

def doSHSExperiment(IDs, Ks, CSMTypes, BeatsPerBlock, Kappa):
    mfccs = loadSHSMFCCs(IDs)
    chromas = loadSHSChromas(IDs)
    AllFeatures = [] #{'Chromas':[], 'SSMs':[], 'MFCCs':[]}
    AllOtherFeatures = []

    N = len(IDs)
    tic = time.time()
    for i in range(len(IDs)):
        (BlockFeatures, OtherFeatures) = getSHSBlockFeatures(chromas[IDs[i]], mfccs[IDs[i]], BeatsPerBlock)
        AllFeatures.append(BlockFeatures)
        AllOtherFeatures.append(OtherFeatures)
    print "Elapsed time blocking: ", time.time() - tic

    Results = {}
    for FeatureName in AllFeatures[0]:
        print "Doing %s"%FeatureName
        CSMType = 'Euclidean' #Euclidean comparison by default
        if FeatureName in CSMTypes:
            CSMType = CSMTypes[FeatureName]
        Scores = np.zeros((N, N))
        for i in range(N):
            print "Doing %s %i of %i..."%(FeatureName, i, N)
            Features1 = AllFeatures[i][FeatureName]
            for j in range(i+1, N):
                Features2 = AllFeatures[j][FeatureName]
                Scores[i, j] = getCSMSmithWatermanScores([Features1, AllOtherFeatures[i], Features2, AllOtherFeatures[j], Kappa, CSMType])
        Scores = Scores + Scores.T
        Results[FeatureName] = Scores
        sio.savemat("SHSDataset/SHSScores.mat", Results)

    #Now do similarity fusion
    Scores = np.zeros((N, N))
    NIters = 10
    K = 20
    for i in range(N):
        print "Doing SNF %i of %i..."%(i, N)
        tic = time.time()
        for j in range(i+1, N):
            Scores[i, j] = getCSMSmithWatermanScoresEarlyFusion([AllFeatures[i], AllOtherFeatures[i], AllFeatures[j], AllOtherFeatures[j], Kappa, K, NIters, CSMTypes])
        print "Elapsed Time: ", time.time() - tic
        Results['SNF'] = Scores + Scores.T
        sio.savemat("SHSDataset/SHSScores.mat", Results)

if __name__ == '__main__2':
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}
    BeatsPerBlock = 25
    Kappa = 0.1

    N = 200
    np.random.seed(100)
    (IDs, Ks) = getSHSSubset(N, 4)
    sio.savemat("SHSDataset/SHSIDs.mat", {"IDs":IDs, "Ks":Ks})
    tic = time.time()
    doSHSExperiment(IDs, Ks, CSMTypes, BeatsPerBlock, Kappa)
    print "Elapsed Time All Comparisons: ", time.time() - tic

if __name__ == '__main__':
    CSMTypes = {'MFCCs':'Euclidean', 'SSMs':'Euclidean', 'Chromas':'CosineOTI'}
    BeatsPerBlock = 25
    Kappa = 0.1
    #Similarity fusion parameters
    NIters = 10
    K = 20

    database = getSHSInfo()
    song = "Hips Don't Lie"
    cliques = getSHSCliques()

    fout = open("SHSDataset/songs.txt", "w")
    for s in cliques.keys():
        fout.write("%s\n"%s)
    fout.close()


    c = cliques[song]
    idx1 = c[0]
    idx2 = c[1]
    print database[idx1]
    print database[idx2]

    mfccs = loadSHSMFCCs(c)
    chromas = loadSHSChromas(c)

    (Features1, O1) = getSHSBlockFeatures(chromas[idx1], mfccs[idx1], BeatsPerBlock)
    (Features2, O2) = getSHSBlockFeatures(chromas[idx2], mfccs[idx2], BeatsPerBlock)

    compareTwoFeatureSets({}, Features1, O1, Features2, O2, CSMTypes, Kappa, "cocaine", NIters = NIters, K = K, song1name = database[idx1]['artist'], song2name = database[idx2]['artist'])
