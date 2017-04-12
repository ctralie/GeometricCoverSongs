import numpy as np
import scipy.io as sio

def getIDDict():
    m = {}
    fin = open("SHSDataset/Chromas/msd_keys_mapping.cly")
    for l in fin.readlines():
        l = l.rstrip()
        f = l.split(",")
        m[f[0]] = int(f[1])
    fin.close()
    return m

def getCliques():
    m = getIDDict()
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

def loadChromas():
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
        x = fin.readline().rstrip()
        x = np.array([float(a) for a in x.split(",")])
        x = np.reshape(x, (len(x)/12, 12))
        chromas[ID] = x
    fin.close()
    return chromas

def loadMFCCs():
    IDDict = getIDDict()
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

if __name__ == '__main__':
    mfccs = loadMFCCs()
    chromas = loadChromas()
    for i in range(len(mfccs)):
        print len(mfccs[i]), ", ", len(chromas[i])
