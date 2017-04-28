#Evaluation Statistics / Results Web Page Generation
import numpy as np
import scipy.io as sio

def getCovers80EvalStatistics(ScoresParam, N, NSongs, topsidx, fout, name = "default"):
    Scores = np.array(ScoresParam)
    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf)
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    for i in range(N):
        cover = (i+NSongs)%N #The index of the correct song
        for k in range(N):
            if idx[i, k] == cover:
                ranks[i] = k+1
                break
    print(ranks)
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print("MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR))
    fout.write("<tr><td>%s</td><td>%g</td><td>%g</td><td>%g</td>"%(name, MR, MRR, MDR))
    tops = np.zeros(len(topsidx))
    for i in range(len(tops)):
        tops[i] = np.sum(ranks <= topsidx[i])
        print("Top-%i: %i"%(topsidx[i], tops[i]))
        fout.write("<td>%i</td>"%tops[i])

    #Covers80 score
    Scores = Scores[0:NSongs, NSongs::]
    idx = np.argmax(Scores, 1)
    score = np.sum(idx == np.arange(len(idx)))
    print("Covers80 Score: %i / %i"%(score, NSongs))
    fout.write("<td>%i/%i</td></tr>\n\n"%(score, NSongs))

    return (MR, MRR, MDR, tops)

def getEvalStatistics(ScoresParam, Ks, topsidx, fout, name):
    Scores = np.array(ScoresParam)
    N = Scores.shape[0]
    #Compute MR, MRR, MAP, and Median Rank
    #Fill diagonal with -infinity to exclude song from comparison with self
    np.fill_diagonal(Scores, -np.inf)
    idx = np.argsort(-Scores, 1) #Sort row by row in descending order of score
    ranks = np.zeros(N)
    startidx = 0
    kidx = 0
    for i in range(N):
        if i >= startidx + Ks[kidx]:
            startidx += Ks[kidx]
            kidx += 1
        print startidx
        for k in range(N):
            diff = idx[i, k] - startidx
            if diff >= 0 and diff < Ks[kidx]:
                ranks[i] = k+1
                break
    print(ranks)
    MR = np.mean(ranks)
    MRR = 1.0/N*(np.sum(1.0/ranks))
    MDR = np.median(ranks)
    print("MR = %g\nMRR = %g\nMDR = %g\n"%(MR, MRR, MDR))
    fout.write("<tr><td>%s</td><td>%g</td><td>%g</td><td>%g</td>"%(name, MR, MRR, MDR))
    tops = np.zeros(len(topsidx))
    for i in range(len(tops)):
        tops[i] = np.sum(ranks <= topsidx[i])
        print("Top-%i: %i"%(topsidx[i], tops[i]))
        fout.write("<td>%i</td>"%tops[i])
    fout.write("</tr>\n\n")
    return (MR, MRR, MDR, tops)

def getCovers1000Ks():
    import glob
    Ks = []
    for i in range(1, 396):
        songs = glob.glob("Covers1000/%i/*.txt"%i)
        Ks.append(len(songs))
    return Ks

if __name__ == '__main__':
    Scores = sio.loadmat("Covers1000Results.mat")
    Ks = getCovers1000Ks()
    fout = open("Covers1000Results.html", "a")
    for FeatureName in ['MFCCs', 'SSMs', 'Chromas', 'SNF']:
        S = Scores[FeatureName]
        S = np.maximum(S, S.T)
        getEvalStatistics(S, Ks, [1, 25, 50, 100], fout, FeatureName)
    fout.close()

if __name__ == '__main__2':
    Scores = sio.loadmat("SHSResults.mat")
    SHSIDs = sio.loadmat("SHSIDs.mat")
    Ks = SHSIDs['Ks'].flatten()
    fout = open("SHSDataset/results.html", "a")
    for FeatureName in ['Chromas', 'MFCCs', 'SSMs']:
        getEvalStatistics(Scores[FeatureName], Ks, [1, 25, 50, 100], fout, FeatureName)
    fout.close()
