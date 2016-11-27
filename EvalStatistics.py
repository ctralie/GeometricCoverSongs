#Evaluation Statistics / Results Web Page Generation
import numpy as np

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
