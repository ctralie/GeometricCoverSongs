#Programmer: Chris Tralie
#Purpose: To implement similarity network fusion approach described in
#[1] Wang, Bo, et al. "Unsupervised metric fusion by cross diffusion." Computer Vision and Pattern Recognition (CVPR), 2012 IEEE Conference on. IEEE, 2012.
#[2] Wang, Bo, et al. "Similarity network fusion for aggregating data types on a genomic scale." Nature methods 11.3 (2014): 333-337.
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import scipy.io as sio
import time
import os
from EvalStatistics import *

#Affinity matrix
def getW(D, K, Mu = 0.5):
    #W(i, j) = exp(-Dij^2/(mu*epsij))
    DSym = 0.5*(D + D.T)
    np.fill_diagonal(DSym, 0)

    Neighbs = np.sort(DSym, 1)[:, 1:K+1]
    MeanDist = np.mean(Neighbs, 1)
    #Equation 1 in SNF paper [2] for estimating local neighborhood radii
    #by looking at k nearest neighbors, not including point itself
    Eps = MeanDist[:, None] + MeanDist[None, :] + DSym
    Eps = Eps/3
    W = np.exp(-DSym**2/(2*(Mu*Eps)**2))#/(Mu*Eps*np.sqrt(2*np.pi))
    return W

def getWCSMSSM(SSMA, SSMB, CSMAB, Mu = 0.5):
    print "TODO"

#Probability matrix
def getP(W, diagRegularize = False):
    if diagRegularize:
        P = 0.5*np.eye(W.shape[0])
        WNoDiag = np.array(W)
        np.fill_diagonal(WNoDiag, 0)
        RowSum = np.sum(WNoDiag, 1)
        RowSum[RowSum == 0] = 1
        P = P + 0.5*WNoDiag/RowSum[:, None]
        return P
    else:
        RowSum = np.sum(W, 1)
        RowSum[RowSum == 0] = 1
        P = W/RowSum[:, None]
        return P

#Same thing as P but restricted to K nearest neighbors only
#(**note that nearest neighbors here include the element itself)
def getS(W, K):
    N = W.shape[0]
    J = np.argsort(-W, 1)[:, 0:K]
    I = np.tile(np.arange(N)[:, None], (1, K))
    I = I.flatten()
    J = J.flatten()
    V = W[I, J]
    S = sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()
    S = S.toarray()
    SNorm = np.sum(S, 1)
    SNorm[SNorm == 0] = 1
    S = S/SNorm[:, None]
    return S

#Scores: An array of NxN similarity matrices for N songs
#K: Number of nearest neighbors
#NIters: Number of iterations
#reg: Identity matrix regularization parameter for self-similarity promotion
#PlotNames: Strings describing different similarity measurements.
#If this array is specified, an animation will be saved of the cross-diffusion process
def doSimilarityFusion(Scores, K = 5, NIters = 20, reg = 1, PlotNames = []):
    #Affinity matrices
    Ws = [getW(D, K) for D in Scores]
    #Full probability matrices
    Ps = [getP(W) for W in Ws]
    #Nearest neighbor truncated matrices
    Ss = [getS(W, K) for W in Ws]

    #Now do cross-diffusion iterations
    Pts = [np.array(P) for P in Ps]
    N = len(Pts)
    for it in range(NIters):
        if len(PlotNames) == N:
            k = int(np.ceil(np.sqrt(N)))
            for i in range(N):
                res = np.argmax(Pts[i][0:80, 80::], 1)
                res = np.sum(res == np.arange(80))
                plt.subplot(k, k, i+1)
                Im = 1.0*Pts[i]
                Idx = np.arange(Im.shape[0], dtype=np.int64)
                Im[Idx, Idx] = 0
                plt.imshow(Im, interpolation = 'none')
                plt.title("%s: %i/80"%(PlotNames[i], res))
                plt.axis('off')
            plt.savefig("SSMFusion%i.png"%it, dpi=150, bbox_inches='tight')

        nextPts = [np.zeros(P.shape) for P in Pts]
        for i in range(N):
            for k in range(N):
                if i == k:
                    continue
                nextPts[i] += Pts[k]
            nextPts[i] /= float(N-1)
            nextPts[i] = Ss[i].dot(nextPts[i].dot(Ss[i].T))
            if reg > 0:
                nextPts[i] += reg*np.eye(nextPts[i].shape[0])
        Pts = nextPts

    FusedScores = np.zeros(Pts[0].shape)
    for Pt in Pts:
        FusedScores += Pt
    return FusedScores/N

if __name__ == '__main__':
    X = sio.loadmat('Scores4.mat')
    PlotNames = ['ScoresSSMs', 'ScoresHPCP', 'ScoresMFCCs', 'ScoresCENS']
    Scores = [X[s] for s in PlotNames]
    for i in range(len(Scores)):
        Scores[i] = 1.0/Scores[i]

    W = 20

    FusedScores = doSimilarityFusion(Scores, W, 20, 1, [])
    fout = open("resultsFusion.html", "a")
    getCovers80EvalStatistics(FusedScores, 160, 80,  [1, 25, 50, 100], fout, name = "SSMs/MFCCs/HPCP/CENS, 20NN, 1Reg")
    fout.close()
