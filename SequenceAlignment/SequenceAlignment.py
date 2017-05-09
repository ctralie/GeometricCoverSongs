#Programmer: Chris Tralie
#Purpose: To implement a implicit versions of Smith-Waterman that work on
#a binary cross-similarity matrix

import numpy as np

#Implicit smith waterman align
#Inputs: CSM (a binary N x M cross-similarity matrix)
#Outputs: 1) Distance (scalar)
#2) (N+1) x (M+1) dynamic programming matrix
def swalignimp(CSM, matchScore = 2, mismatchScore = -3, gapScore = -2):
    N = CSM.shape[0]+1
    M = CSM.shape[1]+1
    D = np.zeros((N, M))
    maxD = 0
    for i in range(1, N):
        for j in range(1, M):
            d1 = D[i-1, j] + gapScore
            d2 = D[i, j-1] + gapScore
            d3 = D[i-1, j-1]
            if CSM[i-1, j-1] > 0:
                d3 += matchScore
            else:
                d3 += mismatchScore
            D[i, j] = np.max(np.array([d1, d2, d3, 0.0]))
            if (D[i, j] > maxD):
                maxD = D[i, j]
    return (maxD, D)

#Helper function for swalignimpconstrained
def Delta(a, b, gapOpening = -0.5, gapExtension = -0.7):
    if b > 0:
        return 0
    if b == 0 and a > 0:
        return gapOpening
    return gapExtension

#Helper function for swalignimpconstrained
def Match(i, matchScore = 1, mismatchScore = -1):
    if (i == 0):
        return mismatchScore
    return matchScore

#Implicit smith waterman align with diagonal constraints
#Inputs: CSM (a binary N x M cross-similarity matrix)
#Outputs: 1) Distance (scalar)
#2) (N+1) x (M+1) dynamic programming matrix
def swalignimpconstrained(CSM):
    N = CSM.shape[0]+1
    M = CSM.shape[1]+1
    D = np.zeros((N, M))
    maxD = 0
    for i in range(3, N):
        for j in range(3, M):
            MS = Match(CSM[i-1, j-1])
            #H_(i-1, j-1) + S_(i-1, j-1) + delta(S_(i-2,j-2), S_(i-1, j-1))
            d1 = D[i-1, j-1] + MS + Delta(CSM[i-2, j-2], CSM[i-1, j-1])
            #H_(i-2, j-1) + S_(i-1, j-1) + delta(S_(i-3, j-2), S_(i-1, j-1))
            d2 = D[i-2, j-1] + MS + Delta(CSM[i-3, j-2], CSM[i-1, j-1])
            #H_(i-1, j-2) + S_(i-1, j-1) + delta(S_(i-2, j-3), S_(i-1, j-1))
            d3 = D[i-1, j-2] + MS + Delta(CSM[i-2, j-3], CSM[i-1, j-1])
            D[i, j] = np.max(np.array([d1, d2, d3, 0.0]))
            if (D[i, j] > maxD):
                maxD = D[i, j]
    return (maxD, D)


def SWBacktrace(CSM):
    N = CSM.shape[0]+1
    M = CSM.shape[1]+1
    D = np.zeros((N, M))
    B = np.zeros((N, M), dtype = np.int64) #Backpointer indices
    pointers = [[-1, -1], [-2, -1], [-1, -2], None] #Backpointer directions
    maxD = 0
    maxidx = [0, 0]
    for i in range(3, N):
        for j in range(3, M):
            MS = Match(CSM[i-1, j-1])
            #H_(i-1, j-1) + S_(i-1, j-1) + delta(S_(i-2,j-2), S_(i-1, j-1))
            d1 = D[i-1, j-1] + MS + Delta(CSM[i-2, j-2], CSM[i-1, j-1])
            #H_(i-2, j-1) + S_(i-1, j-1) + delta(S_(i-3, j-2), S_(i-1, j-1))
            d2 = D[i-2, j-1] + MS + Delta(CSM[i-3, j-2], CSM[i-1, j-1])
            #H_(i-1, j-2) + S_(i-1, j-1) + delta(S_(i-2, j-3), S_(i-1, j-1))
            d3 = D[i-1, j-2] + MS + Delta(CSM[i-2, j-3], CSM[i-1, j-1])
            arr = [d1, d2, d3, 0.0]
            D[i, j] = np.max(arr)
            B[i, j] = np.argmax(arr)
            if (D[i, j] > maxD):
                maxD = D[i, j]
                maxidx = [i, j]
    #Backtrace starting at the largest index
    path = [maxidx]
    idx = maxidx
    while B[idx[0], idx[1]] < 3:
        i = B[idx[0], idx[1]]
        idx = [idx[0]+pointers[i][0], idx[1] + pointers[i][1]]
        if idx[0] < 3 or idx[1] < 3:
            break
        path.append(idx)
    return (maxD, D, path)
