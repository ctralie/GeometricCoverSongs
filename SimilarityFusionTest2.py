import numpy as np
import matplotlib.pyplot as plt
from SimilarityFusion import *
from CSMSSMTools import *

if __name__ == '__main__':
    np.random.seed(10)
    N = 40
    NClusters = 3
    K = 5
    NIters = 50
    
    plt.figure(figsize=(5*(NClusters+1), 4))
    
    X = np.zeros((N*NClusters, 2))
    for i in range(NClusters):
        X[i*N:(i+1)*N, :] = 5*np.random.randn(1, 2)
    
    Ds = []
    for i in range(NClusters):
        Xi = np.array(X)
        Xi[i*N:(i+1)*N, :] += np.random.randn(N, 2)
        for k in range(NClusters):
            if k == i:
                continue
            Xi[k*N:(k+1)*N, :] += 3*np.random.randn(N, 2)
        Ds.append(getCSM(Xi, Xi))
    
    for i in range(NClusters):
        plt.subplot(1, NClusters+1, i+1)
        plt.imshow(Ds[i], cmap = 'afmhot', interpolation = 'nearest')
        plt.title("SSM %i"%(i+1))
    
    
    FusedScores = doSimilarityFusion(Ds, K, NIters, 1)
    plt.subplot(1, NClusters+1, NClusters+1)
    plt.imshow(FusedScores, cmap = 'afmhot', interpolation = 'nearest')
    plt.title("Final Affinity Matrix")
    
    plt.savefig("SimilarityFusionExample.svg", bbox_inches = 'tight')
