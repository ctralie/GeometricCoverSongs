import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA

from CSMSSMTools import *

if __name__ == '__main__':
    zeroReturn = False
    N = 100
    t = np.linspace(0, 2*np.pi, N)
    x = np.zeros((N, 2))
#    x[:, 0] = np.cos(t)
#    x[:, 1] = np.sin(t)
    x[:, 0] = (1.5 + np.cos(2*t))*np.cos(t)
    x[:, 1] = (1.5 + np.cos(2*t))*np.sin(t)
    [SSM, _] = getSSM(x, N)

    sigma = 0.5
    K = np.exp(-SSM**2/sigma**2)
    if not zeroReturn:
        np.fill_diagonal(K, np.zeros(K.shape[0])) #Make the diagonal zero

    KSum = np.sum(K, 1)
    KSum[KSum == 0] = 1
    DInvK = np.zeros((N, N))
    np.fill_diagonal(DInvK, 1/KSum)
    DInvK = np.dot(DInvK, K)
    #Make symmetric matrix
    Alpha = K / np.sqrt(KSum[None, :]*KSum[:, None])

    c = plt.get_cmap('jet')
    C = c(np.array(np.round(np.linspace(0, 255, N)), dtype=np.int32))
    C = C[:, 0:3]

    w, v = np.linalg.eig(Alpha)
    plt.plot(np.real(w), 'r')
    plt.hold(True)
    plt.plot(np.imag(w), 'b')
    plt.show()
    w = np.real(w)
    v = np.real(v)
    pca = PCA(n_components=3)
    NEigs = 100
    for t in range(1, 15):
        fig = plt.figure()

        #Use the eigenvector method
        lambdas = w**t
        Y1 = lambdas[None, 0:NEigs]*v[:, 0:NEigs]
        [D1, _] = getSSM(Y1, N)
        Y1 = pca.fit_transform(Y1)

        #Do it out manually
        Y2 = np.eye(N)
        for k in range(t):
            Y2 = Y2.dot(DInvK)
        [D2, _] = getSSM(Y2, N)
        Y2 = pca.fit_transform(Y2)


        ax = fig.add_subplot(231, projection = '3d')
        ax.set_title("t = %i"%t)
        ax.scatter(Y1[:, 0], Y1[:, 1], Y1[:, 2], c=C)
        ax.set_aspect('equal', 'datalim')
        plt.subplot(232)
        plt.imshow(D1)
        plt.title('Eigenvector Method')

        plt.subplot(233)
        plt.scatter(x[:, 0], x[:, 1], c=C)
        plt.title('Original Curve')
        plt.subplot(236)
        plt.imshow(SSM)
        plt.title('Original SSM')

        ax = fig.add_subplot(234, projection = '3d')
        ax.scatter(Y2[:, 0], Y2[:, 1], Y2[:, 2], c=C)
        ax.set_aspect('equal', 'datalim')
        plt.subplot(235)
        plt.imshow(D2)
        plt.title('Brute Force Method')

        plt.savefig("Diffusion%i.svg"%t, bbox_inches='tight')
