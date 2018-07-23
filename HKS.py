import numpy as np 
from scipy import sparse 
from scipy.sparse.linalg import lsqr, cg, eigsh
import matplotlib.pyplot as plt 


def getHeat(eigvalues, eigvectors, t, initialVertices, heatValue = 100.0):
    """
    Simulate heat flow by projecting initial conditions
    onto the eigenvectors of the Laplacian matrix, and then sum up the heat
    flow of each eigenvector after it's decayed after an amount of time t
    Parameters
    ----------
    eigvalues : ndarray (K)
        Eigenvalues of the laplacian
    eigvectors : ndarray (N, K)
        An NxK matrix of corresponding laplacian eigenvectors
        Number of eigenvectors to compute
    t : float
        The time to simulate heat flow
    initialVertices : ndarray (L)
        indices of the verticies that have an initial amount of heat
    heatValue : float
        The value to put at each of the initial vertices at the beginning of time
    
    Returns
    -------
    heat : ndarray (N) holding heat values at each vertex on the mesh
    """
    N = eigvectors.shape[0]
    I = np.zeros(N)
    I[initialVertices] = heatValue
    coeffs = I[None, :].dot(eigvectors)
    coeffs = coeffs.flatten()
    coeffs = coeffs*np.exp(-eigvalues*t)
    heat = eigvectors.dot(coeffs[:, None])
    return heat

def getHKS(eigvalues, eigvectors, ts, scaleinv = True):
    """
    Given a triangle mesh, approximate its curvature at some measurement scale
    by recording the amount of heat that remains at each vertex after a unit impulse
    of heat is applied.  This is called the "Heat Kernel Signature" (HKS)

    Parameters
    ----------
    eigvalues : ndarray (N, 1)
        Eigenvalues of the graph Laplacian
    eigvectors : ndarray (N, N)
        Corresponding array of eigenvectors of the graph Laplacian, where
        each eigenvector is along a column
    ts : ndarray (T, 1)
        The time scales at which to compute the HKS
    scaleinv : boolean
        Whether or not to do the scale invariant version where the log is taken
        and the non-DC magnitude Fourier coefficients are reported
    Returns
    -------
    hks : ndarray (N, T)
        The heat kernel signature at each point, sampled at T time indices
    """
    res = (eigvectors[:, :, None]**2)*np.exp(-eigvalues[None, :, None]*ts[None, None, :])
    res = np.sum(res, 1)
    if scaleinv:
        res = np.log(res)
        res = np.abs(np.fft.fft(res, axis=1))
        res = res[:, 1::]
    return res