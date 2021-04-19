import numpy as np
from numpy.linalg import svd


def rouwenhorst(rho, sigma, N):
    """
    Approximate an AR1 process by a finite markov chain using Rouwenhorst's method.
 
    :param rho: autocorrelation of the AR1 process
    :param sigma: conditional standard deviation of the AR1 process
    :param N: number of states
    :return [nodes, P]: equally spaced nodes and transition matrix
    """
 
    from numpy import sqrt, linspace, array,zeros
 
    sigma = float(sigma)
 
    if N == 1:
      nodes = array([0.0])
      transitions = array([[1.0]])
      return [nodes, transitions]
 
    p = (rho+1)/2
    q = p
    nu = sqrt( (N-1)/(1-rho**2) )*sigma
 
    nodes = linspace( -nu, nu, N)
    n = 1

    mat0 = array([[p,1-p],[1-q,q]])
    if N == 2:
        return [nodes,mat0]
    for n in range(3,N+1):
        mat = zeros( (n,n) )
        mat_A = mat.copy()
        mat_B = mat.copy()
        mat_C = mat.copy()
        mat_D = mat.copy()
        mat_A[:-1,:-1] = mat0
        mat_B[:-1,1:] = mat0
        mat_C[1:,:-1] = mat0
        mat_D[1:,1:] = mat0
 
        mat0 = p*mat_A + (1-p)*mat_B + (1-q)*mat_C + q*mat_D
        mat0[1:-1,:] = mat0[1:-1,:]/2
    P = mat0
    
    # ergodic dist
    ergodic = _find_ergodic(P)
    
    return [nodes, ergodic, P]
 
def _find_ergodic(trans,atol=1e-13,rtol=0):
    """ find ergodic distribution from transition matrix 
    
    Args:

        trans (numpy.ndarray): transition matrix
        atol (double): absolute tolerance
        rtol (double): relative tolerance
    
    Returns:

        (nump.ndarray): ergodic distribution

    """

    I = np.identity(len(trans))
    A = np.atleast_2d(np.transpose(trans)-I)
    _u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T

    return (ns/(sum(ns))).ravel()