"""
This file contains basic miscellaneous utility functions used throughout the code.
"""


import numpy as np
import matplotlib.pyplot as plt
from numba import njit

svd = np.linalg.svd
norm = np.linalg.norm
imshow = plt.imshow
plot = plt.plot


def relative_error(A, B, order=None, rnd=300):
    """
    A, B: tensors with the same shape

    Computes the relative error from A to B
    """
    return (norm(A-B, order)/norm(B, order)).round(rnd)


def compare_theta(th1, th2):
    """
    th1, th2: two values of theta (see NOTATIONS.md)

    For each parameter, computes the relative error between th1 and th2.
    """

    return np.array(list(map(lambda x: relative_error(*x), zip(th1, th2))))


@njit
def sqrtm(A):
    """
    A: square positive semi-definite matrix

    Nnumba-compatible implementation of the matrix square root
    """
    va, ve = np.linalg.eigh(A)
    return ve@np.diag(np.sqrt(np.maximum(va, 0)))@ve.T


@njit
def logdifexp(a, b):
    """
    a, b: floats

    Computes log(|exp(a) - exp(b)|)
    """
    a, b = min(a, b), max(a, b)
    if b-a < 20:
        return a + np.log(np.exp(b-a)-1)
    else:
        return b + np.log(1-np.exp(a-b))


@njit
def logsumexp(a):
    """
    a: numpy array of numbers given in logarithmic form

    Numba-compatible implementation of the function scipy.special.logsumexp
    """
    res = -np.inf
    M = np.max(a)
    for x in a:
        res = np.logaddexp(res, x-M)
    return res + M


def log_mean(log_array):
    """
    log_array: numpy array of numbers given in logarithmic form

    Arithmetic mean of the numbers, also in logarithmic form
    """
    return logsumexp(log_array) - np.log(len(log_array))


def log_var(log_array, center=None):
    """
    log_array: numpy array of numbers given in logarithmic form

    Empirical variance of the numbers, also in logarithmic form
    """
    if center is None:
        center = log_mean(log_array)
    centered_array = np.zeros_like(log_array)
    for i in range(len(log_array)):
        centered_array[i] = logdifexp(log_array[i], center)
    return logsumexp(2*centered_array) - np.log(len(log_array))


@njit
def digitize(x, tab):
    """
    x: float number
    tab: array of numbers, sorted in increasing order

    numba-compatible equivalent of np.digitize
    """
    return int(((x-tab)>0).sum())


@njit
def choice(a, p):
    """
    a: numpy array
    p: array of probabilities with same length as a

    Numba-compatible implementation of the function numpy.random.choice
    """
    cp = np.cumsum(p)
    return a[(np.random.rand() < cp).argmax()]


@njit
def orthogonal_complement(A):
    """
    A: list of n vectors in dimension m.

    Returns a list of n-m vectors orthogonal to those of A.
    """
    n, m = A.shape
    _, _, Q = np.linalg.svd(A)
    return Q[n:]


def AR1_spectral_density(series):
    """
    series: 1D numpy array

    Returns the estimated spectral density of an AR1 process fitted on
    series at point 0. This function uses the formula p. 150 in
    Frühwirth-Schnatter 2004, and p. 1316 in Chib 1995 (better explanation).

    Used in src_base.selection.relative_marginal_likelihood_error.
    """
    V = np.var(series)
    M = np.mean(series)
    S = 30 # number of autocorrelations to compute
    r = np.zeros(S)
    for s in range(1,S):
        r[s] = ((series[s:]-M)*(series[:-s]-M) / V).sum()/S
    s = np.arange(S)+1
    return 1 + 2 * np.sum((1-s/(S+1))*r)


def get_y_mat(y):
    """
    y: list of N series of d-dimensional observations

    Returns a (N, M, d) tensor y_mat such that y_mat[i,j,k] gives the k-th coordinate
    for the j-th observation of individual i.
    """
    N = len(y)
    n = list(map(len, y))
    d = len(y[0][0])
    y_mat = np.zeros((N, max(n), d))
    for i in range(N):
        y_mat[i, :n[i]] = y[i]
    return y_mat

def get_t_mat(t):
    """
    t: list of N series of time points

    Returns a (N, M) tensor t_mat such that t_mat[i,j] gives the j-th time point of individual i.
    """
    N = len(t)
    n = list(map(len, t))
    t_mat = np.zeros((N, max(n)))
    for i in range(N):
        t_mat[i, :n[i]] = t[i]
    return t_mat


@njit
def set_numba_seed(seed):
    """
    seed: integer number

    Change the numba random seed.
    """
    np.random.seed(seed)


def set_random_seed(seed):
    """
    seed: integer number

    Set the random seeds of both numba and numpy.
    """
    np.random.seed(seed)
    set_numba_seed(seed)
