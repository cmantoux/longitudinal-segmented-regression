"""
This file contains the functions used to estimate the model parameters via the
MCMC-SAEM algorithm.

It mainly features the following functions:
- init_saem: provides starting values for the parameters and the latent variables,
- S: encodes the E-step of the SAEM algorithm,
- MAP: encodes the M-step of the SAEM algorithm,
- MCMC-SAEM: runs the MCMC-SAEM algorithm.
"""


import numpy as np
from tqdm.auto import *
from numba import njit

import src_common.utils
from src_exp.model import *
from src_exp.mcmc import *


def init_saem(y, t, K, orthogonality_condition=False, heteroscedastic=False):
    """
    y, t: see NOTATIONS.md
    K: number of breaks in the population trajectory,
    orthogonality_condition: if True, psi is initialized orthogonal to v0.
    heteroscedastic: if True, the standard deviations may take different values
                     per coordinate. If False, the standard deviations are the
                     same across coordinates.

    Initialize z and theta for the SAEM.

    Returns:
    z_init: initial value of z
    theta_init: initial value of theta
    """

    N, d = len(y), len(y[0][0])
    n = list(map(len, y))
    ys = np.concatenate(y)
    ts = np.concatenate(t)
    min_ind = np.array([t[i][0] for i in range(N)])
    max_ind = np.array([t[i][-1] for i in range(N)])
    t1 = min_ind.mean()
    t2 = max_ind.mean()
    t0_min = t1 * K/(K+1) + t2 * 1/(K+1)
    t0_max = t1 * 1/(K+1) + t2 * K/(K+1)

    if d==1:
        # Initialize with p0=0 for identifiability
        idx = np.where(1-np.isnan(ys.reshape(-1)))[0]
        ys = np.array([ys[i] for i in idx])
        ts = np.array([ts[i] for i in idx])
        features = np.hstack([ts[:,None], np.ones((len(ts),1))])
        lin, _, _, _ = np.linalg.lstsq(features, ys[:,0], rcond=1)
        v0_init = np.repeat(lin[[0]][None,:], K+1, axis=0)
        t0_init = np.linspace(t0_min, t0_max, max(K,1))
        p0_init = np.array([lin[1] + t0_init[0]*v0_init[0,0]])

    else:
        p0_init = np.zeros(d)
        t0_init = np.linspace(t0_min, t0_max, max(K,1))
        v0_init = np.zeros((K+1, d))
        # Estimate average direction with linear regression ; lin[0] contains the leading direction
        for k in range(d):
            idx = np.where(1-np.isnan(ys[:,k]))[0]
            ys_k = np.array([ys[i,k] for i in idx])
            ts_k = np.array([ts[i] for i in idx])
            features = np.hstack([ts_k[:,None], np.ones((len(ts_k),1))])
            lin, _, _, _ = np.linalg.lstsq(features, ys_k, rcond=1)
            v0_init[:,k] = lin[[0]] # same speed for all pieces
            p0_init[k] = lin[1] + t0_init[0]*v0_init[0,k]

    min_break_ind = min_ind * K/(K+1) + max_ind * 1/(K+1) # individual position for first break
    max_break_ind = min_ind * 1/(K+1) + max_ind * K/(K+1) # individual position for last break
    # tau_init = np.random.randn(N)
    tau_init = (min_break_ind - t0_init[0])
    xi_init = np.random.randn(N,K+1)/10
    # xi_init = np.ones((N,K+1))*(np.log((t0_init[-1] - t0_init[0])/(max_break_ind - min_break_ind + 1e-5)))[:,None]
    if orthogonality_condition:
        A = orthogonal_complement(v0_init)
        psi_dim = d-K-1
        psi_init = np.random.randn(N, psi_dim)@A / 10 # random init to favor exploration
    else:
        psi_dim = d
        psi_init = np.random.randn(N, psi_dim) / 10 # random init to favor exploration

    sigma_tau_init = np.array([tau_init.std()])
    if heteroscedastic:
        sigma_psi_init = psi_init.std(axis=0)
        sigma_xi_init = xi_init.std(axis=0)
        sigma_init = np.ones(d)*0.1
    else:
        sigma_psi_init = np.ones(d) * psi_init.std() * np.sqrt(d/psi_dim)
        sigma_xi_init = np.ones(K+1) * xi_init.std()
        sigma_init = np.ones(d) * 0.1

    theta_init = p0_init, t0_init, v0_init, sigma_xi_init, sigma_tau_init, sigma_psi_init, sigma_init
    z_init = (p0_init, t0_init, v0_init, tau_init, xi_init, psi_init)

    return z_init, theta_init


@njit
def _S0_aux(ys, z, n, t_mat, heteroscedastic=False):
    """
    z, t_mat: see NOTATIONS.md
    ys: list of all observations, concatenated across individuals,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    heteroscedastic: if True, the standard deviations may take different values
                     per coordinate. If False, the standard deviations are the
                     same across coordinates.

    Auxiliary computation function for the sufficient statistics S.
    This auxiliary part allows using numba.

    Returns:
    diff: for each coordinate, gives the list of squared differences between the
          model prediction and the observed data,
    len_diff: for each coordinate, gives the number of observed non-missing
              data points.
    If heteroscedastic is False, the sum over coordinates of these
    quantities is returned instead.
    """

    N, d = len(n), len(ys[0])
    z_full = get_z_full(z)
    Ds = np.zeros((sum(n), d))
    k = 0
    for i in range(len(n)):
        for j in range(n[i]):
            Ds[k] = D_i(z_full, i, t_mat[i,j])
            k += 1
    diff = np.zeros(d)
    len_diff = np.zeros(d)
    for k in range(d):
        nan_mask = np.where(1-np.isnan(ys[:,k]))[0]
        diff[k] = ((ys-Ds)[nan_mask,k]**2).sum()
        len_diff[k] = len(nan_mask)
    if heteroscedastic:
        return diff, len_diff
    else:
        return np.array([diff.sum()]), np.array([len_diff.sum()])


def S(y, z_vect, n, t_mat, K, heteroscedastic=False):
    """
    y, z_vect, t_mat: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    K: number of breaks in the population trajectory,
    heteroscedastic: if True, the standard deviations may take different values
                     per coordinate. If False, the standard deviations are the
                     same across coordinates.

    Computes and returns the sufficient statistics of the observations.
    """

    N, d = len(n), len(y[0][0])
    z = unpack_z(z_vect, d=d, K=K)
    p0, t0, v0, tau, xi, psi = z

    ys = np.concatenate(y)
    S0, S7 = _S0_aux(ys, z, n, t_mat, heteroscedastic)
    if heteroscedastic:
        S1 = (xi**2).sum(axis=0)
    else:
        S1 = np.array([(xi**2).sum()])
    S2 = np.array([(tau**2).sum()])
    S3 = t0
    S4 = v0
    S5 = p0
    if heteroscedastic:
        S6 = (psi**2).sum(axis=0)
    else:
        S6 = np.array([(psi**2).sum()])
    S8 = N
    return np.array([S0, S1, S2, S3, S4, S5, S6, S7, S8], dtype=object)


def MAP(s, prior, orthogonality_condition=False, heteroscedastic=False):
    """
    s: sufficient statistics, as returned by function S,
    prior: see NOTATIONS.md
    orthogonality_condition: if True, psi is initialized orthogonal to v0.
    heteroscedastic: if True, the standard deviations may take different values
                     per coordinate. If False, the standard deviations are the
                     same across coordinates.

    Given prior and a value for the sufficient statistics, returns the result
    of the M-step of the SAEM algorithm.
    """

    sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v = prior
    K, d = len(v0_barbar) - 1, len(p0_barbar)
    psi_dim = d-K-1 if orthogonality_condition else d
    t0_bar = (s[3]/sigma_t0**2 + t0_barbar/sigma_t0_bar**2)/(1/sigma_t0**2 + 1/sigma_t0_bar**2)
    v0_bar = (s[4]/sigma_v0**2 + v0_barbar/sigma_v0_bar**2)/(1/sigma_v0**2 + 1/sigma_v0_bar**2)
    p0_bar = (s[5]/sigma_p0**2 + p0_barbar/sigma_p0_bar**2)/(1/sigma_p0**2 + 1/sigma_p0_bar**2)
    sigma_tau = np.sqrt((s[2] + v[1]**2)/(s[8]+m[1]+2))
    if heteroscedastic:
        sigma_xi = np.sqrt((s[1] + v[0]**2)/(s[8]+m[0]+2))
        sigma_psi = np.sqrt((s[6] + v[2]**2)/(s[8]+m[2]+2)) ### CAUTION (depends whether we use psi or s)
        sigma = np.sqrt((s[0] + v[3]**2)/(s[7]+m[3]+2))
    else:
        sigma_xi = np.sqrt((s[1] + v**2)/((K+1)*s[8]+m+2)) * np.ones(K+1)
        sigma_psi = np.sqrt((s[6] + v**2)/(psi_dim*s[8]+m+2)) * np.ones(d) ### CAUTION (depends whether we use psi or s)
        sigma = np.sqrt((s[0] + v**2)/(s[7]+m+2)) * np.ones(d)

    theta_new = p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma
    return theta_new


def MCMC_SAEM(y, t, prior, n_iter, n_mh=1,
            init=None, prop_mh=0.03, orthogonality_condition=False,
            heteroscedastic=True, track_history=False, verbose=False):
    """
    y, t, prior: see NOTATIONS.md,
    n_iter: number of SAEM steps,
    n_mh: number of Metropolis-Hastings steps per SAEM step,
    init: initial value for (z, theta)
    orthogonality_condition: if True, psi is initialized orthogonal to v0.
    heteroscedastic: if True, the standard deviations may take different values
                     per coordinate. If False, the standard deviations are the
                     same across coordinates.
    track_history: if False, does not store the successive values of z and theta
                   across the convergence. If True, stores every value of z and theta.
                   If given an integer number, stores one value of z and theta
                   every track_history SAEM iterations.
    verbose: if True, prints the MCMC acceptance rates and transition variances
             at the end of the SAEM loop.

    Main function running the MCMC-SAEM algorithm. The algorithm jointly
    computes
    - the Maximum A Posteriori estimator of the exponentialized model,
    - the posterior distribution of the latent variables given the observed data.

    Returns:
    z: final MCMC value of the latent variable
    theta: final value of the model parameters
    history: dictionary storing the evolution of the model's complete likelihood
             (in history["log_likelihood"]), and, optionally, the intermediate
             values of z and theta (in history["z"] and history["theta"]).

    NB: the sequence given by history["log_likelihood"] is not necessarily
        increasing, as it represents log p(y, z | theta). Therefore:
        - It does not include the prior on theta,
        - Even if we stored log p(y, z, theta) instead, the curve would still not
          necessarily increase: the SAEM algorithm aims at increasing the marginal
          likelihood p(y, theta), which is different from p(y, z, theta).
        For these reasons, the values in history["log_likelihood"] should be
        regarded as an additional information rather than a tool to diagnose the
        SAEM convergence.
    """

    # Principle of the MCMC-SAEM algorithm:
    # At each step, we simulate the new sufficient statistics.
    # and compute the new function to maximize using stochastic approx.
    # Then, we maximize it using the MAP formulas.

    burn_in = 2*n_iter//4
    optimal_rate = 0.3 # Target acceptance MH rate
    max_block_size = 50 # Time interval for MH proposal variance update
    # Initialize the MCMC proposal scale used in src_exp.mcmc.mh_z:
    prop_mh = prop_mh * np.array([1,         # scale for p0
                                  0.1,       # scale for t0
                                  0.01,      # scale for v0
                                  10,10,10]) # scale for tau, xi and psi

    N, d, K = len(y), len(y[0][0]), len(prior[5]) - 1
    n = np.array(list(map(len, y)))
    # Specific cases where the orthogonality condition is automatically set:
    if K==0 and not heteroscedastic: # If there is no break, the condition is useful and comes at no cost
        orthogonality_condition = True
    elif d<=K+1: # If there are too many breaks, the condition imposes psi_i = 0
        orthogonality_condition = False

    # pack y and t into tensors and matrices
    y_mat = src_common.utils.get_y_mat(y)
    t_mat = src_common.utils.get_t_mat(t)

    if init is None:
        z, theta = init_saem(y, t, K, orthogonality_condition, heteroscedastic)
    else:
        z, theta = init
    z_vect = pack_z(z)

    S_SAEM = S(y, z_vect, n, t_mat, K)

    history = {}
    history["log_likelihood"] = np.zeros(n_iter)
    if track_history:
        history["theta"] = np.zeros((n_iter//track_history, len(pack_theta(theta))))
        history["z"] = np.zeros((n_iter//track_history, len(z_vect)))

    avg_rate = np.zeros(6)+0.5
    block_size = 0
    block_accepts = np.zeros(6)
    iterator = trange(n_iter)
    for i in iterator:
        # Simulation step : we need to sample from p(z | y, theta)
        # We use Metropolis-Hastings :
        z_vect, accepts, lks_full = mh_z(
            y_mat, t_mat, n, theta, prior, n_iter=n_mh,
            prop=prop_mh, z_vect=z_vect,
            orthogonality_condition=orthogonality_condition)
        block_accepts += accepts
        block_size += n_mh

        # Approximation step : we compute the new sufficient statistic
        if i < burn_in:
            gamma = 1
        else:
            gamma = 1/(i+1-burn_in)**0.7 # sum^2 CV, sum DV

        # Stochastic Approximation
        S_SAEM = (1-gamma)*S_SAEM + gamma*S(y, z_vect, n, t_mat, K, heteroscedastic)

        # Maximization step
        theta = MAP(S_SAEM, prior, orthogonality_condition, heteroscedastic)

        # Adaptive MH variance update
        if block_size > max_block_size:
            g_mh = 3/(i+1)**0.55
            rate = block_accepts / block_size # Compute MCMC acceptance rate
            avg_rate = (1-g_mh)*avg_rate + g_mh*rate # Update averaged rate for display
            D = rate > optimal_rate
            D = 2*D - 1
            approx_sto = np.log(prop_mh) + g_mh * D
            prop_mh = np.exp(approx_sto)
            block_accepts = np.zeros(6)
            block_size = 0

        lk = lks_full.sum()
        history["log_likelihood"][i] = lk

        if int(track_history) > 0 and (i%int(track_history) == 0):
            history["z"][i//track_history] = z_vect.copy()
            history["theta"][i//track_history] = pack_theta(theta)


    if verbose:
        print("MCMC Transition rates:", np.round(avg_rate, 2))
        print("MCMC proposal standard deviation:", prop_mh)

    z = unpack_z(z_vect, d=d, K=K)
    return z, theta, history
