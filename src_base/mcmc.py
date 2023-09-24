"""
This file contains functions performing Markov Chain Monte-Carlo
on variables of the non-exponentialized model. The Metropolis-Hastings within Gibbs
algorithm is used in each case, with custom blocks depending on the target.

Most functions aim at sampling from the posterior distribution of (tau, xi | y), i.e.
the individual variables coding the time shift and accelerations with respect to
the population trajectory. The remaining latent variable (psi) does not need to
be sampled from, as the distribution (psi | tau, xi) is known explicitly.

Unlike in other Metropolis-Hastings-based MCMCs in the code, the proposals for
(xi, tau) are generated with an adaptive covariance matrix, proportional to the
empirical covariance of the previous samples.

More specifically, the functions in this file are:
- mh_tau_xi_ind: MCMC on (tau_i, xi_i | y, theta) for individual i
- mh_tau_xi_ind_mean_cov: slight variation of mh_tau_xi_ind
- mh_tau_xi: MCMC on (tau, xi | y, theta) for all individuals
- pmh_tau_xi: parallelized version of mh_tau_xi
- posterior_z_mean: returns the posterior mean of z=(tau, xi, psi) given y and theta
- mh_v0: MCMC on the distribution (v0 | y, (theta\v0))
"""


import numpy as np
from tqdm.auto import *
from numba import njit

from src_base.model import *
from src_common.utils import *


@njit
def mh_tau_xi_ind(y_mat, t_mat, n, z_vect, theta, i, n_iter, prop=1e-1):
    """
    y_mat, t_mat, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    i: index of the individual,
    n_iter: number of MCMC steps,
    prop: initial scale factor for the transition covariance.

    Adaptive Metropolis-Hastings targetting (tau_i, xi_i | y), i.e. the distribution
    of the temporal latent variables for individual i. The chain starts at position
    z_vect and runs for n_iter iterations.

    Returns:
    result: list of MCMC samples (in z_vect format),
    lks: list of log-likelihood p(xi_i, tau_i, y_i, theta) along the chain,
    rate_z: MCMC acceptance rate,
    sigma_prop_z: final scale factor for the adaptive covariance,
    Mean_z: mean of the MCMC samples,
    Cov_z: empirical covariance of the MCMC samples.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta
    K, N, d = len(theta[2]) - 1, len(y_mat), len(y_mat[0,0])
    block_size = 50
    target_rate = 0.3

    # Variance parameters of the transition kernel
    sigma_prop_z = np.ones((1))*prop   # Scale factor of the proposal covariance
    L_prop       = np.zeros((K+2,K+2)) # Square root of the empirical covariance
    # Variables for empirical covariance estimation
    Cov_z    = np.zeros((K+2,K+2)) # Empirical covariance = E[XX.T] - E[X]E[X].T
    Lambda_z = np.zeros((K+2,K+2)) # E[XX.T]
    M_z      = np.zeros((K+2))     # E[X]
    Lambda_z = np.eye(K+2)
    # Variables for adaptive proposal tuning
    accepts_z       = np.zeros((1))
    block_accepts_z = np.zeros((1))

    ps0 = get_breakpoint_positions(p0, t0, v0)
    theta_full = (ps0, *theta[1:])

    current_log_lk = log_lk_tau_xi_ind(y_mat, t_mat, n, z_vect, theta_full, i)

    idx = np.concatenate((np.array([i]), np.arange(N+(K+1)*i, N+(K+1)*(i+1))))

    result = np.zeros((n_iter, len(idx)))
    lks = np.zeros(n_iter)

    z_vect = z_vect.copy()
    it = range(n_iter)
    for m in it:
        # Generate next move
        old = z_vect[idx].copy()
        if m%10==0:
            Cov_z = Lambda_z/(m+1) - M_z.reshape(-1,1)@M_z.reshape(1,-1) / (m+1)**2
            L_prop = sqrtm(Cov_z)
        z_vect[idx] = z_vect[idx] + sigma_prop_z * L_prop @ np.random.randn(len(idx))
        # Compute the acceptance log-probability
        new_log_lk = log_lk_tau_xi_ind(y_mat, t_mat, n, z_vect, theta_full, i)
        log_alpha = new_log_lk - current_log_lk

        if np.log(np.random.rand()) < log_alpha:
            current_log_lk = new_log_lk
            accepts_z += 1
            block_accepts_z += 1
        else:
            z_vect[idx] = old

        cov_sample = z_vect[idx].reshape(-1,1)@z_vect[idx].reshape(1,-1)
        Lambda_z += cov_sample
        M_z += z_vect[idx]

        # Adaptive MH rate
        if m>1 and m%block_size == 0:
            d_z = (block_accepts_z / block_size) > target_rate
            d_z = 2*d_z - 1
            sigma_prop_z = np.exp(np.log(sigma_prop_z) + d_z * 3/(m+1)**0.3)
            block_accepts_z *= 0

        result[m] = z_vect[idx]
        lks[m] = current_log_lk

    rate_z = accepts_z/n_iter
    Mean_z = M_z/n_iter
    return result, lks, rate_z, sigma_prop_z, Mean_z, Cov_z


@njit
def mh_tau_xi_ind_mean_cov(y_mat, t_mat, n, z_vect, theta, i, n_iter, prop=1e-1):
    """
    y_mat, t_mat, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    i: index of the individual,
    n_iter: number of MCMC steps,
    prop: initial scale factor for the transition covariance.

    Computes the sample mean and covariance of (tau_i, xi_i | y, theta) online
    with a covariance adaptive MCMC, as in src_base.mcmc.mh_tau_xi_ind.

    Returns:
    - Mean_z: empirical mean of MCMC samples,
    - Coz_z: empirical covariance of MCMC samples.

    NB: The code in this function only slightly differs from mh_tau_xi_ind, but
    it has to be written in a separate function to comply with the numba function
    format requirements.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta
    K, N, d = len(theta[2]) - 1, len(y_mat), len(y_mat[0,0])
    block_size = 50
    target_rate = 0.3

    # Variance parameters of the transition kernel
    sigma_prop_z    = np.ones((1))*prop
    L_prop = np.zeros((K+2,K+2))
    # Variables for empirical covariance estimation
    Cov_z    = np.zeros((K+2,K+2)) # Empirical covariance = E[XX.T] - E[X]E[X].T
    Lambda_z = np.zeros((K+2,K+2)) # E[XX.T]
    M_z      = np.zeros((K+2))     # E[X]
    Lambda_z = np.eye(K+2)
    # Variables for adaptive proposal tuning
    accepts_z       = np.zeros((1))
    block_accepts_z = np.zeros((1))

    ps0 = get_breakpoint_positions(p0, t0, v0)
    theta_full = (ps0, *theta[1:])

    current_log_lk = log_lk_tau_xi_ind(y_mat, t_mat, n, z_vect, theta_full, i)

    idx = np.concatenate((np.array([i]), np.arange(N+(K+1)*i, N+(K+1)*(i+1))))

    z_vect = z_vect.copy()
    it = range(n_iter)
    for m in it:
        # Generate next move
        old = z_vect[idx].copy()
        if m%10==0:
            Cov_z = Lambda_z/(m+1) - M_z.reshape(-1,1)@M_z.reshape(1,-1) / (m+1)**2
            L_prop = sqrtm(Cov_z)
        z_vect[idx] = z_vect[idx] + sigma_prop_z * L_prop @ np.random.randn(len(idx))
        # Compute the acceptance log-probability
        new_log_lk = log_lk_tau_xi_ind(y_mat, t_mat, n, z_vect, theta_full, i)
        log_alpha = new_log_lk - current_log_lk

        if np.log(np.random.rand()) < log_alpha:
            current_log_lk = new_log_lk
            accepts_z += 1
            block_accepts_z += 1
        else:
            z_vect[idx] = old

        cov_sample = z_vect[idx].reshape(-1,1)@z_vect[idx].reshape(1,-1)
        Lambda_z += cov_sample
        M_z += z_vect[idx]

        # Adaptive MH rate
        if m>1 and m%block_size == 0:
            d_z = (block_accepts_z / block_size) > target_rate
            d_z = 2*d_z - 1
            sigma_prop_z = np.exp(np.log(sigma_prop_z) + d_z * 3/(m+1)**0.3)
            block_accepts_z *= 0

    Mean_z = M_z/n_iter
    Cov_z = Lambda_z/n_iter - M_z.reshape(-1,1)@M_z.reshape(1,-1) / n_iter**2
    return Mean_z, Cov_z


def mh_tau_xi(y_mat, t_mat, n, z_vect, theta, n_iter, prop=1e-1, only_mean_cov=False, verbose=True):
    """
    y_mat, t_mat, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    n_iter: number of MCMC steps,
    prop: initial scale factor for the transition covariance,
    only_mean_cov: if True, discard the MCMC values on the fly to only retain the
                   sample variance and covariance,
    verbose: if True, displays a progress bar for the loop on individuals.

    Adaptive Metropolis-Hastings chain on (tau, xi | y). This function sequentially
    calls src_base.mcmc.mh_tau_xi_ind for each value of i.

    Returns:
    z_mh: list of MCMC samples (in z_vect format),
    lks_mh_post: list of log-likelihood p(xi, tau, y, theta) along the chain,
    rate_z: MCMC acceptance rate,
    sigma_prop_z: final scale factor for the adaptive covariance,
    Mean_z: mean of the MCMC samples,
    Cov_z: empirical covariance of the MCMC samples.
    """

    K, N, d = len(theta[2]) - 1, len(y_mat), len(y_mat[0,0])

    if only_mean_cov:
        Mean_z = np.zeros((N, K+2))
        Cov_z = np.zeros((N, K+2, K+2))
        iterator = range(N) if not verbose else trange(N)
        for i in iterator:
            Mean_z[i], Cov_z[i] = mh_tau_xi_ind_mean_cov(y_mat, t_mat, n, z_vect, theta, i, n_iter, prop)

        return Mean_z, Cov_z

    else:
        z_mh = np.zeros((n_iter, len(z_vect)))
        lks_mh_post = np.zeros((n_iter, N))
        rate_z = np.zeros(N)
        sigma_prop_z = np.zeros(N)
        Mean_z = np.zeros((N, K+2))
        Cov_z = np.zeros((N, K+2, K+2))

        iterator = range(N) if not verbose else trange(N)
        for i in iterator:
            idx = np.concatenate((np.array([i]), np.arange(N+(K+1)*i, N+(K+1)*(i+1))))
            z_mh[:,idx], lks_mh_post[:,i], rate_z[i], sigma_prop_z[i], Mean_z[i], Cov_z[i] = mh_tau_xi_ind(y_mat, t_mat, n, z_vect, theta, i, n_iter, prop)

        return z_mh, lks_mh_post, rate_z, sigma_prop_z, Mean_z, Cov_z


def pmh_tau_xi(y_mat, t_mat, n, z_vect, theta, n_iter, pool=None, prop=1e-1, split=False):
    """
    y_mat, t_mat, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    n_iter: number of MCMC steps,
    pool: running pool from the multiprocessing package,
    prop: initial scale factor for the transition covariance,
    split: if True, the MCMC log-likelihoods are given for each individual.
           If False, the log-likelihoods are summed pver individuals
           (split=False saves memory if the option is not needed).

    Adaptive Metropolis-Hastings chain on (tau, xi | y). This function is a parallelized
    version of src_base.mcmc.mh_tau_xi relying on the multiprocessing package.

    Returns:
    z_mh: list of MCMC samples (in z_vect format),
    lks_mh_post: list of log-likelihood p(xi, tau, y, theta) along the chain,
    rate_z: MCMC acceptance rate,
    sigma_prop_z: final scale factor for the adaptive covariance,
    Mean_z: mean of the MCMC samples,
    Cov_z: empirical covariance of the MCMC samples.
    """

    if pool is None:
        # If no pool is given, run the sequential version of the code
        z_mh, lks_mh_post, rate_z, sigma_prop_z, Mean_z, Cov_z = mh_tau_xi(y_mat, t_mat, n, z_vect, theta, n_iter, prop=prop)
        if not split:
            lks_mh_post = lks_mh_post.sum(axis=1)
        return z_mh, lks_mh_post, rate_z, sigma_prop_z, Mean_z, Cov_z

    N, K, d = len(y_mat), len(theta[2])-1, len(theta[0])
    presult = pool.starmap(mh_tau_xi_ind, [(y_mat, t_mat, n, z_vect.copy(), theta, i, n_iter, 1e-1) for i in range(N)])
    z_mh = np.zeros((n_iter, N*(K+2+d)))
    if split:
        lks_mh_post = np.zeros((n_iter, N))
    else:
        lks_mh_post = np.zeros(n_iter)
    rate_z = np.zeros(N)
    sigma_prop_z = np.zeros(N)
    Mean_z = np.zeros((N, K+2))
    Cov_z = np.zeros((N, K+2, K+2))
    for i in range(N):
        idx = np.concatenate((np.array([i]), np.arange(N+(K+1)*i, N+(K+1)*(i+1))))
        z_mh[:,idx] = presult[i][0]
        if split:
            lks_mh_post[:,i] += presult[i][1]
        else:
            lks_mh_post += presult[i][1]
        rate_z[i] = presult[i][2]
        sigma_prop_z[i] = presult[i][3]
        Mean_z[i] = presult[i][4]
        Cov_z[i] = presult[i][5]
    return z_mh, lks_mh_post, rate_z, sigma_prop_z, Mean_z, Cov_z


def posterior_z_mean(y, t, z, theta, n_iter, pool=None, return_std=False):
    """
    y, t, z, theta: see NOTATIONS.md
    n_iter: number of MCMC steps,
    pool: running pool from the multiprocessing package,
    return_std: if True, additionally returns the standard deviation from the mean
                for each coordinate in (tau, xi, psi). If False, only returns
                the mean.

    Computes the posterior mean of the latent variable z = (tau, xi, psi) given y.

    Returns:
    mean: posterior mean of (z | y),
    (std): standard deviation from the mean for each coordinate (i.e., square root
           of the diagonal coefficients of Cov(z | y)).
    """

    z_vect = pack_z(z)
    y_mat = get_y_mat(y)
    t_mat = get_t_mat(t)
    n = np.array(list(map(len, y)))
    N, K, d = len(y_mat), len(theta[2])-1, len(theta[0])
    z_mh, _, _, _, _, _ = pmh_tau_xi(y_mat, t_mat, n, z_vect, theta, n_iter, pool, prop=1e-1, split=False)
    psi_var = np.zeros((n_iter, N, d))
    if pool is not None:
        presults = pool.starmap(posterior_psi, [(y_mat, t_mat, n, z_mh[k], theta) for k in range(n_iter)])
    else:
        presults = []
        for k in range(n_iter):
            presults.append(posterior_psi(y_mat, t_mat, n, z_mh[k], theta))
    for k in range(len(z_mh)):
        z_mh[k, N*(K+2):] = presults[k][0].reshape(-1)
        psi_var[k] = presults[k][1]
    psi_var = psi_var.mean(axis=0)
    mean = z_mh.mean(axis=0)
    std = np.zeros(z_mh.shape[1])
    std[:N*(K+2)] = z_mh[:,:N*(K+2)].std(axis=0)
    # Decomposition of the total variance of (psi | y) into terms conditional to (tau, xi)
    std[N*(K+2):] = np.sqrt(psi_var.reshape(-1) + z_mh[:,N*(K+2):].var(axis=0))

    mean = unpack_z(mean, d=d, K=K)
    std = unpack_z(std, d=d, K=K)
    if return_std:
        return mean, std
    else:
        return mean


@njit
def mh_v0(y_mat, t_mat, n, z_vect, theta, prior, n_iter, prop=(1e-2, 1e-2)):
    """
    y_mat, t_mat, prior, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    n_iter: number of MCMC steps,
    prop: tuple of initial values for the transition variances on v0 and z.

    Performs n_iter steps of adaptive Metropolis-Hastings on the distribution
    (v0 | y, (theta\v0)). This allows quantifying the uncertainty on
    v0 given the available information.

    Returns:
    result: the list of MCMC samples,
    rates: tuple giving the acceptance rates on theta and z,
    final_prop: final value of the adaptive transition variances.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta
    K, N, d = len(theta[1]), len(y_mat), len(y_mat[0,0])
    block_size = 50
    target_rate = 0.3

    # We use a Gaussian transition kernel with variance sigma_prop
    sigma_prop_theta = prop[0]*np.ones(K+1)
    sigma_prop_z     = prop[1]*np.ones(3)
    accepts_theta = np.zeros(K+1)
    accepts_z     = np.zeros(3)
    block_accepts_theta = np.zeros(K+1)
    block_accepts_z     = np.zeros(3)

    current_log_lk = log_lk_full_split(y_mat, t_mat, n, z_vect, theta, prior)
    result = np.zeros((n_iter, K+1, d))

    it = range(n_iter)
    for m in it:
        # MHwG Jump for v0
        for k in range(K+1):
            v0_2 = v0.copy()
            v0_2[k] += np.random.randn(d) * sigma_prop_theta[k]
            theta2 = p0, t0, v0_2, sigma_xi, sigma_tau, sigma_psi, sigma
            # Compute the acceptance log-probability
            new_log_lk = log_lk_full_split(y_mat, t_mat, n, z_vect, theta2, prior)
            log_alpha = new_log_lk.sum() - current_log_lk.sum()
            if np.log(np.random.rand()) < log_alpha:
                v0 = v0_2
                theta = theta2
                current_log_lk = new_log_lk
                accepts_theta[k] += 1
                block_accepts_theta[k] += 1

        # MH Jump for z
        ps0 = get_breakpoint_positions(p0, t0, v0)
        theta_full = (ps0, *theta[1:])
        for i in range(N):
            indices_ind = get_indices(i, N, d, K)
            for k, (a, b) in enumerate(indices_ind):
                idx = np.arange(a, b)
                # Generate next move
                old = z_vect[idx].copy()
                z_vect[idx] = z_vect[idx] + sigma_prop_z[k] * np.random.randn(len(idx))
                # Compute the acceptance log-probability
                new_log_lk = log_lk_ind(y_mat, t_mat, n, z_vect, theta_full, i)
                log_alpha = new_log_lk - current_log_lk[i]

                if np.log(np.random.rand()) < log_alpha:
                    current_log_lk[i] = new_log_lk
                    accepts_z[k] += 1/N
                    block_accepts_z[k] += 1/N
                else:
                    z_vect[idx] = old

        # Adaptive MH rate
        if m>1 and m%block_size == 0:
            d_theta = (block_accepts_theta / block_size) > target_rate
            d_theta = 2*d_theta - 1
            approx_sto = np.log(sigma_prop_theta) + d_theta * 3/(m+1)**0.5
            sigma_prop_theta = np.exp(approx_sto)
            block_accepts_theta = np.zeros(K+1)

            d_z = (block_accepts_z / block_size) > target_rate
            d_z = 2*d_z - 1
            sigma_prop_z = np.exp(np.log(sigma_prop_z) + d_z * 3/(m+1)**0.3)
            block_accepts_z = np.zeros(3)

        result[m] = v0

    rates = (accepts_theta/n_iter, accepts_z/n_iter)
    final_prop = (sigma_prop_theta, sigma_prop_z)
    return result, rates, final_prop
