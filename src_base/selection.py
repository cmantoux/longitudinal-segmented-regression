"""
This file contains functions to compute the model's marginal likelihood and
information criteria for model selection.

The main functions meant to be used outside of this file are:
- compute_marginal_likelihood
- relative_marginal_likelihood_error
and the information criteria:
- AIC
- BIC
- BIC_hybrid
"""

import numpy as np
import multiprocessing
from tqdm.auto import trange
import pickle

from src_base.model import *
from src_base.mcmc import *
from src_common.utils import *


@njit
def _get_lks_is_post(y_mat, t_mat, n, z_is, theta):
    """
    Auxiliary function used within _get_lks. The function computes the model likelihood of z_is.
    """
    N, n_is = len(y_mat), len(z_is)
    lks_is_post = np.zeros((n_is, N))
    for k in range(n_is):
        lks_is_post[k] = log_lk_tau_xi_split(y_mat, t_mat, n, z_is[k], theta)
    return lks_is_post


def _get_lks(y, t, z, theta, n_mh, n_is, burn, pool=None, split=True, verbose=True):
    """
    y, t, z, theta: see NOTATIONS.md,
    n_mh: number of Metropolis-Hastings MCMC samples,
    n_is: number of samples for Importance Sampling,
    burn: number of samples discarded at the start of the MCMC,
    pool: optional jobs pool from the multiprocessing package,
    split: if False, the likelihoods are summed over individuals. If True, the
           likelihood is returned separately for each individual,
    verbose: if True, prints progress bars for each MCMC or loop.

    This function performs sampling and likelihood computations from the posterior
    distribution (z | y) and its importance sampling (IS) Gaussian approximation.
    The posterior sample is denoted z_mh and the importance sample z_is.
    Their likelihoods are denoted as follows:
    lks_mh_post & lks_mh_gauss : likelihoods of z_mh for the posterior distribution
                                 and the importance distribution
    lks_is_post & lks_is_gauss : likelihoods of z_is for the posterior distribution
                                 and the importance distribution
    """

    parallel = (pool is not None)
    if verbose:
        print(f"Sampling z_mh (parallel={parallel}).")
    else:
        iterator = trange(6)

    n = np.array(list(map(len, y)))
    y_mat = get_y_mat(y)
    t_mat = get_t_mat(t)
    z_vect = pack_z(z)

    # First, run two MCMCs (one to compute the posterior mean of z, the other to get new independent samples)
    if parallel:
        z_mh, lks_mh_post, rate, prop, _, _ = pmh_tau_xi(
            y_mat, t_mat, n, z_vect, theta, n_iter=n_mh, pool=pool, prop=0.0001, split=True)
        if not verbose:
            iterator.update(1)
        _, _, _, _, mu_tau_xi, cov_tau_xi = pmh_tau_xi(
            y_mat, t_mat, n, z_vect, theta, n_iter=n_mh, pool=pool, prop=0.0001, split=True)
    else:
        z_mh, lks_mh_post, rate, prop, _, _ = mh_tau_xi(
            y_mat, t_mat, n, z_vect, theta, n_iter=n_mh, prop=0.0001, verbose=verbose)
        if not verbose:
            iterator.update(1)
        mu_tau_xi, cov_tau_xi = mh_tau_xi(
            y_mat, t_mat, n, z_vect, theta, n_iter=n_mh, only_mean_cov=True, prop=0.0001, verbose=verbose)

    N, _, d = y_mat.shape
    K = len(theta[2]) - 1
    indices = [np.concatenate([
                np.arange(i, i+1), # tau_i
                np.arange(N+(K+1)*i, N+(K+1)*(i+1)), # xi_i
            ]) for i in range(N)]

    if verbose:
        print("Computing IS parameters.")
    else:
        iterator.update(1)

    # Posterior mean of z, with the (N*(d+K+2)) shape of z (unlike mu_tau_xi, which is a (N, K+2) matrix)
    mu_vector = np.zeros(z_mh.shape[1])

    for i in range(N):
        # Fill mu_vector with the values of mu_tau_xi (leaving the psi coordinates empty because of marginalization)
        mu_vector[indices[i]] = mu_tau_xi[i]
    sqrt_cov = np.zeros((N, K+2, K+2))
    for i in range(N):
        sqrt_cov[i] = sqrtm(cov_tau_xi[i])

    logdet = np.linalg.slogdet(cov_tau_xi)[1]

    if verbose: print("Computing lks_mh_gauss.")
    Z = z_mh[burn:]-mu_vector
    lks_mh_gauss = np.zeros((len(Z), N))
    for i in range(N):
        ld = np.linalg.slogdet(cov_tau_xi[i])[1]
        L = np.linalg.inv(cov_tau_xi[i])
        lks_tmp = np.einsum("ij, ik, jk -> i", Z[:,indices[i]], Z[:,indices[i]], L)
        lks_mh_gauss[:,i] = -lks_tmp/2 - ld/2 - (K+2)*np.log(2*np.pi)/2

    if verbose:
        print("Sampling z_is.")
    else:
        iterator.update(1)

    eps = np.random.randn(n_is, N*(K+2))
    z_is = mu_vector[None,:].repeat(n_is, axis=0)
    for i in range(N):
        z_is[:,indices[i]] = mu_tau_xi[i] + eps[:,indices[i]] @ sqrt_cov[i]

    if verbose:
        print(f"Computing lks_is_post (parallel={parallel}).")
    else:
        iterator.update(1)

    if parallel:
        lks_is_post = np.array(pool.starmap(log_lk_tau_xi_split,
                                        [(y_mat, t_mat, n, z_is[k], theta) for k in range(n_is)]))
    else:
        lks_is_post = _get_lks_is_post(y_mat, t_mat, n, z_is, theta)

    if verbose:
        print("Computing lks_is_gauss.")
    else:
        iterator.update(1)

    lks_is_gauss = np.zeros((n_is, N))
    for i in range(N):
        ld = np.linalg.slogdet(cov_tau_xi[i])[1]
        lks_is_gauss[:,i] = -(eps[:,indices[i]]**2).sum(axis=1)/2 - ld/2 - (K+2)*np.log(2*np.pi)/2

    if not verbose:
        iterator.update(1)
        iterator.close()

    if split:
        lks = {
            "mh_post":  lks_mh_post[burn:],
            "mh_gauss": lks_mh_gauss,
            "is_post":  lks_is_post,
            "is_gauss": lks_is_gauss,
        }
    else:
        lks = {
            "mh_post":  lks_mh_post[burn:].sum(axis=1),
            "mh_gauss": lks_mh_gauss.sum(axis=1),
            "is_post":  lks_is_post.sum(axis=1),
            "is_gauss": lks_is_gauss.sum(axis=1),
        }

    return z_mh, z_is, mu_tau_xi, cov_tau_xi, lks


def _bridge_sampling(lks, tol=1e-10, improved_cv=False):
    """
    lks: dictionary of likelihood arrays mh_post, mh_gauss, is_post, is_gauss
         for a single individual (see _get_lks documentation for more details)
    tol: tolerance threshold to stop the bridge sampling fixed-point iteration.
         A smaller threshold increases the precision but requires more time to
         reach convergence.
    improved_cv: determines which fixed-point iteration is used. If improved_cv
                 is False, the default iteration x_{t+1} = f(x_t) is used.
                 if improved_cv is True, the averaged iteration
                    x_{t+1} = 0.5*f(x_t) + 0.5*x_t
                 is used instead. The averaged iteration ensures the convergence,
                 which otherwise can be hindered in high dimension. However, the
                 asymptotic convergence speed is slower when using improved_cv.

    Given the likelihoods of the posterior MCMC samples and the importance samples,
    this function executes the bridge sampling fixed-point iteration to estimate the
    marginal likelihood of an individual.
    """
    l1 = lks["mh_post"] - lks["mh_gauss"]
    l2 = lks["is_post"] - lks["is_gauss"]
    N1 = len(lks["mh_post"])
    N2 = len(lks["is_post"])
    s1 = N1/(N1+N2)
    s2 = N2/(N1+N2)

    lp = 1
    lp2 = 0
    k = 0
    while abs((lp-lp2)/lp) > tol or k==0:
        lp2 = lp
        num = logsumexp(l2 - np.logaddexp(l2+np.log(s1), lp+np.log(s2))) - np.log(N2)
        den = logsumexp(-np.logaddexp(l1+np.log(s1), lp+np.log(s2))) - np.log(N1)
        if improved_cv:
            lp = ((num - den) + lp2)/2
        else:
            lp = num - den
        k += 1
    return lp


def _relative_marginal_likelihood_error(lks, BS):
    """
    lks: dictionary produced by running _get_lks
    BS: list of individual marginal likelihoods estimated by bridge_sampling

    Compute the relative square error on the marginal likelihood (see Gronau et al. 2017, Eq. 17 p. 89)
    """
    N1 = len(lks["mh_post"])
    N2 = len(lks["is_post"])
    s1 = N1/(N1+N2)
    s2 = N2/(N1+N2)
    f1 = lks["is_post"] - np.logaddexp(lks["is_post"]+np.log(s1), lks["is_gauss"]+BS+np.log(s2))
    f2 = lks["mh_gauss"]+BS - np.logaddexp(lks["mh_post"]+np.log(s1), lks["mh_gauss"]+BS+np.log(s2))
    a1 = log_var(f1) - 2*log_mean(f1) - np.log(N2)
    normalized_spectral_density = AR1_spectral_density(np.exp(f2))
    a2 = log_var(f2) - 2*log_mean(f2) - np.log(N1) + np.log(normalized_spectral_density)
    return np.exp(a1) + np.exp(a2)


def compute_marginal_likelihood(y, t, z, theta, n_iter, pool=None, reduce=True):
    """
    y, t, z, theta: see NOTATIONS.md,
    n_iter: number of iterations in bridge sampling MCMC,
    pool: optional jobs pool from the multiprocessing package,
    reduce: if True, only returns the sum of marginal likelihoods and
            the aggregated error term. If False, returns the likelihood of each
            individual separately, along with its own error estimate.

    Computes an estimation of the marginal log-likelihood log p(y_i | theta)
    for each individual using bridge sampling.

    Returns:
    BS: the list of marginal log-likelihoods for each individuals
    RE: an estimate of the relative error on BS, also equal to the asymptotic
        variance around BS.
    """

    N = len(y)

    print("Generating full likelihood samples with MCMC.")
    z_mh, z_is, mu_z, cov_z, lks = _get_lks(
        y, t, z, theta,
        n_mh=n_iter, n_is=n_iter, burn=n_iter//10,
        pool=pool, split=True, verbose=False)
    del z_mh, z_is, mu_z, cov_z

    print("Computing the bridge sampling estimation of the marginal likelihood.")
    BS = np.zeros(N) # Bridge Sampling (BS) result, in log form
    RE = np.zeros(N) # Estimate of the BS relative square error (in log form)
    # RE is also the asymptotic variance of the log-likelihood
    # It is thus used for confidence intervals
    for i in trange(N):
        # Create a dictionary of likelihoods containing only individual i
        lks_i = {
            "mh_post":  lks["mh_post"][:,i],
            "mh_gauss": lks["mh_gauss"][:,i],
            "is_post":  lks["is_post"][:,i],
            "is_gauss": lks["is_gauss"][:,i],
        }
        BS[i] = _bridge_sampling(lks_i, tol=1e-5, improved_cv=True)
        RE[i] = _relative_marginal_likelihood_error(lks_i, BS[i])

    if reduce:
        return BS.sum(), np.sqrt(RE.sum())
    else:
        return BS, np.sqrt(RE)


def model_dimension(K, d, heteroscedastic=True):
    """
    K: number of breaks,
    d: data dimension,
    N: number of individuals,
    heteroscedastic: if True (default case), sigma_psi, sigma_xi and sigma are
                     allowed to take different values in each coordinate.

    Returns the number of parameters in the model.
    """

    if heteroscedastic:
        return d + max(K, 1) + d*(K+1) + 2*d+K+3
    else:
        return d + max(K, 1) + d*(K+1) + 4


def AIC(marginal_likelihood, K, d, heteroscedastic=True):
    """
    marginal_likelihood: log p(y | theta),
    K: number of breaks,
    d: data dimension,
    N: number of individuals,
    heteroscedastic: if True (default case), sigma_psi, sigma_xi and sigma are
                     allowed to take different values in each coordinate.

    Returns the AIC information criterion.
    """

    penalty = 2 * model_dimension(K, d, heteroscedastic)
    return penalty - 2 * marginal_likelihood


def BIC(marginal_likelihood, K, d, N, heteroscedastic=True):
    """
    marginal_likelihood,
    K: number of breaks,
    d: data dimension,
    N: number of individuals,
    heteroscedastic: if True (default case), sigma_psi, sigma_xi and sigma are
                     allowed to take different values in each coordinate.

    Returns the BIC information criterion.
    """

    penalty = np.log(N) * model_dimension(K, d, heteroscedastic)
    return penalty - 2 * marginal_likelihood


def BIC_hybrid(marginal_likelihood, K, d, N, n, heteroscedastic=True):
    """
    K: number of breaks
    d: data dimension
    N: number of individuals
    n: list of observation counts (e.g., n[3] is the number of observations
       for individual 3)
    heteroscedastic: if True (default case), sigma_psi, sigma_xi and sigma are
                     allowed to take different values in each coordinate.

    Modified version of the BIC proposed by Delattre et al. (2014), to better
    handle mixed effects model.
    """

    if heteroscedastic:
        penalty = np.log(sum(n))*(d + max(K, 1) + d*(K+1) + 1) + np.log(N)*(2*d+K+2)
    else:
        penalty = np.log(sum(n))*(d + max(K, 1) + d*(K+1) + 1) + np.log(N)*3
    return penalty - 2 * marginal_likelihood
