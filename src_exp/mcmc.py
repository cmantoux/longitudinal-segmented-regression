"""
This file contains functions performing Markov Chain Monte-Carlo
on variables of the exponentialized model. The Metropolis-Hastings within Gibbs
algorithm is used in each case, with custom blocks depending on the target.

More specifically, the functions in this file are:
- mh_z: MCMC targetting the distribution (z | y, theta)
- mh_v0: MCMC targetting the distribution (v0_bar | y, (theta\v0_bar))
"""


import numpy as np
from tqdm.auto import *
from numba import njit

from src_exp.model import *
from src_common.utils import *


@njit
def mh_z(y_mat, t_mat, n, theta, prior, n_iter, z_vect=None,
         prop=1e-2*np.ones(6), only_individuals=False,
         orthogonality_condition=False):
    """
    y_mat, t_mat, theta, z_vect, prior: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    n_iter: number of MCMC steps,
    prop: list of Metropolis transition variances, respectively for:
        - prop[0]: p0
        - prop[1]: t0
        - prop[2]: v0
        - prop[3]: tau
        - prop[4]: xi
        - prop[5]: psi
    only_individuals: if True, the MCMC only updates the individual latent variables
                      (i.e. tau, xi, psi) and leaves the population latent variables
                      (i.e. p0, t0, v0) unchanged. If False, the MCMC updates all
                      the components of z.
    orthogonality_condition: if True, psi_i is constrained to be orthogonal to each speed
                             vector v0[k]. This option is disabled by default, but can
                             be activated to see its impact.

    Performs n_iter steps of Metropolis-Hastings on the distribution (z | y, theta),
    starting at the point z_vect. This function is mainly used in saem.MCMC_SAEM.

    Returns:
    z_vect: the final state of the latent variable
    accepts: array giving, for each Gibbs block, the number of transitions accepted
    current_log_lk: final value of the complete model likelihood.

    NB: The proposition variances in prop are not updated within the mh_z function.
    The adaptation of prop is performed in src_exp.saem.MCMC_SAEM.
    """

    # We use Gaussian transition kernels with variances prop,
    # which gives prop_p0, prop_t0, prop_v0, prop_tau, prop_xi, prop_psi
    K, N, d = len(theta[2]) - 1, len(y_mat), len(y_mat[0,0])
    if z_vect is None:
        z_vect = np.zeros(d+max(K,1)+d*(K+1)+(K+d+2)*N)

    accepts = np.zeros(6) # number of accepts on each variable

    p0, t0, v0, _, _, _ = unpack_z(z_vect, d=d, K=K)
    ps0 = get_breakpoint_positions(p0, t0, v0)
    theta_full = (ps0, *theta[1:]) # alternative representation of theta with ps0 instead of p0

    current_log_lk = log_lk_split(y_mat, t_mat, n, z_vect, theta, prior, orthogonality_condition)
    t_min = t_mat[:,0].min()
    t_max = max([t_mat[i,n[i]-1] for i in range(N)])

    for _ in range(n_iter):
        # Update population variables
        if not only_individuals:
            indices_pop = [
                (0,d),
                (d,d+max(K,1)),
                (d+max(K,1),d+max(K,1)+d*(K+1))
            ]
            for k, (a, b) in enumerate(indices_pop):
                idx = np.arange(a, b)
                old = z_vect[idx]
                z_vect[idx] = z_vect[idx] + prop[k] * np.random.randn(len(idx))
                # Compute the acceptance log-probability
                new_log_lk = log_lk_split(y_mat, t_mat, n, z_vect, theta, prior, orthogonality_condition)
                log_alpha = new_log_lk.sum() - current_log_lk.sum()

                if k==1: # When we try to sample on t0
                    new_t0 = z_vect[idx]
                    t0_increasing = ((new_t0[1:]-new_t0[:-1])>=0).all()
                    t0_bounds = (new_t0[0] > t_min) and (new_t0[-1] < t_max)
                    t0_valid = t0_increasing and t0_bounds
                else:
                    t0_valid = True

                if np.log(np.random.rand()) < log_alpha and t0_valid:
                    p0, t0, v0, _, _, _ = unpack_z(z_vect, d=d, K=K)
                    ps0 = get_breakpoint_positions(p0, t0, v0)
                    theta_full = (ps0, *theta[1:]) # Update theta_full for the loop on i
                    current_log_lk = new_log_lk
                    accepts[k] += 1
                else:
                    z_vect[idx] = old

        if orthogonality_condition:
            A = orthogonal_complement(v0)
            Projection = A.T@A # Orthogonal projection onto the complement of Span(v0)

        # Update every individual
        for i in range(N):
            indices_ind = get_indices(i, N, d, K) # Obtain the indices of individual i variables in z_vect
            for k, (a, b) in enumerate(indices_ind):
                idx = np.arange(a, b)
                # Generate next move
                old = z_vect[idx]
                z_vect[idx] = z_vect[idx] + prop[k+3] * np.random.randn(len(idx))

                if orthogonality_condition and k==2: # Project psi on the orthogonal complement of Span(v0)
                    z_vect[idx] = Projection @ z_vect[idx]

                # Compute the acceptance log-probability
                new_log_lk = log_lk_ind(y_mat, t_mat, n, z_vect, theta_full, i, orthogonality_condition)
                log_alpha = new_log_lk - current_log_lk[i]

                if np.log(np.random.rand()) < log_alpha:
                    current_log_lk[i] = new_log_lk
                    accepts[k+3] += 1/N
                else:
                    z_vect[idx] = old

    return z_vect, accepts, current_log_lk


@njit
def mh_v0(y_mat, t_mat, n, z_vect, theta, prior, n_iter, prop=(1e-2, 1e-2)):
    """
    y_mat, t_mat, prior, theta, z_vect: see NOTATIONS.md
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    n_iter: number of MCMC steps,
    prop: tuple of initial values for the transition variances on v0 and z.

    Performs n_iter steps of adaptive Metropolis-Hastings on the distribution
    (v0_bar | y, (theta\v0_bar)). This allows quantifying the uncertainty on
    v0 given the available information.

    Returns:
    result: the list of MCMC samples,
    rates: tuple giving the acceptance rates on theta and z,
    final_prop: final value of the adaptive transition variances.
    """

    p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma = theta
    K, N, d = len(theta[1]), len(y_mat), len(y_mat[0,0])

    block_size = 100
    target_rate = 0.3

    # We use a Gaussian transition kernel with variance sigma_prop
    sigma_prop_theta = prop[0]*np.ones(K+1)
    sigma_prop_z     = prop[1]
    accepts_theta = np.zeros(K+1)
    accepts_z     = 0
    block_accepts_theta = np.zeros(K+1)
    block_accepts_z     = 0

    current_log_lk = log_lk_full(y_mat, t_mat, n, z_vect, theta, prior)
    result = np.zeros((n_iter, K+1, d))


    it = range(n_iter)
    for i in it:
        # MHwG Jump for v0
        for k in range(K+1):
            v0_bar_2 = v0_bar.copy()
            v0_bar_2[k] += np.random.randn(d) * sigma_prop_theta[k] / d
            theta2 = p0_bar, t0_bar, v0_bar_2, sigma_xi, sigma_tau, sigma_psi, sigma
            # Compute the acceptance log-probability
            new_log_lk = log_lk_full(y_mat, t_mat, n, z_vect, theta2, prior)
            log_alpha = new_log_lk - current_log_lk
            if np.log(np.random.rand()) < log_alpha:
                v0_bar = v0_bar_2
                theta = theta2
                current_log_lk = new_log_lk
                accepts_theta[k] += 1
                block_accepts_theta[k] += 1

        # MH Jump for z
        z_vect2 = z_vect + np.random.randn(len(z_vect)) * sigma_prop_z / len(z_vect)
        # Compute the acceptance log-probability
        new_log_lk = log_lk_full(y_mat, t_mat, n, z_vect2, theta, prior)
        log_alpha = new_log_lk - current_log_lk
        if np.log(np.random.rand()) < log_alpha:
            z_vect = z_vect2
            current_log_lk = new_log_lk
            accepts_z += 1
            block_accepts_z += 1

        # Adaptive MH rate
        if i%block_size == 0:
            for k in range(K+1):
                d_theta = (block_accepts_theta[k] / block_size) > target_rate
                d_theta = 2*d_theta - 1
                approx_sto = np.log(sigma_prop_theta[k]) + d_theta / (i+1)**0.3
                sigma_prop_theta[k] = np.exp(approx_sto)
                block_accepts_theta[k] = 0

            d_z = (block_accepts_z / block_size) > target_rate
            d_z = 2*d_z - 1
            sigma_prop_z = np.exp(np.log(sigma_prop_z) + d_z / (i+1)**0.3)
            block_accepts_z = 0

        result[i] = v0_bar

    rates = (accepts_theta/n_iter, accepts_z/n_iter)
    final_prop = (sigma_prop_theta, sigma_prop_z)
    return result, rates, final_prop
