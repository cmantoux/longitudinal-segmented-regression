"""
This file mainly contains functions to
- manipulate model variables
- compute various likelihoods
within the framework of the base model.

It is divided into three sections:
1) functions to manipulate the model variables and compute trajectories
2) various computations of model likelihoods
3) tools to estimate of the proportion of variance explained by the model
"""


import numpy as np
from numba import njit

import src_common.utils


########################################################
### VARIABLE MANIPULATION AND TRAJECTORY COMPUTATION ###
########################################################


def get_prior(K, d, regularize_xi=6.):
    """
    K: number of breakpoints in the population trajectory
    d: data dimension
    regularize_xi: inverse scale parameter for the prior of sigma_xi. A bigger
                   value for regularize_xi means a stronger regularization on sigma_xi.

    Returns a standard prior for the base model.
    """

    sigma_p0 = 500.
    sigma_t0 = 500.
    sigma_v0 = 500.
    p0_bar = np.zeros(d)
    t0_bar = np.arange(max(K, 1)).astype(np.float)
    v0_bar = np.zeros((K+1, d))
    m, v = ([regularize_xi, 6., 6., 6.], [1., 1., 1., 1.]) # inverse gamma prior for the variances of sigma_xi, sigma_tau, sigma_psi, sigma (in this order)
    return (sigma_p0,sigma_t0,sigma_v0,p0_bar,t0_bar,v0_bar,m,v)


@njit
def pack_theta(theta):
    """
    theta: see NOTATIONS.md

    Pack the model parameters theta into a vector.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta
    return np.concatenate((p0, t0, v0.reshape(-1), sigma_xi, sigma_tau, sigma_psi, sigma))


@njit
def unpack_theta(theta_vect, d, K):
    """
    theta_vect: vector representation of the model parameters theta.

    Unpack the vector representation of theta into a tuple.
    """

    p0, theta_ = theta_vect[:d], theta_vect[d:]
    t0, theta_ = theta_[:max(K,1)], theta_[max(K,1):]
    v0, theta_ = theta_[:(K+1)*d].reshape((K+1, d)), theta_[(K+1)*d:]
    sigma_xi  = theta_[:K+1]
    sigma_tau = np.array([theta_[K+1]])
    sigma_psi = theta_[K+2:K+d+2]
    sigma     = theta_[K+d+2:]
    return p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma


@njit
def pack_z(z):
    """
    z: see NOTATIONS.md

    Pack the model latent variable z into a vector.
    """

    tau, xi, psi = z
    z_vect = np.concatenate((tau, xi.reshape(-1), psi.reshape(-1))) # N*(K+1) + N + N*d = (K+2+d)*N
    return z_vect


@njit
def unpack_z(z_vect, d, K):
    """
    z_vect: vector representation of the model parameters z,
    d: data dimension,
    K: number of breaks in the population trajectory.

    Unpack the vector representation of z into a tuple.
    """

    N = int(len(z_vect)/(K+d+2))
    tau = z_vect[:N]
    xi = z_vect[N:(K+2)*N].reshape(N, K+1)
    psi = z_vect[(K+2)*N:(K+2+d)*N].reshape(N, d)
    return (tau, xi, psi)


@njit
def get_indices(i, N, d, K):
    """
    i: index of an individual,
    N: number of individuals (i < N),
    d: data dimension,
    K: number of breaks in the population trajectory.

    Returns three arrays of coordinates giving the indices of tau_i, xi_i and
    psi_i in the vector representation z_vect.
    More precisely, if we set I = get_indices(i, N, d, K), we can then retrieve:
    tau_i = z_vect[I[0][0] : I[0][1]]
    xi_i  = z_vect[I[1][0] : I[1][1]]
    psi_i = z_vect[I[2][0] : I[2][1]]
    This representation ensures the compliance with numba standards.
    """

    return np.array((
        (i, i+1), # tau_i
        (N+(K+1)*i, N+(K+1)*(i+1)), # xi_i
        (N*(K+2)+d*i, N*(K+2)+d*(i+1)) # psi_i
    ))


@njit
def unpack_ind(z_vect, i, d, K):
    """
    z_vect: see NOTATIONS.md,
    i: index of an individual,
    d: data dimension,
    K: number of breaks in the population trajectory.

    Unpack the latent variables (tau_i, xi_i, psi_i) of a single individual
    into a tuple.
    """

    N = int(len(z_vect)/(K+d+2))
    indices = get_indices(i, N, d, K)
    tau = z_vect[np.arange(indices[0,0], indices[0,1])][0]
    xi  = z_vect[np.arange(indices[1,0], indices[1,1])]
    psi = z_vect[np.arange(indices[2,0], indices[2,1])]
    return (tau, xi, psi)


@njit
def get_taus(tau_i, xi_i, t0):
    """
    tau_i: time shift of an individual,
    xi_i: list of the acceleration factors of an individual,
    t0: time position of the trajectory breaks.

    Computes difference between:
    - the break time positions of a given individual trajectory,
    - and the break time positions in the population trajectory.
    """

    K = len(xi_i) - 1
    taus_i = np.zeros((K+1)) # tau_i0 = tau_i1, one shift for each segment (K+1 segments)
    taus_i[0] = tau_i
    if K > 0:
        taus_i[1] = tau_i
        for k in range(2, K+1):
            taus_i[k] = taus_i[k-1] + (t0[k-1]-t0[k-2])*(np.exp(-xi_i[k-1])-1)
            # Rupture time of individual i for break k: t0[k]+taus_i[i,k+1]
    return taus_i


@njit
def get_breakpoint_positions(p0, t0, v0):
    """
    p0: spatial position of the first break,
    t0: time position of the trajectory breaks,
    v0: list of speed vectors for each break.

    Returns the list of spatial positions for all breakpoints.

    NB: if K=0, the result still contains at least p0.
    """

    d, K = len(p0), len(v0) - 1
    p = np.zeros((max(K, 1), d))
    p[0] = p0
    for i in range(1, K):
        p[i] = p[i-1] + (t0[i]-t0[i-1])*v0[i]
    return p


@njit
def get_z_full(z, theta):
    """
    z, theta: see NOTATIONS.md.

    Computes z_full, an augmented version of z containing the spatial position of
    all trajectory breakpoints and the time position of all individual breakpoints.

    NB: although the shape of z differ in the base model and the exponential model,
    z_full concides in both models.
    """

    p0, t0, v0, _, _, _, _ = theta
    tau, xi, psi = z
    N, d, K = len(tau), len(p0), len(v0) - 1
    ps = get_breakpoint_positions(p0, t0, v0)
    taus = np.zeros((N, K+1)) #tau_i0 = tau_i1, one shift for each segment (K+1 segments)
    for i in range(N):
        taus[i] = get_taus(tau[i], xi[i], t0)
    return (ps, t0, v0, taus, xi, psi)


@njit
def time_reparam(taus_i, xi_i, time, t0):
    """
    taus_i: list of trajectory breakpoints shifts for a given individual,
    xi_i: list of acceleration factors for a given individual,
    time: float number representing a point in the timeline of the individual,
    t0: list of time position of the population trajectory breakpoints.

    Reparameterizes the input time into the timeline of the population trajectory.

    NB: the formulas come from Debavelaere 2021 (PhD thesis), p. 60.
    """

    K = len(xi_i) - 1
    if K==0:
        return t0[0] + np.exp(xi_i[0]) * (time - t0[0] - taus_i[0])
    else:
        k = src_common.utils.digitize(time, t0+taus_i[1:])
        if k==0:
            return t0[0] + np.exp(xi_i[0]) * (time - t0[0] - taus_i[0])
        else:
            return t0[k-1] + np.exp(xi_i[k]) * (time - t0[k-1] - taus_i[k])


@njit
def D(ps0, t0, v0, time):
    """
    ps0 : list of spatial positions of the trajectory breakpoints,
    t0: time position of the trajectory breaks,
    v0: list of speed vectors for each break,
    time: float number representing a point in the timeline of the
          population trajectory.

    Computes the value of the population trajectory at the given imput time.
    """

    K, d = len(v0) - 1, ps0.shape[1]
    if K==0:
        return ps0[0]+(time-t0[0])*v0[0]
    else:
        k = src_common.utils.digitize(time, t0) # speed index
        if k==0:
            return ps0[0]+(time-t0[0])*v0[0]
        else:
            return ps0[k-1]+(time-t0[k-1])*v0[k]


@njit
def D_i(z_full, i, time):
    """
    z_full: see NOTATIONS.md,
    i: index of an individual,
    time: float number representing a point in the timeline of the individual.

    Computes the value of the individual trajectory at the given imput time.
    """

    ps0, t0, v0, taus, xi, psi = z_full
    return D(ps0, t0, v0, time_reparam(taus[i], xi[i], time, t0)) + psi[i]


@njit
def D_i_alt(ps0, t0, v0, z_i_full, time):
    """
    ps0 : list of spatial positions of the trajectory breakpoints,
    t0: time position of the trajectory breaks,
    v0: list of speed vectors for each break,
    time: float number representing a point in the timeline of the
          population trajectory.
    z_i_full: tuple containing:
        - taus_i: the time shift of the individual breakpoints
        - xi_i : the acceleration factors of the individual
        - psi_i : the spatial shift of the individual

    Computes the value of the individual trajectory at the given imput time.
    This function provides an alternative implementation of D_i with
    different arguments.
    """

    taus_i, xi_i, psi_i = z_i_full
    return D(ps0, t0, v0, time_reparam(taus_i, xi_i, time, t0)) + psi_i


############# OCCUPATION RATES COMPUTATION ##############


def _occupation_rate(t, taus, xi, t0, t1, t2, rnd=3):
    """
    t, taus, xi, t0: see NOTATIONS.md
    t1, t2: distinct time points
    rnd: rounding on the result

    Find the overall proportion of times t[i,:] which,
    once reparameterized, fall between t1 and t2.
    """
    N = len(taus)
    t1, t2 = sorted([t1, t2])
    tab = np.concatenate([[time_reparam(taus[i], xi[i], t[i][j], t0) for j in range(len(t[i]))] for i in range(N)])
    return np.round(((tab >= t1) * (tab <= t2)).mean(), rnd)


def occupation_rates(t, z, theta, rnd=3):
    """
    t, z, theta: see NOTATIONS.md
    rnd: rounding on the result

    Get the sequence of all K+1 occupation rates for each piece of the
    population trajectory.
    """

    tau, xi, _ = z
    t0 = theta[1]

    N = len(t)
    taus = np.array([get_taus(tau[i], xi[i], t0) for i in range(N)])

    result = []
    result.append(_occupation_rate(t, taus, xi, t0, -1e6, t0[0], rnd=rnd))
    for k in range(len(t0)-1):
        result.append(_occupation_rate(t, taus, xi, t0, t0[k], t0[k+1], rnd=rnd))
    result.append(_occupation_rate(t, taus, xi, t0, t0[-1], 1e6, rnd=rnd))
    return result


def occupation_rates_ind(t, z, theta, rnd=3):
    """
    t, z, theta: see NOTATIONS.md
    rnd: rounding on the result

    Get the sequence of all K+1 occupation factors for each individual separately.
    The result thus has shape (N, K+1).

    NB: as a consequence, averaging the result of this function gives the same
    output as occupation_rates.
    """
    tau, xi, _ = z
    t0 = theta[1]

    N = len(t)
    K = xi.shape[1]-1
    result = np.zeros((N,K+1))
    for i in range(N):
        z_tmp = [tau[i]], [xi[i]], None
        result[i] = occupation_rates([t[i]], z_tmp, theta, rnd)
    return result


########################################################
######### COMPUTATION OF THE MODEL LIKELIHOOD ##########
########################################################


@njit
def log_lk(y_mat, t_mat, n, z_vect, theta, orthogonality_condition=False):
    """
    y_mat, z_vect, theta, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    orthogonality_condition: if True, each psi[i] is constrained to be orthogonal
                             to each speed vector v0[k].

    Computes log p(y, z | theta).
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1
    psi_dim = d-K-1 if orthogonality_condition else d

    z = unpack_z(z_vect, K=K, d=d)
    z_full = get_z_full(z, theta)
    tau, xi, psi = z

    c = 0.5*np.log(2*np.pi)
    res = 0
    for i in range(N):
        D_tmp = np.zeros((n[i], d))
        for j in range(n[i]):
            D_tmp[j] = D_i(z_full, i, t_mat[i,j])
            for k in range(d):
                if not np.isnan(y_mat[i,j,k]):
                    res -= 0.5*(y_mat[i,j,k]-D_tmp[j,k])**2/sigma[k]**2 + (np.log(sigma[k]) + c)
    res -= 0.5*np.sum(xi**2/sigma_xi**2)      + N*(np.log(sigma_xi).sum() + (K+1)*c)
    res -= 0.5*np.sum(tau**2/sigma_tau[0]**2) + N*(np.log(sigma_tau[0]) + c)
    res -= 0.5*np.sum(psi**2/sigma_psi**2)    + N*(psi_dim/d)*(np.log(sigma_psi).sum() + d*c)
    return res


@njit
def log_lk_tau_xi_split(y_mat, t_mat, n, z_vect, theta):
    """
    y_mat, z_vect, theta, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    Computes the array of log p(tau_i, xi_i, y_i | theta) for all i.

    NB: the posterior density has been integrated along psi.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1

    z = unpack_z(z_vect, K=K, d=d)
    tau, xi, psi = z
    z = tau, xi, 0*psi
    z_full = get_z_full(z, theta)

    c = 0.5*np.log(2*np.pi)
    res = np.zeros(N)
    for i in range(N):
        D_tmp = np.zeros((n[i], d))
        for j in range(n[i]):
            D_tmp[j] = D_i(z_full, i, t_mat[i,j])
        for k in range(d):
            Ak = 0
            Bk = 0
            n_ik = 0 # number of samples with non-missing coordinate k
            for j in range(n[i]):
                if not np.isnan(y_mat[i,j,k]):
                    diff = y_mat[i,j,k] - D_tmp[j,k]
                    Ak += diff
                    Bk += diff**2
                    n_ik += 1
            sigma_ik = np.sqrt(1/(n_ik/sigma[k]**2 + 1/sigma_psi[k]**2))
            res[i] += 0.5*(Ak**2)*sigma_ik**2/sigma[k]**4  - 0.5*Bk/sigma[k]**2
            res[i] += np.log(sigma_ik) - np.log(sigma_psi[k])
            res[i] -= n_ik*(np.log(sigma[k]) + c)
        res[i] -= 0.5*np.sum(xi[i]**2/sigma_xi**2)   + (np.log(sigma_xi).sum() + (K+1)*c)
        res[i] -= 0.5*(tau[i]**2)/sigma_tau[0]**2 + (np.log(sigma_tau[0]) + c)
    return res


@njit
def log_lk_tau_xi(y_mat, t_mat, n, z_vect, theta):
    """
    y_mat, z_vect, theta, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    Computes log p(tau, xi, y | theta).

    NB: the posterior density has been integrated along psi.
    """
    return log_lk_tau_xi_split(y_mat, t_mat, n, z_vect, theta).sum()


@njit
def posterior_psi_ind(y_mat, t_mat, n, z_vect, theta, i):
    """
    y_mat, z_vect, theta, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    i: index of an individual.

    Returns the conditional mean and variance of psi_i given y, theta and
    (tau_i, xi_i), for individual i.
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1

    z = unpack_z(z_vect, K=K, d=d)
    tau, xi, psi = z
    z = tau, xi, 0*psi
    z_full = get_z_full(z, theta)

    cond_mean = np.zeros(d)
    cond_std = np.zeros(d)

    c = 0.5*np.log(2*np.pi)
    res = 0
    D_tmp = np.zeros((n[i], d))
    for j in range(n[i]):
        D_tmp[j] = D_i(z_full, i, t_mat[i,j])
    for k in range(d):
        Ak = 0
        Bk = 0
        n_ik = 0 # number of samples with non-missing coordinate k
        for j in range(n[i]):
            if not np.isnan(y_mat[i,j,k]):
                diff = y_mat[i,j,k] - D_tmp[j,k]
                Ak += diff
                Bk += diff**2
                n_ik += 1
        sigma_ik = np.sqrt(1/(n_ik/sigma[k]**2 + 1/sigma_psi[k]**2))
        cond_mean[k] = Ak * sigma_ik**2 / sigma[k]**2
        cond_std[k] = sigma_ik
    return cond_mean, cond_std


@njit
def posterior_psi(y_mat, t_mat, n, z_vect, theta):
    """
    y_mat, z_vect, theta, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    Returns the conditional mean and variance of psi given y and (tau, xi).
    """

    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1

    z = unpack_z(z_vect, K=K, d=d)
    tau, xi, psi = z
    z = tau, xi, 0*psi
    z_full = get_z_full(z, theta)

    cond_mean = np.zeros((N, d))
    cond_std = np.zeros((N, d))

    c = 0.5*np.log(2*np.pi)
    res = 0
    for i in range(N):
        D_tmp = np.zeros((n[i], d))
        for j in range(n[i]):
            D_tmp[j] = D_i(z_full, i, t_mat[i,j])
        for k in range(d):
            Ak = 0
            Bk = 0
            n_ik = 0 # number of samples with non-missing coordinate k
            for j in range(n[i]):
                if not np.isnan(y_mat[i,j,k]):
                    diff = y_mat[i,j,k] - D_tmp[j,k]
                    Ak += diff
                    Bk += diff**2
                    n_ik += 1
            sigma_ik = np.sqrt(1/(n_ik/sigma[k]**2 + 1/sigma_psi[k]**2))
            cond_mean[i,k] = Ak * sigma_ik**2 / sigma[k]**2
            cond_std[i,k] = sigma_ik
    return cond_mean, cond_std


@njit
def log_lk_ind(y_mat, t_mat, n, z_vect, theta_full, i, orthogonality_condition=False):
    """
    y_mat, z_vect, theta_full, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    i: index of an individual.

    Computes log p(y_i, z_i | theta) for an individual i.
    """

    ps0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta_full

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1
    psi_dim = d-K-1 if orthogonality_condition else d

    zi = unpack_ind(z_vect, i, K=K, d=d)
    tau, xi, psi = zi
    z_i_full = get_taus(tau, xi, t0), xi, psi

    c = 0.5*np.log(2*np.pi)
    res = 0
    D_tmp = np.zeros((n[i], d))
    for j in range(n[i]):
        D_tmp[j] = D_i_alt(ps0, t0, v0, z_i_full, t_mat[i,j])
        for k in range(d):
            if not np.isnan(y_mat[i,j,k]):
                res -= 0.5*(y_mat[i,j,k]-D_tmp[j,k])**2/sigma[k]**2 + (np.log(sigma[k]) + c)
    res -= 0.5*np.sum(xi**2/sigma_xi**2)   + (np.log(sigma_xi).sum() + (K+1)*c)
    res -= 0.5*tau**2/sigma_tau[0]**2      + (np.log(sigma_tau[0]) + c)
    res -= 0.5*np.sum(psi**2/sigma_psi**2) + (psi_dim/d)*(np.log(sigma_psi).sum() + d*c)
    return res


@njit
def log_lk_tau_xi_ind(y_mat, t_mat, n, z_vect, theta_full, i):
    """
    y_mat, z_vect, theta_full, t_mat: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    i: index of an individual.

    Computes log p(tau_i, xi_i, y_i | theta) for an individual i.

    NB: the posterior density has been integrated along psi.
    """

    ps0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta_full

    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0) - 1

    zi = unpack_ind(z_vect, i, K=K, d=d)
    tau, xi, psi = zi

    # [CAUTION] Here, we compute Di without psi, who has been integrated (cf. posterior formulas)
    z_i_full = get_taus(tau, xi, t0), xi, 0*psi
    c = 0.5*np.log(2*np.pi)
    res = 0
    D_tmp = np.zeros((n[i], d))
    for j in range(n[i]):
        D_tmp[j] = D_i_alt(ps0, t0, v0, z_i_full, t_mat[i,j])
    for k in range(d):
        Ak = 0
        Bk = 0
        n_ik = 0 # number of samples with non-missing coordinate k
        for j in range(n[i]):
            if not np.isnan(y_mat[i,j,k]):
                diff = y_mat[i,j,k] - D_tmp[j,k]
                Ak += diff
                Bk += diff**2
                n_ik += 1
        sigma_ik = np.sqrt(1/(n_ik/sigma[k]**2 + 1/sigma_psi[k]**2))
        res += 0.5*(Ak**2)*sigma_ik**2/sigma[k]**4  - 0.5*Bk/sigma[k]**2
        res += np.log(sigma_ik) - np.log(sigma_psi[k])
        res -= n_ik*(np.log(sigma[k]) + c)

    res -= 0.5*np.sum(xi**2/sigma_xi**2) + (np.log(sigma_xi).sum() + (K+1)*c)
    res -= 0.5*tau**2/sigma_tau[0]**2    + (np.log(sigma_tau[0]) + c)
    return res


@njit
def log_prior(theta, prior):
    """
    theta, prior: see NOTATIONS.md

    Computes log p(theta) (apart from constant normalizing terms depending on
    the prior).
    """

    sigma_p0,sigma_t0,sigma_v0,p0_bar,t0_bar,v0_bar,m,v = prior
    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    res = 0
    res -= 0.5*np.linalg.norm(p0-p0_bar)**2/sigma_p0**2
    res -= 0.5*np.linalg.norm(t0-t0_bar)**2/sigma_t0**2
    res -= 0.5*np.linalg.norm(v0-v0_bar)**2/sigma_v0**2

    res -= v[0]**2/(2*sigma_xi**2).sum()     + (m[3]+2)*np.log(sigma_xi).sum()
    res -= v[1]**2/(2*sigma_tau[0]**2)       + (m[3]+2)*np.log(sigma_tau[0])
    res -= v[2]**2/(2*sigma_psi**2).sum()    + (m[3]+2)*np.log(sigma_psi).sum()
    res -= v[3]**2/(2*sigma**2).sum()        + (m[3]+2)*np.log(sigma).sum()
    return res


@njit
def log_lk_full_split(y_mat, t_mat, n, z_vect, theta, prior, orthogonality_condition=False):
    """
    y_mat, z_vect, theta, t_mat, prior: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    orthogonality_condition: if True, psi_i is constrained to be orthogonal to each speed
                             vector v0[k].

    Computes log p(y, z, theta), split into N individual components, and a final
    component containing the unnormalized prior.
    """

    sigma_p0,sigma_t0,sigma_v0,p0_bar,t0_bar,v0_bar,m,v = prior
    N, d, K = len(y_mat), len(y_mat[0,0]), len(v0_bar) - 1
    psi_dim = d-K-1 if orthogonality_condition else d

    z = unpack_z(z_vect, K=K, d=d)
    z_full = get_z_full(z, theta)
    tau, xi, psi = z
    p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma = theta

    c = 0.5*np.log(2*np.pi)
    res = np.zeros(N+1)
    for i in range(N):
        D_tmp = np.zeros((n[i], d))
        for j in range(n[i]):
            D_tmp[j] = D_i(z_full, i, t_mat[i,j])
            for k in range(d):
                if not np.isnan(y_mat[i,j,k]):
                    res[i] -= 0.5*(y_mat[i,j,k]-D_tmp[j,k])**2/sigma[k]**2 + (np.log(sigma)[k] + c)
        res[i] -= 0.5*np.sum(xi[i]**2/sigma_xi**2)   + (np.log(sigma_xi).sum() + (K+1)*c)
        res[i] -= 0.5*tau[i]**2/sigma_tau[0]**2      + (np.log(sigma_tau[0]) + c)
        res[i] -= 0.5*np.sum(psi[i]**2/sigma_psi**2) + (psi_dim/d)*(np.log(sigma_psi).sum() + d*c)

    # res[-1] contains the prior terms
    res[-1] = log_prior(theta, prior)
    return res


@njit
def log_lk_full(y_mat, t_mat, n, z_vect, theta, prior, orthogonality_condition=False):
    """
    y_mat, z_vect, theta, t_mat, prior: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2),
    orthogonality_condition: if True, psi_i is constrained to be orthogonal to each speed
                             vector v0[k].

    Computes log p(y, z, theta) (the prior is unnormalized).
    """
    return log_lk_full_split(y_mat, t_mat, n, z_vect, theta, prior, orthogonality_condition).sum()


########################################################
########### PROPORTION OF CAPTURED VARIANCE ############
########################################################


@njit
def total_variance(y_mat, n):
    """
    y_mat, z_full: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    For each data dimension, computes the variance of the non-missing observations.
    """

    N, _, d = y_mat.shape
    result = np.zeros(d)

    total_mean = np.zeros(d)
    for k in range(d):
        nan_mask = np.isnan(y_mat[:,:,k])
        for i in range(N):
            if (~nan_mask[i,:n[i]]).sum() > 0:
                total_mean[k] += y_mat[i,:n[i],k][~nan_mask[i,:n[i]]].mean()
    total_mean /= N

    for k in range(d):
        nan_mask = np.isnan(y_mat[:,:,k])
        for i in range(N):
            if (~nan_mask[i,:n[i]]).sum() > 0:
                result[k] += ((y_mat[i,:n[i],k][~nan_mask[i,:n[i]]]-total_mean[k])**2).mean()
    result /= N

    return result


@njit
def explained_variance(y_mat, n, t_mat, z_full):
    """
    y_mat, t_mat, z_full: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    For each data dimension, computes the variance explained by the latent variables.
    """

    N, _, d = y_mat.shape
    result = np.zeros(d)

    value_counts = np.array([
        sum([(~np.isnan(y_mat[i,:n[i],k])).sum() for i in range(N)])
    for k in range(d)])

    total_mean_D = np.zeros(d)
    D_obs = np.zeros_like(y_mat)

    for i in range(N):
        for j in range(n[i]):
            D_obs[i,j] = D_i(z_full, i, t_mat[i,j])
            for k in range(d):
                if not np.isnan(y_mat[i,j,k]):
                    total_mean_D[k] += D_obs[i,j,k]
    total_mean_D /= value_counts

    for i in range(N):
        for j in range(n[i]):
            for k in range(d):
                if not np.isnan(y_mat[i,j,k]):
                    result[k] += (D_obs[i,j,k]-total_mean_D[k])**2

    return result / value_counts


@njit
def unexplained_variance(y_mat, n, t_mat, z_full):
    """
    y_mat, t_mat, z_full: see NOTATIONS.md,
    n: list of observation sizes (for instance, n[2] is the number of
       observations for individual 2).

    For each data dimension, computes the remaining variance unexplained by
    the latent variables.
    """

    N, _, d = y_mat.shape
    result = np.zeros(d)

    value_counts = np.array([
        sum([(~np.isnan(y_mat[i,:n[i],k])).sum() for i in range(N)])
    for k in range(d)])

    for i in range(N):
        for j in range(n[i]):
            D_obs = D_i(z_full, i, t_mat[i,j])
            for k in range(d):
                if not np.isnan(y_mat[i,j,k]):
                    result[k] += (y_mat[i,j,k]-D_obs[k])**2
    return result / value_counts
