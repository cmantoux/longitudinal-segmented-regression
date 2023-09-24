"""
This file contains an example script on synthetic data.

The code features three applications:
- estimating the parameters with the MCMC-SAEM algorithm,
- plotting the estimation result and comparing with the ground truth,
- computing the estimated model's marginal likelihood

Each of these sections can be skipped. For instance, running:

    python script_demo.py --skip-estimation

will jump to the plotting section, using the ground truth instead
of the estimation result. Similarly, running

    python script_demo.py --skip-estimation --skip-plots

will only compute the model's marginal likelihood. Any combination
of sections can be excluded.

Remark: the iteration counts given below, i.e.:
- 100K for parameter estimation,
- 10K for average posterior mean computation,
- 100K for marginal likelihood computation
can be considered a reference point to run the algorithm on new data.
"""


#################################################
#################### IMPORTS ####################
#################################################


import numpy as np
import matplotlib.pyplot as plt
import argparse
import pickle

import src_exp.model # use the exponential model for the SAEM estimation
import src_exp.saem
import src_base.model # use the base model for posterior densities and model selection
import src_base.mcmc
import src_base.selection
import src_common.utils
import src_common.plotting as plotting


# Silence Numba warnings
import warnings
from numba.core.errors import NumbaPerformanceWarning
warnings.simplefilter("ignore", category=NumbaPerformanceWarning)


#################################################
########## HEADER AND ARGUMENT PARSING ##########
#################################################


n_iter_saem       = 100000 # number of SAEM steps for the parameter estimation
n_iter_posterior  = 10000  # number of MCMC iterations to estimate the posterior mean of (z|y)
n_iter_likelihood = 100000 # number of MCMC iterations used in the marginal likelihood computation
K      = 2  # number of breaks in the trajectory
seed   = 0  # global random seed
data   = "data/test.pkl" # path to the data file


parser = argparse.ArgumentParser(description="Performs parameter estimation and model selection on an input data set")
parser.add_argument("-d", "--data", type=str, default=data, dest="data", help="Input synthetic data file")
parser.add_argument("-s", "--seed", type=int, default=seed, dest="seed", help="Random seed")
parser.add_argument("-K", "--breaks", type=int, default=K, dest="K", help=f"Number of breaks in the population trajectory (default: {K})")
parser.add_argument("--skip-estimation", action=argparse.BooleanOptionalAction, dest="skip_estimation", default=False, help="Skip the estimation (use the true parameters)")
parser.add_argument("--skip-plots", action=argparse.BooleanOptionalAction, dest="skip_plots", default=False, help="Skip the plots")
parser.add_argument("--skip-likelihood", action=argparse.BooleanOptionalAction, dest="skip_likelihood", default=False, help="Skip the computation of the marginal likelihood")
args = parser.parse_args()


data = args.data
seed = args.seed
K    = args.K


#################################################
############ ESTIMATION PROCEDURE ###############
#################################################


src_common.utils.set_random_seed(seed) # Set all random seeds to 0

f = open(data, "rb") # rb: Read binary data
y, t, z, theta = pickle.load(f)
f.close()

d = len(y[0][0]) # data dimension

if not args.skip_estimation:
    # Run the estimation procedure, saving the SAEM progression every 100 steps
    print("Running the estimation procedure.")
    prior = src_exp.model.get_prior(K=K, d=d) # Prior of the exponential model
    z_MAP, theta_MAP, history = src_exp.saem.MCMC_SAEM(y, t, prior, n_iter=n_iter_saem,
                                                track_history=10)
    # Display the evolution of the convergence
    if not args.skip_plots:
        plotting.plot_convergence(history, d, K, true_theta=theta, output="demo/figures/convergence.png")

else:
    z_MAP, theta_MAP = z, theta

# Convert the exponential latent variable into the base model format
z_MAP_base = z_MAP[3:]


#################################################
############### TRAJECTORY PLOTS ################
#################################################


if not args.skip_plots:
    # First, we plot the average population trajectory D(t)
    plotting.plot_population_trajectories(Theta=[theta, theta_MAP], labels=["Ground truth", "Estimation"],
                                          output="demo/figures/population_trajectory.png")

    # Next, we compare individual trajectories with the ground truth
    indices = [0, 1, 2] # For instance, choose individuals 0, 1 and 2

    # To do so, we first compute the posterior mean of z given the observed data
    print("\nPosterior mean estimation of the individual latent variables.")
    z_posterior_mean = src_base.mcmc.posterior_z_mean(y, t, z_MAP_base, theta_MAP, n_iter=n_iter_posterior)

    plotting.plot_individual_trajectories(
        y, t, indices=indices, Z=[z, z_posterior_mean], Theta=[theta, theta_MAP],
        labels=["Ground truth", "Estimation"], output="demo/figures/individual_trajectories.png")


#################################################
############## MARGINAL LIKELIHOOD ##############
#################################################


if not args.skip_likelihood:
    # Estimates the marginal log-likelihood for each individual, and the related estimation error
    print("\nComputing an estimation of the marginal likelihood.")
    marginal_likelihood, likelihood_error = src_base.selection.compute_marginal_likelihood(
                y, t, z_MAP_base, theta_MAP, n_iter=n_iter_likelihood)
    print("Total marginal log-likelihood:", marginal_likelihood.sum())
    print("Error estimate:", likelihood_error.sum())

    # The marginal log-likelihood can then be used to compute information criteria
    N = len(y) # number of individuals
    n = [len(y[i]) for i in range(N)] # n[i] = number of observations for individual i
    AIC   = src_base.selection.AIC(marginal_likelihood, K=K, d=d) # Akaike Information Criterion
    BIC   = src_base.selection.BIC(marginal_likelihood, K=K, d=d, N=N) # Bayesian Information Criterion
    BIC_h = src_base.selection.BIC_hybrid(marginal_likelihood, K=K, d=d, N=N, n=n) # Hybrid BIC for mixed-effects models

    print("\nResulting information criteria:")
    print("AIC:  ", AIC)
    print("BIC:  ", BIC)
    print("BIC_h:", BIC_h)
