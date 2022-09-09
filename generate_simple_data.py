"""
This script generates several toy data sets that feature typical simple
trajectories. By default, the generated data is stored in "data/simple".
This location can be changed with the "--output" option.

This file is divided into three sections:
- A header for package import and argument parsing,
- Data set generation functions,
- Several applications of these functions to generate and save data sets.

The synthetic data sets are described below. Unless specified otherwise, on
average the observations are spread relatively evenly across the trajectory pieces.
- K2_small_noise  : 2D data set with K=2 breaks and small noise
- K2_big_noise    : 2D data set with K=2 breaks and a big noise
- K2_small_angle  : 2D data set with K=2 breaks, one of them harder to detect
- K2_change_speed : 2D data set with K=2 breaks, one of them only changing
                    the magnitude of the speed (not the direction)
- K2_uneven_occ   : 2D data set with K=2 breaks, with big difference in the
                    occupation rates of the three pieces.
- K3_small_noise  : 2D data set with K=3 breaks and small noise.
- K2_missing      : 2D data set with K=2 breaks and 40% missing data
- K2_missing_5D   : 5D data set with K=2 breaks and 40% missing data
- K2_missing      : 2D data set with K=0 breaks and 40% missing data
"""


#################################################
########## IMPORT AND ARGUMENT PARSING ##########
#################################################


import numpy as np
from tqdm.auto import *
import pickle
import argparse
import os

import src_base.model # contains basic functions


parser = argparse.ArgumentParser(description="Generates several toy data sets that feature typical simple trajectories.")
parser.add_argument("-o", "--output", type=str, default="data/simple", dest="output", help="Directory where the data will be stored (default: data/simple)")
parser.add_argument("-s", "--seed", type=int, default=0, dest="seed", help="Random seed")
args = parser.parse_args()

output_folder = args.output
os.makedirs(output_folder, exist_ok=True)

np.random.seed(0)


#################################################
########### DATA GENERATION FUNCTIONS ###########
#################################################


# Constant parameters used in the definition of the prior for the exponentialized model
sigma_p0 = 0.1
sigma_t0 = 0.1
sigma_v0 = 0.1
sigma_p0_bar = 4.
sigma_t0_bar = 4.
sigma_v0_bar = 4.
m, v = 6., 1.

def simulate(prior, N=100, poisson=3, variability=1):
    """
    Generates a data set from the given parameters.
    - prior: model prior; the model parameters are chosen as the mean values of the prior.
    - N: number of subjects
    - poisson: average number of observation per individual
    - variability: controls sigma_tau and sigma_psi
    - noise_ratio: proportion of noise in the data (between 0 and 1)
    - missing_prop: proportion
    """

    sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v = prior

    K = len(v0_barbar) - 1
    d = len(p0_barbar)

    p0_bar = p0_barbar.copy()
    t0_bar = t0_barbar.copy()
    v0_bar = v0_barbar.copy()

    sigma_xi = 0.3 * np.linspace(0.8,1.2,K+1) # heteroscedastic noise
    sigma_tau = 10 * variability * np.ones(1)
    sigma_psi = 2 * variability * np.linspace(0.8,1.2,d) # heteroscedastic noise
    sigma = 1 * variability * np.linspace(0.8,1.2,d) # heteroscedastic noise

    theta = (p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma)

    p0 = p0_bar
    t0 = t0_bar
    v0 = v0_bar

    xi = sigma_xi[None,:] * np.random.randn(N, K+1)
    psi = sigma_psi[None,:] * np.random.randn(N, d)
    tau = sigma_tau[0] * np.random.randn(N)

    # Enforce centered individual deviations
    xi -= xi.mean()
    psi -= psi.mean()
    tau -= tau.mean()

    z = (p0, t0, v0, tau, xi, psi)
    z_full = src_base.model.get_z_full(z[3:], theta)

    n = np.maximum(np.random.poisson(poisson, size=N), 1)
    t1, t2 = t0[0], t0[-1]

    t = []
    y = []
    for i in range(N):
        if K > 1:
            time_width = (t2-t1)*(K+1)/(K-1)
        else:
            time_width = 3*sigma_tau
        times = (np.arange(n[i])-n[i]/2)*time_width/poisson + (t1+t2)/2 + tau[i]
        times.sort()
        t.append(times)
        D_obs = np.array([src_base.model.D_i(z_full, i, t[i][j]) for j in range(n[i])])
        y.append(D_obs + sigma[None,:] * np.random.randn(n[i], d))

    return y, t, z, theta

def simulate_2(prior, N=100, poisson=3, variability=1):
    """
    Generates a data set from the given parameters.
    - prior: model prior; the model parameters are chosen as the mean values of the prior.
    - N: number of subjects
    - poisson: average number of observation per individual
    - variability: controls sigma_tau and sigma_psi
    - noise_ratio: proportion of noise in the data (between 0 and 1)
    - missing_prop: proportion

    /!\ This function is only used in the generation of the data set K2_uneven_occ.
    It only differs from the function simul in the spread of the time points.
    The code change is highlighted below.
    """

    sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v = prior

    K = len(v0_barbar) - 1
    d = len(p0_barbar)

    p0_bar = p0_barbar.copy()
    t0_bar = t0_barbar.copy()
    v0_bar = v0_barbar.copy()

    sigma_xi = 0.3 * np.linspace(0.8,1.2,K+1) # heteroscedastic noise
    sigma_tau = 10 * variability * np.ones(1)
    sigma_psi = 2 * variability * np.linspace(0.8,1.2,d) # heteroscedastic noise
    sigma = 1 * variability * np.linspace(0.8,1.2,d) # heteroscedastic noise

    theta = (p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma)

    p0 = p0_bar
    t0 = t0_bar
    v0 = v0_bar

    xi = sigma_xi[None,:] * np.random.randn(N, K+1)
    psi = sigma_psi[None,:] * np.random.randn(N, d)
    tau = sigma_tau[0] * np.random.randn(N)

    z = (p0, t0, v0, tau, xi, psi)
    z_full = src_base.model.get_z_full(z[3:], theta)

    n = np.maximum(np.random.poisson(poisson, size=N), 1)
    t1, t2 = t0[0], t0[-1]

    t = []
    y = []
    for i in range(N):
        if K > 1:
            time_width = (t2-t1)*(K+1)/(K-1)
        else:
            time_width = 3*sigma_tau
        ########### THE CHANGE IS HERE #############
        times = (np.arange(n[i])-n[i]/2)*time_width/poisson + 1.3*t2 + tau[i] # SHIFT TIME TO CREATE LESS OCCUPIED BLOCKS
        ########### THE CHANGE IS HERE #############
        times.sort()
        t.append(times)
        D_obs = np.array([src_base.model.D_i(z_full, i, t[i][j]) for j in range(n[i])])
        y.append(D_obs + sigma[None,:] * np.random.randn(n[i], d))

    return y, t, z, theta


#################################################
################ DATA GENERATION ################
#################################################

########## K2_small_noise ##########

simul_name = "K2_small_noise"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 100 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 5.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,2.]), np.zeros(d-2)]),
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7, variability=0.3)

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_big_noise ##########

simul_name = "K2_big_noise"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 150 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 5.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,2.]), np.zeros(d-2)]),
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7)

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_small_angle ##########

simul_name = "K2_small_angle"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 100 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 5.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,2.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.5,1.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7, variability=0.7)

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K3_small_noise ##########

simul_name = "K3_small_noise"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 3 # number of breakpoints
N = 100 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 5., 10.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,2.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.5,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([2.,1.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7, variability=0.5)

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_change_speed ##########

simul_name = "K2_change_speed"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 100 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 7.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,0.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.5,4.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7, variability=0.7)

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_uneven_occ ##########

simul_name = "K2_uneven_occ"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 100 # number of subjects

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 7.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,0.]), np.zeros(d-2)]),
    np.concatenate([np.array([1.,1.])/np.sqrt(2), np.zeros(d-2)]),
    np.concatenate([np.array([1.,0.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate_2(prior, N, poisson=7, variability=0.5)
n = list(map(len, y)) # Get the number of observations of each individual

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_missing ##########

simul_name = "K2_missing"
print(f"Generating {simul_name}.")

d = 2 # observation dimension
K = 2 # number of breakpoints
N = 400 # number of subjects
missing_proportion = 0.4 # proportion of randomly missing data

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([2., 5.]) # length = K
v0_barbar = np.array([
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
    np.concatenate([np.array([0.,2.]), np.zeros(d-2)]),
    np.concatenate([np.array([1.,1.]), np.zeros(d-2)]),
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=7)
n = list(map(len, y)) # Get the number of observations of each individual

for i in range(N):
    for j in range(n[i]):
        coord = np.random.randint(2)
        if np.random.rand() < missing_proportion:
            y[i][j,coord] = np.nan

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K2_missing_5D ##########

simul_name = "K2_missing_5D"
print(f"Generating {simul_name}.")

d = 5 # observation dimension
K = 2 # number of breakpoints
N = 400 # number of subjects
missing_proportion = 0.4 # proportion of randomly missing data

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([3., 8.]) # length = K
v0_barbar = np.array([
    [1.,1.,0.,2.,1.],
    [0.,2.,2.,2.,1.],
    [1.,1.,0.,-1.,1.],
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=10, variability=1)
n = list(map(len, y))

for i in range(N):
    for j in range(n[i]):
        mask = np.random.rand(d) < missing_proportion
        while mask.sum()==d: # Ensure that every row has at least one non missing coordinate
            mask = np.random.randint(2, size=d)
        for k in range(d):
            if mask[k]:
                y[i][j,k] = np.nan

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

########## K0_missing ##########

simul_name = "K0_missing"
print(f"Generating {simul_name}.")

d = 5 # observation dimension
K = 0 # number of breakpoints
N = 200 # number of subjects
missing_proportion = 0.4 # proportion of randomly missing data

p0_barbar = np.concatenate([np.array([1.,1.]), np.zeros(d-2)])
t0_barbar = np.array([3]) # length = max(K,1)
v0_barbar = np.array([
    [1.,2.,0.,-1.,1.],
])
prior = (sigma_p0,sigma_t0,sigma_v0,p0_barbar,t0_barbar,v0_barbar,sigma_p0_bar,sigma_t0_bar,sigma_v0_bar,m,v)

y, t, z, theta = simulate(prior, N, poisson=10, variability=1)
n = list(map(len, y)) # Get the number of observations of each individual

for i in range(N):
    for j in range(n[i]):
        mask = np.random.rand(d) < missing_proportion
        while mask.sum()==d: # Ensure that every row has at least one non missing coordinate
            mask = np.random.randint(2, size=d)
        for k in range(d):
            if mask[k]:
                y[i][j,k] = np.nan

f = open(f"{output_folder}/{simul_name}.pkl", "wb") # Read binary data
pickle.dump((y, t, z, theta), f)
f.close()

print("Done!")
