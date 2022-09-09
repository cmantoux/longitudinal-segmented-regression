"""
This script generates a large series of synthetic data sets with randomly
generated population trajectories. The script generates data sets with several
random seeds for various numbers of breaks and noise levels in given ranges.
By default, the generated data is stored in "data/simple". This location can be
changed with the "--output" option.

The two typical commands to use this file are:
python generate_series_data.py --observations increasing
python generate_series_data.py --observations constant

Depending on the "--observations" (or "--obs") argument, two series can be generated:
- If "--obs increasing" is used, the average number of observations per
    individual will grow with the number of breaks
- If "--obs constant" is used, the average number of observations per
    individual will remain constant as the number of breaks grows

This file is divided into four sections:
- A header for package import and argument parsing,
- A data set generation function,
- A main loop generating the data sets,
- Sanity check to empirically compute the proportion of variance captured by
the model in the simulated data, and the average occupation rates of the
simulated observations.

The synthetic data will be stored in files whose name describes their content.
For instance, for the file "series_cst/K3_series/K3_series_D1_V7.pkl":
- series_cst: the file was generated with the option "--obs constant"
              (series_inc is used instead for "--obs increasing")
- K3_series: the population trajectory has K=3 breaks
- D1: the difficulty level (i.e., the noise level in the data) is 1
      (difficulty goes from 0 to 5)
- V7: the ID of the data set (for each number of breaks and each difficulty
       level, 10 i.i.d. data sets are generated: ID goes from 0 to 9)
"""


#################################################
########## IMPORT AND ARGUMENT PARSING ##########
#################################################


import numpy as np
from numba import njit
from tqdm.auto import *
import pickle
import argparse
import os

import src_base.model # contains basic functions
import src_common.utils


d = 6 # data dimension
K_max = 5 # maximal number of breaks
N = 400 # number of individuals per data set
noise_ratios = np.linspace(0.1, 0.80, 5) # list of possible noise levels in the data, from 10% to 80%
missing_prop = 0.4 # proportion of missing data
seeds_per_noise_level = 10 # for each noise level and each number of breaks, 10 data sets are generated


"""
For instance, if:
- K_max = 4
- noise_ratios has length 6
- seeds_per_noise_level = 10
Then the script will generate a total of (4+1) * 6 * 10 = 300 data sets.
"""


parser = argparse.ArgumentParser(description="Generates a large series of synthetic data sets with varying number of observations.")
parser.add_argument("-o", "--output", type=str, default="", dest="output", help="Directory where the data will be stored (default: data/series_inc)")
parser.add_argument("-s", "--seed", type=int, default=0, dest="base_seed", help="Random seed")
parser.add_argument("--observations", "--obs", type=str, default="increasing", dest="observations", help="\"increasing\" or \"constant\"")
args = parser.parse_args()

# The poissons[K] variable contains, for a given value of the number of breaks K,
# the average number of observations for each individual
if args.observations=="increasing":
    poissons = [6, 8, 10, 12, 14, 16] # poisson[K] grows with K
elif args.observations=="constant":
    poissons = [16, 16, 16, 16, 16, 16] # poisson[K] stays constant
else:
    raise Exception("--observations argument must be either \"increasing\" or \"constant\".")

base_seed = args.base_seed # Global random seed used to generate the data

output_folder = args.output
if output_folder=="":
    if args.observations=="increasing":
        output_folder = "data/series_inc"
    else: # --> args.observations=="constant"
        output_folder = "data/series_cst"
os.makedirs(output_folder, exist_ok=True)

np.random.seed(0)


################################################
############### DATA GENERATION ################
################################################


def simulate_series(d, K, N=100, poisson=3, variability=1, noise_ratio=0.3, missing_prop=0, seed=0):
    """
    Generates a data set with randomly chosen parameters.
    - d: data dimension
    - K: number of breaks (can be zero)
    - N: number of subjects
    - poisson: average number of observation per individual
    - variability: controls sigma_tau and sigma_psi
    - noise_ratio: proportion of noise in the data (between 0 and 1)
    - missing_prop: proportion of missing data (between 0 and 1)
    - seed: random seed
    """

    np.random.seed((args.observations=="increasing")*10**6 + base_seed + seed)

    p0_bar = 5*np.random.randn(d)
    x = 3*np.abs(np.random.randn(max(K,1)))+3
    t0_bar = x.cumsum()
    v0_bar = 1.5*np.random.randn(K+1,d)

    sigma_xi = 0.3 * np.linspace(0.8,1.2,K+1)
    sigma_tau = 10 * variability * np.ones(1)
    sigma_psi = 2 * variability * np.linspace(0.8,1.2,d)
    # sigma is defined later ; it depends on the latent trajectory variance

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
    theta_tmp = (p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, 1)
    z_full = src_base.model.get_z_full(z[3:], theta_tmp)

    n = np.maximum(np.random.poisson(poisson, size=N), 1)
    t1, t2 = t0[0], t0[-1]

    t = []
    D_obs = []
    for i in range(N):
        if K > 1:
            time_width = (t2-t1)*(K+1)/(K-1)
        else:
            time_width = 10
        times = (np.arange(n[i])-n[i]/2)*time_width/poisson + (t1+t2)/2 + tau[i]
        times.sort()
        t.append(times)
        D_obs.append(np.array([src_base.model.D_i(z_full, i, t[i][j]) for j in range(n[i])]))

    t_mat = src_common.utils.get_t_mat(t) # Convert t to a rectangular matrix
    D_mat = np.zeros((N, n.max(), d))
    for i in range(N):
        D_mat[i, :n[i]] = D_obs[i]
    explained_var =  src_base.model.explained_variance(D_mat, n, t_mat, z_full)
    unexplained_var = explained_var * (noise_ratio/(1-noise_ratio))
    sigma = np.sqrt(unexplained_var) * np.linspace(0.8,1.2,d)

    theta = (p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma)

    y = []
    y_full = [] # y_full is a copy of y without missing values
    for i in range(N):
        y.append(D_obs[i] + sigma[None,:] * np.random.randn(n[i], d))
        y_full.append(y[-1])

    for i in range(N):
        for j in range(n[i]):
            mask = np.random.rand(d) < missing_prop
            while mask.sum()==d: # Ensure that every row has at least one non missing coordinate
                mask = np.random.randint(2, size=d)
            for k in range(d):
                if mask[k]:
                    y[i][j,k] = np.nan

    return y, y_full, t, z, theta


#################################################
################ DATA GENERATION ################
#################################################


print(f"Generating data from K=0 to K={K_max} breaks.")

for K in range(K_max+1):
    print(f"Generating data with K={K} breaks.")
    os.makedirs(f"{output_folder}/K{K}_series", exist_ok=True)
    for H in trange(len(noise_ratios)):
        for V in range(seeds_per_noise_level):
            seed = K*seeds_per_noise_level*len(noise_ratios)+seeds_per_noise_level*H+V
            y, y_full, t, z, theta = simulate_series(d=6, K=K, N=N, poisson=poissons[K],
                                         variability=1, noise_ratio=noise_ratios[H],
                                         missing_prop=missing_prop, seed=seed)
            f = open(f"{output_folder}/K{K}_series/K{K}_D{H}_V{V}.pkl", "wb") # wb = Write binary data
            pickle.dump((y, t, z, theta), f)
            f.close()
            # Aside of the data set, we also store the full (= non-missing) value of the data:
            f = open(f"{output_folder}/K{K}_series/K{K}_D{H}_V{V}_full.pkl", "wb") # wb = Write binary data
            pickle.dump(y_full, f)
            f.close()


#################################################
################# SANITY CHECKS #################
#################################################


print("Checking the occupation rates (showing the average occupation for each break)")

for K in range(1,K_max+1):
    occ = np.zeros((len(noise_ratios), seeds_per_noise_level, K+1))
    for H in range(len(noise_ratios)):
        for V in range(seeds_per_noise_level):
            f = open(f"{output_folder}/K{K}_series/K{K}_D{H}_V{V}.pkl", "rb")
            y, t, z, theta = pickle.load(f)
            f.close()

            occ[H, V] = src_base.model.occupation_rates(t, z[3:], theta)
    print(f"K={K}:", occ.mean(axis=(0,1)))


print("Checking the proportion of captured variance (showing the average proportion for each noise level)")

captured_variance = np.zeros((K_max+1,len(noise_ratios),seeds_per_noise_level,d))
full_variance = np.zeros((K_max+1,len(noise_ratios),seeds_per_noise_level,d))
for K in range(1,K_max+1):
    for H in range(len(noise_ratios)):
        for V in range(seeds_per_noise_level):
            f = open(f"data/series_inc/K{K}_series/K{K}_D{H}_V{V}.pkl", "rb")
            y, t, z, theta = pickle.load(f)
            f.close()

            z_full = src_base.model.get_z_full(z[3:], theta)

            n = np.array(list(map(len, y)))
            y_mat = src_common.utils.get_y_mat(y)
            t_mat = src_common.utils.get_t_mat(t)
            captured_variance[K,H,V] =  src_base.model.explained_variance(y_mat, n, t_mat, z_full)
            full_variance[K,H,V] =  src_base.model.total_variance(y_mat, n)
    print(f"K={K}:", (captured_variance[K]/full_variance[K]).mean(axis=(1,2)))

print("Done!")
