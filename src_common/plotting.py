"""
This file contains several functions to:
- display the convergence of the MCMC-SAEM algorithm
- plot and compare population trajectories
- plot and compare individual trajectories with the observations
"""


import numpy as np
import matplotlib.pyplot as plt

import src_base.model


def plot_convergence(history, d, K, true_theta=None, output=None):
    """
    history: object produced by src_exp.saem.MCMC_SAEM,
    d: data dimension,
    K: number of breaks,
    true_theta: true parameter to compare with the convergence result,
    output: if given a string, the function saves the plot at the given location.

    Plots the evolution of the components in theta. Components of p0 are shown
    in red, t0 in green and v0 in blue.

    Each subplots shows both the evolution of a parameter feature (in black) and
    its exponential latent variable twin (in color).
    """

    colors = ["red"]*d + ["green"]*max(K, 1) + ["blue"]*(d*(K+1))
    labels = ["p0"]*d + ["t0"]*max(K, 1) + ["v0"]*(d*(K+1))
    m = len(colors)

    size = int(np.sqrt(m+1))+1
    plt.figure(figsize=(2*size,2*size))

    if true_theta is not None:
        theta_vect = src_base.model.pack_theta(true_theta)

    for i in range(m):
        plt.subplot(size,size,i+1)
        plt.plot([z[i] for z in history["z"]], color=colors[i], label=labels[i], alpha=0.5)
        plt.plot([theta[i] for theta in history["theta"]], color="black", label=f"{labels[i]}_bar", alpha=0.5)

        if true_theta is not None:
            plt.axhline(theta_vect[i], color="gray", linestyle="--")

        plt.legend()

    plt.subplot(size,size,m+1)
    n_iter = len(history["log_likelihood"])
    plt.plot(history["log_likelihood"][n_iter//10:], color="orange", label="Full log-likelihood")
    plt.legend()

    plt.suptitle("Parameter convergence along the MCMC-SAEM algorithm")
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, bbox_inches="tight")
    plt.show()


def plot_population_trajectories(Theta, labels=None, feature_names=None, output=None):
    """
    Theta: list of possible values for the model parameters (at most 10 models),
    labels: list of labels for each model,
    feature_names: list of names for each feature,
    output: if given a string, the function saves the plot at the given location.

    Displays the population trajectory for all parameters in the list.
    """

    colors = ["black", "red", "blue", "green", "orange",
              "purple", "dimgray", "cyan", "magenta", "lime"]

    M = len(Theta)
    d = len(Theta[0][0]) # data dimension

    if labels is None:
        labels = [None]*len(Theta)

    if feature_names is None:
        feature_names = [f"Feature {k}" for k in range(d)]

    times = [] # time points to use for each model
    breakpoints = [] # list of trajectory breakpoints for each model
    for m in range(M):
        p0, t0, v0, _, _, _, _ = Theta[m]

        time_range = t0[-1] - t0[0]
        T = np.concatenate(([t0[0]-time_range/2], t0, [t0[-1]+time_range/2]))
        times.append(T)

        B = src_base.model.get_breakpoint_positions(p0, t0, v0)
        breakpoints.append(B)

    fig, axes = plt.subplots(nrows=1, ncols=d+1, figsize=(3*d+2, 3))

    Lines = []
    Labels = []
    for m in range(M):
        # Compute the population trajectory for model m
        p0, t0, v0, _, _, _, _ = Theta[m]
        trajectory = np.array([src_base.model.D(breakpoints[m], t0, v0, t) for t in times[m]])

        # Display the trajectory on each coordinate k
        for k in range(d):
            # Plot the trajectory
            label = labels[m] if k==0 else None
            axes[k].plot(times[m], trajectory[:,k], label=label, color=colors[m], zorder=m)

            K = len(v0) - 1 # number of breaks
            if K > 0: # If there are some breaks
                # Show the breakpoints
                axes[k].scatter(t0, breakpoints[m][:,k], color=colors[m], zorder=m)


    line, label = axes[0].get_legend_handles_labels()
    Lines = line + Lines
    Labels = label + Labels

    for k in range(d):
        axes[k].set_title(feature_names[k], fontsize=11)
    axes[-1].axis("off")

    if len(Lines) > 0:
        prop = 1-0.9/(d+1)
        fig.legend(Lines, Labels, loc=(prop, 0.5))
    plt.suptitle("Population trajectory", x=(1-1/(d+1))/2)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, bbox_inches="tight")
    plt.show()


def plot_individual_trajectories(y, t, indices, Z=[], Theta=[], labels=None, feature_names=None, output=None):
    """
    y: list of observations,
    t: list of times,
    Z: list of possible values for the latent variables parameters, either in
       base format or in exponential format (refer to NOTATIONS.md),
    Theta: list of possible values for the model parameters (at most 10 models),
    indices: list of indices of individuals to display,
    labels: list of labels for each model,
    feature_names: list of names for each feature,
    output: if given a string, the function saves the plot at the given location.

    Displays individual trajectories for all given models and individuals.
    """

    colors = ["black", "red", "blue", "green", "orange",
              "purple", "dimgray", "cyan", "magenta", "lime"]
    scatter_color = "tab:cyan"

    M = len(Theta)
    d = len(y[0][0]) # data dimension

    if labels is None:
        labels = [None]*len(Theta)

    if feature_names is None:
        feature_names = [f"Feature {k}" for k in range(d)]

    # First, convert all latent variables in z_full base model format
    Z_full = []
    for m in range(M):
        if len(Z[m])==3: # if Z[m] is in base model format
            z_full = src_base.model.get_z_full(Z[m], Theta[m])
        else: # if Z[m] is in exponential model format
            z_full = src_base.model.get_z_full(Z[m][3:], Theta[m])
        Z_full.append(z_full)

    # Next, compute the time points to be used in the pots
    times = [] # time points to use for each model and individual
    for m in range(M):
        t0 = Theta[m][1]
        taus = Z_full[m][3]

        times_model = []
        for ind in indices:
            t0_reparam = t0 + taus[ind, 1:] # break positions for the individual
            time_range = t0_reparam[-1] - t0_reparam[0]
            T_min = min(t0_reparam[0], t[ind][0]) - time_range/2
            T_max = max(t0_reparam[-1], t[ind][-1]) + time_range/2
            T = np.concatenate(([T_min], t0_reparam, [T_max]))
            times_model.append(T)
        times.append(times_model)

    fig, axes = plt.subplots(nrows=len(indices), ncols=d+1, figsize=(2*d+2, 2*len(indices)))

    # Containers for the legend components
    Lines = []
    Labels = []

    for i, ind in enumerate(indices):
        # Show the model trajectory
        for m in range(M):
            # Compute the individual trajectory for model m and individual ind
            trajectory = np.array([src_base.model.D_i(Z_full[m], ind, t) for t in times[m][i]])

            # Display the trajectory on each coordinate k
            for k in range(d):
                coord = k if len(indices)==1 else (i,k)

                # Plot the trajectory
                label = labels[m] if k==0 else None
                axes[coord].plot(times[m][i], trajectory[:,k], label=label,
                                 color=colors[m], zorder=m+1)

                K = len(Theta[m][2]) - 1 # number of breaks
                if K > 0: # If there are some breaks
                    # Show the breakpoints
                    axes[coord].scatter(times[m][i][1:-1], trajectory[1:-1,k],
                                        color=colors[m], zorder=m+1)

        # Show the data points
        for k in range(d):
            not_nan = np.where(~np.isnan(y[ind][:,k]))[0]
            coord = k if len(indices)==1 else (i,k)
            label = "Observations" if (i==0 and k==0) else None
            axes[coord].scatter(t[ind][not_nan], y[ind][not_nan,k],
                                color=scatter_color, zorder=0, label=label)

            if M==0:
                # If no model is given, harmonize the x-limits of the plots
                axes[coord].set_xlim(min(t[i])-0.5, max(t[i])+0.5)
            if (i==0 and k==0):
                line, label = axes[coord].get_legend_handles_labels()
                Lines = line + Lines
                Labels = label + Labels

    for k in range(d):
        coord = k if len(indices)==1 else (0,k)
        axes[coord].set_title(feature_names[k], fontsize=11)
    for i in range(len(indices)):
        coord = i if len(indices)==1 else (i,0)
        axes[coord].set_ylabel(f"Individual {indices[i]}", fontsize=11)

        # Hide the last column, used to display the legend
        coord = -1 if len(indices)==1 else (i,-1)
        axes[coord].axis("off")

    if len(Lines) > 0:
        prop = 1-0.9/(d+1)
        fig.legend(Lines, Labels, loc=(prop, 0.5))
    plt.suptitle("Individual trajectories", x=(1-0.65/(d+1))/2)
    plt.tight_layout()
    if output is not None:
        plt.savefig(output, bbox_inches="tight")
    plt.show()
