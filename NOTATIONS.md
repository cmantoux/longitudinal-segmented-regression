# Notations guide

## Naming conventions for the model variables

We briefly give the naming conventions and definitions for the main variables in the code.

- `y` is the list of all subject data: the data of each subject is a matrix containing the sequence of all its observations, some of it possibly missing (missing data are given by `np.nan`),
- `y_mat` is an equivalent representation of `y` as a tensor, such that `y_mat[i,j,k]` gives the k-th coordinate of the j-th point of the i-th individual,
- `t` is a list of lists, giving the time of each observation for each individual,
- `t_mat` is an equivalent representation of `t` as a matrix, such that `t_mat[i,j]` gives the j-th time point of the i-th individual,
- `n` list of observation counts: for instance, `n[3]` gives the number of time points for individual number 3,
- `z` is the latent variable of the model: depending on the model used (base model or its exponentialized counterpart), its definition may vary (see below),
- `theta` is the parameters of the model,
- `prior` is the prior for the model parameters.

The model parameters `theta` contain the following variables:

- `p0`: the first breakpoint in the trajectory (empty if no breakpoints),
- `t0`: the list of breakpoint times in the trajectory (if no breakpoints, `t0` still contains a single reference time used in the model),
- `v0`: the list of speed vectors for each piece of the population trajectory,
- `sigma_xi, sigma_tau, sigma_psi, sigma`: the standard deviation for the variables `tau, xi, psi` and the observation noise.

The latent variable `z` contains the following variables:

- `tau`: the list of time shifts for each individual,
- `xi`: the list of logarithmic acceleration factors for each individual,
- `psi`: the list of spatial shift for each individual.

Finally, the `prior` variable contains the following variables:

- `p0_bar, t0_bar, v0_bar`: prior mean for the population trajectory parameters,
- `sigma_p0, sigma_t0, sigma_v0`: prior (scalar valued) standard deviation for `p0_bar, t0_bar, v0_bar`,
- `m, v`: lists of scalars parameterizing the Inverse Gamma distribution prior on `sigma_xi, sigma_tau, sigma_psi, sigma` (in this order).

In the exponentialized model, `z` additionally contains a duplicate of the population trajectory parameters `p0, t0, v0`. The variable names also slightly change to take this new structure into account (see below).

## Ordering in tuple variables

### Main variables

In this code, the prior `prior`, the parameter `theta` and the latent variable `z` are represented as tuples to simplify the compatibility with `numba`. As a consequence, all these tuples have a canonical ordering that remains unchanged throughout the code. This ordering reads as follow:

```python
# base model
prior = sigma_p0, sigma_t0, sigma_v0, p0_bar, t0_bar, v0_bar, m, v
theta = p0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma
z = tau, xi, psi

# exponentialized model
prior = sigma_p0, sigma_t0, sigma_v0, p0_barbar, t0_barbar, v0_barbar,
         sigma_p0_bar, sigma_t0_bar, sigma_v0_bar, m, v
theta = p0_bar, t0_bar, v0_bar, sigma_xi, sigma_tau, sigma_psi, sigma
z = p0, t0, v0, tau, xi, psi
```

### Auxiliary variables

Although knowing all variable names is by no mean required to run the code and manipulate its results, three specific auxiliary variables are used extensively throughout the code, and their definition may help the user understanding its behavior.

#### `z_full`

The `z` variable is sometimes replaced with the auxiliary variable `z_full`, which contains additional information computed from `z` and possibly `theta`. `z_full` is computed by:

```python
# base model
z_full = src_base.model.get_z_full(z, theta)

# exponentialized model
z_full = src_base.model.get_z_full(z, theta)
```

In both models, `z_full` gives the tuple `(ps0, t0, v0, taus, xi, psi)`, with:

- `ps0` is the matrix such that `ps0[k]` gives the d-dimensional coordinates of the k-th breakpoint in the population trajectory,
- `taus` is the matrix such that `taus[i,k]` gives the time at which individual `i` reached breakpoint number `k` in its trajectory,
- `t0, v0, xi, psi` are left unchanged.

Using the `z_full` variable saves up computation time in the MCMC steps of the SAEM algorithm.

#### `z_vect`

`z_vect` is a vectorized version of the latent variable `z`. Storing `z` as a vector allows to perform elementary operations. The conversion from `z` to `z_vect` is simply computed by:

```python
# also works with src_exp
z_vect = src_base.model.pack_z(z)
z = src_base.model.unpack_z(z_vect, d, K) # d = data dimension ; K = number of trajectory breaks
```

Similarly, functions to vectorize and un-vectorize theta are given by:

```python
# also works with src_exp
theta_vect = src_base.model.pack_theta(theta)
theta = src_base.model.unpack_theta(theta_vect, d, K) # d = data dimension ; K = number of trajectory breaks
```

#### `theta_full`

Just like `z_full` is an augmented version of `z`, `theta_full` is an augmented version of `theta`. The only difference with `theta` is that `p0` (position of the first population breakpoint) is replaced with the list of all population breakpoints, `ps0`:

```python
theta_full = ps0, t0, v0, sigma_xi, sigma_tau, sigma_psi, sigma
```
