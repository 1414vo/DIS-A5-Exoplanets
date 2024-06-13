from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    ConstantKernel,
    WhiteKernel,
)
import numpy as np
from numpy.typing import ArrayLike
import pickle
from scipy.stats import norm, uniform, loguniform
import batman


def substract_activity(
    data: ArrayLike,
    transit_mask: ArrayLike,
    data_path: str = "./activity_fit.txt",
    hyperparam_path="./gp_transit_hyperparam.pkl",
    reset=False,
) -> None:
    """! Removes stellar activity from the transit data and stores the fit hyperparams.

    @param data             The light curve of the stellar system.
    @param transit_mask     A boolean mask containing the transits.
    @param data_path        Where to save or load prior parts of the curve.
    @param hyperparam_path  Where to store the fit hyperparameters.
    @param reset            Whether to rewrite previous entries in the data file.
    """
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    y_mean = y.mean()
    y = y - y_mean

    stride = len(y) // 600 + 1
    X_train = X[~transit_mask][::stride]
    y_train = y[~transit_mask][::stride]
    y_err = data[~transit_mask][::stride, 3]

    # Create Quasi-Periodic Kernel as a product of other kernels components
    k1 = ConstantKernel(constant_value_bounds=(1e-6, 2e-1))
    k2 = RBF(length_scale_bounds=(1e-2, 1e2))
    k3 = ExpSineSquared(periodicity_bounds=(0.6, 3.0), length_scale_bounds=(1e-2, 1e3))
    jitter = WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-11, 1))

    kernel = k1 * k2 * k3 + jitter
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=y_err.var(), n_restarts_optimizer=9, random_state=0
    )

    # Fit GP regression
    gp.fit(X_train, y_train)
    print("Fit Complete")

    fit, sigma = gp.predict(X, return_std=True)
    fit = fit + y_mean
    clean_flux = y - fit
    fit_data = np.column_stack((X[:, 0], clean_flux, sigma))
    hyperparams = {
        "A": [gp.kernel_.k1.k1.k1.get_params()["constant_value"]],
        "P": [gp.kernel_.k1.k2.get_params()["periodicity"]],
        "gamma": [2 / gp.kernel_.k1.k2.get_params()["length_scale"] ** 2],
        "length_scale": [2 / gp.kernel_.k1.k1.k2.get_params()["length_scale"] ** 2],
    }

    if not reset:
        prev_data = np.loadtxt(data_path)
        fit_data = np.vstack((prev_data, fit_data))
        with open(hyperparam_path, "rb") as f:
            prev_hyper = pickle.load(f)
            hyperparams = {
                k: hyperparams[k] + prev_hyper[k] for k in hyperparams.keys()
            }

    with open(data_path, "w") as f:
        np.savetxt(f, fit_data)

    with open(hyperparam_path, "wb") as f:
        pickle.dump(hyperparams, f)


class Prior:
    """! A class for modelling prior distributions and sampling from them.

    @param type     The type of prior. Currently supports the "uniform", "log-uniform" (Jeffrey's),
    and "gaussian" priors.
    @param bounds   Any relevant parameters for the distribution. Includes bounds for uniform distribution
    or mean and std for the Gaussian"""

    def __init__(self, type, bounds):
        """! Creates a Prior object for a parameter.

        @param type     The type of prior. Currently supports the "uniform", "log-uniform" (Jeffrey's),
        and "gaussian" priors.
        @param bounds   Any relevant parameters for the distribution. Includes bounds for uniform distribution
        or mean and std for the Gaussian"""

        assert type in [
            "uniform",
            "log-uniform",
            "gaussian",
        ], 'Possible Prior types include "uniform", "log-uniform" or "gaussian"'
        self.bounds = bounds
        self.type = type

    def nll(self, X: ArrayLike) -> float:
        """! Computes the negative log probability for a set of points.

        @param X    The input data.
        @returns    The negative log probability"""

        if not hasattr(X, "__iter__"):
            X = np.array([X])

        if self.type == "uniform":
            return -np.sum(
                uniform.logpdf(
                    X, loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0]
                )
            )
        elif self.type == "log-uniform":
            return -np.sum(loguniform.logpdf(X, self.bounds[0], self.bounds[1]))
        elif self.type == "gaussian":
            return -np.sum(norm.logpdf(X, loc=self.bounds[0], scale=self.bounds[1]))
        return None

    def __call__(self, u: float) -> float:
        """! Samples from the prior using the PPF of the distribution.

        @param u    A number in the [0, 1] region.
        @returns    The sampled parameter."""
        if self.type == "uniform":
            return uniform.ppf(
                u, loc=self.bounds[0], scale=self.bounds[1] - self.bounds[0]
            )
        elif self.type == "log-uniform":
            return loguniform.ppf(u, self.bounds[0], self.bounds[1])
        elif self.type == "gaussian":
            return norm.ppf(u, loc=self.bounds[0], scale=self.bounds[1])
        return None


def batman_mod(theta, t):
    t0, per, incl, ecc, w, aRs, Rp_Rs, u1, u2 = theta

    params = batman.TransitParams()  # object to store transit parameters

    params.t0 = t0  # time of inferior conjunction
    params.per = per  # orbital period
    params.rp = Rp_Rs  # planet radius (in units of stellar radii)
    params.a = aRs  # semi-major axis (in units of stellar radii)
    params.inc = incl  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.limb_dark = "quadratic"  # limb darkening model
    params.u = [u1, u2]  # limb darkening coefficients [u1, u2]

    mod = batman.TransitModel(params, t)
    flux_mod = mod.light_curve(params)
    return flux_mod
