from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    ConstantKernel,
    WhiteKernel,
)
from astropy.timeseries import BoxLeastSquares
from dynesty.utils import resample_equal
import numpy as np
import batman
import pickle
from scipy.stats import norm, uniform, loguniform
from numpy.typing import ArrayLike
from typing import List, Tuple


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
    clean_flux = y - fit + y_mean
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


def batman_model(theta: ArrayLike, t: ArrayLike, fix_period: float = None) -> ArrayLike:
    """! Produces the batman light curve transit model for a given set of parameters.

    @param theta        Array of sampled parameters.
    @param t            The relevant timespan.
    @param fix_period   A value to which to fix the transit period. If None, expects the period
    to be part of the parameter set.

    @returns        The predicted flux at the given times"""

    params = batman.TransitParams()

    if fix_period is None:
        t0, incl, aRs, Rp_Rs, u1, u2, ecc, w, per = theta
        params.per = per  # orbital period
    else:
        t0, incl, aRs, Rp_Rs, u1, u2, ecc, w = theta
        params.per = 5.358736774937665  # orbital period

    params.t0 = t0  # time of inferior conjunction
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


def loglikelihood(t, y, y_err, theta):
    """! Computes the log likelihood for a given set of parameters.

    @param t        The timestamps of the measured data.
    @param y        The measured normalized flux.
    @param y_err    The errors in the normalized flux.
    @param theta    The set of parameters describing the light curve.

    @returns        The log-likelihood for the model given the data."""
    model_values = batman_model(theta, t)
    residuals = y - model_values
    logL = -len(t) / 2.0 * np.log(2 * np.pi)
    logL += -np.sum(np.log(y_err)) - np.sum(residuals**2 / (2 * y_err**2))
    return logL


def prior(x: ArrayLike, sample_period=False) -> List:
    """! Samples a set of parameters using a sample from a multidimensional
    uniform distribution.

    @param x                A uniform sample from the unit cube.
    @param sample_period    Whether to sample for the period parameter.

    @returns                A sampled set of parameters."""

    theta = []

    theta.append(Prior("gaussian", (0.0, 0.004))(np.array(x[0])))  # Prior on t0
    theta.append(Prior("uniform", (75, 90))(np.array(x[1])))  # Prior on inclination
    theta.append(Prior("uniform", (2.0, 100.0))(np.array(x[2])))  # Prior on aRs
    theta.append(Prior("log-uniform", (0.01, 0.3))(np.array(x[3])))  # Prioir on Rp/Rs
    theta.append(Prior("uniform", (0.0, 1.0))(np.array(x[4])))  # Prior on u1
    theta.append(Prior("uniform", (0.0, 1.0))(np.array(x[5])))  # Prior on u2
    theta.append(Prior("uniform", (0, 0.7))(np.array(x[6])))  # Prior on ecc
    theta.append(Prior("uniform", (-180, 180))(np.array(x[7])))  # Prior on w
    if sample_period:
        theta.append(
            Prior("gaussian", (5.359, 1e-3))(np.array(x[1]))
        )  # Prior on period[days]

    return theta


def summarize_dynesty_results(results, has_period: bool = False) -> None:
    """! Utility function for summarizing the dynesty nested sampling results.

    @param results      The results object from the dynesty sampling.
    @param has-period   Whether the period was fit as part of the curve.
    """
    weights = np.exp(results["logwt"] - results["logz"][-1])
    samples = results.samples

    # sample from posterior
    dynesty_samples = resample_equal(samples, weights)

    # print summary
    par = [np.mean(dynesty_samples[:, i]) for i in range(len(8))]
    t0, incl, a, rp, u1, u2, ecc, w = par
    t0_err, incl_err, a_err, rp_err, u1_err, u2_err, ecc_err, w_err = [
        np.quantile(dynesty_samples[:, i], [0.16, 0.84]) for i in range(len(8))
    ]

    print(f"T0 = {t0:10.6f} + {t0_err[1] - t0:8.6f}- {t0 - t0_err[0]:8.6f} BJD")
    print(
        f"Inclination = {incl:5.2f} + {incl_err[1] - incl:5.2f} - {incl - incl_err[0]:5.2f} deg"
    )
    print(f"a = {a:6.4f} + {a_err[1] - a:6.4f} - {a - a_err[0]:6.4f} R_star")
    print(f"Rp = {rp:8.6f} + {rp_err[1] - rp:8.6f} - {rp - rp_err[0]:8.6f} R_star")
    print(f"u1 = {u1:5.3f} + {u1_err[1] - u1:5.3f} - {u1 - u1_err[0]:5.3f}")
    print(f"u2 = {u2:5.3f} + {u2_err[1] - u2:5.3f} - {u2 - u2_err[0]:5.3f}")
    print(f"e = {ecc:5.3f} + {ecc_err[1] - ecc:5.3f} - {ecc - ecc_err[0]:5.3f}")
    print(f"w = {w:5.3f} + {w_err[1] - w:5.3f} - {w - w_err[0]:5.3f} deg")

    if has_period:
        per = np.mean(dynesty_samples[:, -1])
        per_err = np.quantile(dynesty_samples[:, -1])
        print(
            f"P = {per:6.4f} + {per_err[1] - per:6.4f} - {per - per_err[0]:6.4f} days"
        )


def compute_period_t0(
    data: ArrayLike, component_masks: ArrayLike
) -> Tuple[float, float]:
    durations = np.linspace(0.02, 0.20, 19)
    t0s = []

    for mask in component_masks:
        bls = BoxLeastSquares(t=data[mask, 0], y=data[mask, 1], dy=data[mask, 2])
        results = bls.autopower(durations, frequency_factor=6.0)
        t0s.append(results.transit_time[np.argmax(results.power)])

    period = results.period[np.argmax(results.power)]
    t0s = np.array(t0s)

    epoch = np.round((t0s - t0s[0]) / period)
    fit = np.polyfit(epoch, t0s, deg=1, cov=True)

    print("Prior T0/Period inference: ")
    print("--------------------------")
    print(f"T0 = 2457000+{fit[0][1]:.6f} +- {fit[1][1,1]**0.5:.6f} BJD")
    print(f"P = {fit[0][0]:.6f} +- {fit[1][0,0]**0.5:.6f} days")
    print("--------------------------\n")

    return fit[0]
