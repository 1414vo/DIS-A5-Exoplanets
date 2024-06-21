from scipy.stats import norm, uniform, loguniform, beta
import numpy as np
from numpy.typing import ArrayLike


class Prior:
    """! A class for modelling prior distributions and sampling from them.

    @param type     The type of prior. Currently supports the "uniform", "log-uniform" (Jeffrey's),
    and "gaussian" priors.
    @param bounds   Any relevant parameters for the distribution. Includes bounds for uniform distribution
    or mean and std for the Gaussian"""

    def __init__(self, type, bounds):
        """! Creates a Prior object for a parameter.

        @param type     The type of prior. Currently supports the "uniform", "log-uniform" (Jeffrey's),
        , "gaussian" and "beta" priors.
        @param bounds   Any relevant parameters for the distribution. Includes bounds for the uniform distributions
        or mean and std for the Gaussian"""

        assert type in [
            "uniform",
            "log-uniform",
            "gaussian",
            "beta",
        ], 'Possible Prior types include "uniform", "log-uniform", "gaussian" or "beta"'
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
        elif self.type == "beta":
            return -np.sum(beta.logpdf(X, a=self.bounds[0], b=self.bounds[1]))
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
        elif self.type == "beta":
            return beta.ppf(u, a=self.bounds[0], b=self.bounds[1])
        return None
