from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ExpSineSquared,
    ConstantKernel,
    WhiteKernel,
)
import numpy as np
from numpy.typing import NDArray


def substract_activity(
    data: NDArray, data_path: str = "./activity_fit.txt", reset=False
) -> None:
    """! Removes stellar activity from the transit data.

    @param data         The light curve of the stellar system.
    @param data_path    Where to save or load prior parts of the curve.
    @param reset        Whether to rewrite previous entries in the data file.
    """
    X = data[:, 0].reshape(-1, 1)
    y = data[:, 1]
    y_mean = y.mean()
    y = y - y_mean

    stride = len(y) // 10000 + 1
    X_train = X[::stride]
    y_train = y[::stride]
    y_err = data[::stride, 3]

    # Create Quasi-Periodic Kernel as a product of other kernels components
    k1 = ConstantKernel(constant_value_bounds=(1e-2, 2e-1))
    k2 = RBF(length_scale_bounds=(1e-2, 1e2))
    k3 = ExpSineSquared(periodicity_bounds=(0.6, 3.0), length_scale_bounds=(1e-2, 1e3))
    jitter = WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-11, 1))

    kernel = k1 * k2 * k3 + jitter
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=y_err.var(), n_restarts_optimizer=9
    )

    # Fit GP regression
    gp.fit(X_train, y_train)

    fit, sigma = gp.predict(X, return_std=True)
    fit = fit + y_mean
    fit_data = np.hstack((X, fit, sigma))

    if not reset:
        prev_data = np.readtxt(data_path)
        fit_data = np.vstack((prev_data, fit_data))

    with open(data_path, "w") as f:
        np.savetxt(f, fit_data)
