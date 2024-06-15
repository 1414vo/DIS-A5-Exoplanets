from numpy.typing import NDArray
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from matplotlib import rcParams
import numpy as np


def __modify_params():
    """! Sets default matplotlib parameters."""
    rcParams["pdf.fonttype"] = 42
    rcParams["ps.fonttype"] = 42
    rcParams["font.family"] = "serif"
    rcParams["font.sans-serif"] = ["Palatio"]
    rcParams["text.usetex"] = True
    rcParams["text.latex.preamble"] = r"\usepackage{amsfonts}"


def lomb_scargle(data: NDArray, out_path: str = None) -> float:
    """! Plots the Lomb Scargle periodogram for the given data and
    retrieves the most likely period.

    @param data     The light curve time and flux.
    @param out_path Where to store the plot.

    @returns        The most likely period value."""
    __modify_params()

    plt.figure(figsize=(3, 3), dpi=150)
    frequency, power = LombScargle(data[:, 0], data[:, 1]).autopower()
    plt.plot(2 * frequency, power, label="Frequency power")

    plt.xlim(0, 2)
    plt.xlabel("Frequency [days]")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)

    return frequency[power.argmax()] * 2


def box_plot_periodogram(periodogram, out_path: str = None) -> None:
    # extract the period, duration, time, and depth of the best-fit transit
    period = periodogram.period[np.argmax(periodogram.power)]

    # Plot the periodogram

    _, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.plot(periodogram.period, periodogram.power, "k", lw=0.5)
    ax.set_xlim(periodogram.period.min(), periodogram.period.max())
    ax.set_xlabel("Period [days]")
    ax.set_ylabel("Log-likelihood")

    # Highlight the harmonics of the peak period
    ax.axvline(period, alpha=0.4, lw=4)
    for n in range(2, 10):
        ax.axvline(n * period, alpha=0.4, lw=1, linestyle="dashed")
        ax.axvline(period / n, alpha=0.4, lw=1, linestyle="dashed")

    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
