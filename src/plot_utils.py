from numpy.typing import NDArray
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from matplotlib import rcParams
import numpy as np
from numpy.typing import ArrayLike
from .utils import batman_model


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

    _, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=300)
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


def plot_folded_transit(
    time: ArrayLike, flux: ArrayLike, fit_params: ArrayLike, out_path: str = None
) -> None:
    __modify_params()

    plt.figure(figsize=(4, 4), dpi=300)

    plt.scatter(time, flux, alpha=0.5, color="gray", s=1)
    plt.plot(time, batman_model(fit_params, time), color="tab:orange", linestyle="--")
    plt.ylabel("Normalized flux")

    plt.xlabel("BJD - 2457000 [days]")
    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)


def plot_section_fits(og_data, filtered_data, fit_params, masks, out_path=None):
    __modify_params()
    _, axs = plt.subplots(2, 1, figsize=(8, 4), sharex=True, dpi=300)

    time_set = set(list(filtered_data[:, 0]))
    overlap_mask = np.array([t in time_set for t in og_data[:, 0]])

    fit = (
        og_data[overlap_mask, 1] - filtered_data[:, 1] + og_data[overlap_mask, 1].mean()
    )

    for mask in masks:
        axs[0].scatter(og_data[mask, 0], og_data[mask, 1], alpha=0.5, color="gray", s=1)
        axs[0].plot(
            filtered_data[mask[overlap_mask], 0],
            fit[mask[overlap_mask]],
            color="tab:orange",
            linestyle="--",
        )
        axs[0].fill_between(
            filtered_data[mask[overlap_mask], 0],
            fit[mask[overlap_mask]] - filtered_data[mask[overlap_mask], 2],
            fit[mask[overlap_mask]] + filtered_data[mask[overlap_mask], 2],
            color="tab:orange",
            alpha=0.3,
        )
        axs[0].set_ylabel("Normalized flux")

    X = np.linspace(filtered_data[0, 0], filtered_data[-1, 0], 2000)
    axs[1].scatter(
        filtered_data[:, 0], filtered_data[:, 1], alpha=0.5, color="gray", s=1
    )
    axs[1].plot(X, batman_model(fit_params, X), color="tab:orange", linestyle="--")
    axs[1].set_ylabel("Normalized flux")
    axs[1].set_ylim(0.995, 1.005)

    plt.xlabel("BJD - 2457000 [days]")
    plt.tight_layout()

    # Save or show dependent on the output path
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path)
