import argparse
import dynesty
import random
import numpy as np
import lightkurve as lk
import matplotlib.pyplot as plt
from src.utils import compute_period_t0, summarize_dynesty_results, loglikelihood, prior
from src.plot_utils import box_plot_periodogram, plot_folded_transit, plot_section_fits
import dynesty.plotting

if __name__ == "__main__":
    np.random.seed(0)
    random.seed(0)
    generator = np.random.default_rng(0)
    parser = argparse.ArgumentParser(
        prog="Transit Fitting",
        usage="python -m src.fit_transit <data_path> <out_path> <--print_progress>",
        description="A script that fits transit",
    )

    parser.add_argument("data_path", help="The location of the data files")
    parser.add_argument("out_path", help="Where to put any accompanying plots")
    parser.add_argument("--print_progress", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    og_data = np.loadtxt(f"{args.data_path}/ex1_tess_lc.txt")
    data = np.loadtxt(f"{args.data_path}/ex1_filtered.txt")

    masks = [
        (data[:, 0] < 1500),
        (data[:, 0] < 1500) & (data[:, 0] > 1415),
        (data[:, 0] > 1500),
        (data[:, 0] > 2150),
    ]

    (period, t0), periodogram = compute_period_t0(data, component_masks=masks)

    box_plot_periodogram(periodogram, f"{args.out_path}/transit_periodogram.png")

    # Fold light curve for better parameter estimation
    lc = lk.LightCurve({"time": data[:, 0], "flux": data[:, 1], "flux_err": data[:, 2]})
    folded_lc = lc.fold(period, t0)
    time = folded_lc.time.value
    flux = folded_lc.flux.value
    flux_err = folded_lc.flux_err.value
    signal_mask = abs(time) <= 0.1

    # Perform nested sampling to sample from the posterior
    print("Fitting dynesty: (takes a long time)")
    ll = lambda x: loglikelihood(
        time[signal_mask],
        flux[signal_mask],
        flux_err[signal_mask],
        x,
        fix_period=period,
    )
    sampler = dynesty.NestedSampler(
        ll, prior, 8, nlive=1600, sample="rwalk", rstate=generator
    )
    sampler.run_nested(dlogz=1e-3, print_progress=args.print_progress)
    sresults = sampler.results

    param_means = summarize_dynesty_results(sresults, t0)
    param_means = np.append(param_means, period)

    # Corner plot
    dynesty.plotting.cornerplot(
        sresults, labels=[r"$T_0$", "i", "a", r"$R_p$", r"$u_1$", r"$u_2$", "e", "w"]
    )
    plt.savefig(f"{args.out_path}/posterior_cornerplot.png")
    fig = plt.subplots(8, 2, figsize=(16, 32))

    # Trace plot
    dynesty.plotting.traceplot(
        sresults,
        quantiles=(0.16, 0.5, 0.84),
        title_quantiles=(0.16, 0.5, 0.84),
        labels=[r"$T_0$", "i", "a", r"$R_p$", r"$u_1$", r"$u_2$", "e", "w"],
        fig=fig,
    )
    plt.savefig(f"{args.out_path}/posterior_traceplot.png")

    # Fit plots

    plot_folded_transit(
        (time + t0)[abs(time) < 0.1],
        flux[abs(time) < 0.1],
        param_means,
        out_path=f"{args.out_path}/folded_transit.png",
    )

    # Plot fits of 1st section
    main_mask = og_data[:, 0] < 1500
    region_mask_1 = og_data[:, 0] > 1420
    masks_1 = [(~region_mask_1[main_mask]), (region_mask_1[main_mask])]
    plot_section_fits(
        og_data[main_mask],
        data[data[:, 0] < 1500],
        param_means,
        masks_1,
        out_path=f"{args.out_path}/fits_sec_1.png",
    )

    # Plot partial fit of 1st sec
    main_mask_par1 = (og_data[:, 0] < 1413.5) & (og_data[:, 0] > 1412)
    main_mask_par2 = (data[:, 0] < 1413.5) & (data[:, 0] > 1412)
    plot_section_fits(
        og_data[main_mask_par1],
        data[main_mask_par2],
        param_means,
        [main_mask_par1[main_mask_par1]],
        out_path=f"{args.out_path}/fits_sec_1_partial.png",
    )

    # Plot fits of 2nd section
    region_mask_2 = og_data[:, 0] > 2157
    masks_2 = [(~region_mask_2[~main_mask]), (region_mask_2[~main_mask])]
    plot_section_fits(
        og_data[~main_mask],
        data[data[:, 0] > 1500],
        param_means,
        masks_2,
        out_path=f"{args.out_path}/fits_sec_2.png",
    )
