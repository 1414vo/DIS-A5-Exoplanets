import argparse
import dynesty
import pandas as pd
import radvel
import random
import numpy as np
from .rv_utils import setup_instances, setup_model, rv_priors
from .plot_utils import lomb_scargle, plot_trend

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

    data = pd.read_csv(
        f"{args.data_path}/ex2_RVs.txt",
        comment="#",
        names=["t", "vel", "vel_err", "FWHM", "FWHM_err", "BIS", "BIS_err", "tel"],
    )
    data.tel = np.where(data.tel == "INST1", "INST1", "INST2-4")

    plot_trend(
        data.t, data.vel, out_path=f"{args.out_path}/vel_trend.png", name="RV [km/s]"
    )
    plot_trend(
        data.t,
        data.FWHM,
        out_path=f"{args.out_path}/FWHM_trend.png",
        name="FWHM [km/s]",
    )
    plot_trend(
        data.t, data.BIS, out_path=f"{args.out_path}/BIS_trend.png", name="BIS [km/s]"
    )

    t = lomb_scargle(
        data.t,
        data.BIS,
        data.BIS_err,
        out_path=f"{args.out_path}/bis_periodogram.png",
        detrend=True,
        freqs=np.linspace(0.005, 0.1, 1000),
    )
    print(f"Lomb Scargle periodogram period estimate: {t} days")
    for n_planets in [0, 1, 2]:
        params = setup_model(n_planets=n_planets)
        model = radvel.RVModel(params)
        # The following line is duplicated due to a bug in RadVel that causes "gamma INST2-4" to vary
        like = setup_instances(data.t, data.vel, data.vel_err, data.tel, model)
        like = setup_instances(data.t, data.vel, data.vel_err, data.tel, model)

        sampler = dynesty.NestedSampler(
            like.logprob_array,
            lambda u: rv_priors(u, n_planets=n_planets),
            6 + 5 * n_planets,
            nlive=1600,
            sample="rwalk",
        )
        sampler.run_nested(dlogz=1e-3, print_progress=args.print_progress)
        # print(f'{n_planets} planet(s): log(Z) = {}')
