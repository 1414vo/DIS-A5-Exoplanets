import argparse
import numpy as np
from src.utils import substract_activity
from src.plot_utils import lomb_scargle

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Stellar Activity Remover",
        usage="python -m src.remove_activity <data_path> <out_path>",
        description="A script that removes the stellar activity of a light curve using Gaussian Processes",
    )

    parser.add_argument("data_path", help="The location of the data files")
    parser.add_argument("out_path", help="Where to put any accompanying plots")

    args = parser.parse_args()

    data = np.loadtxt(f"{args.data_path}/ex1_tess_lc.txt")

    period = lomb_scargle(
        data[:, 0], data[:, 1], data[:, 3], f"{args.out_path}/ex1_periodogram.png"
    )
    print("Lomb Scargle Periodogram estimation: ", period)

    data_r = np.arange(len(data))
    transit_mask = (
        ((data_r > 1220) & (data_r < 1275))
        | ((data_r > 4960) & (data_r < 5010))
        | ((data_r > 9240) & (data_r < 9300))
        | ((data_r > 13080) & (data_r < 13130))
        | ((data_r > 16470) & (data_r < 16525))
        | ((data_r > 20310) & (data_r < 20365))
        | ((data_r > 26435) & (data_r < 26490))
        | ((data_r > 30260) & (data_r < 30330))
    )
    component_masks = [
        (data[:, 0] < 1420),
        (data[:, 0] > 1424) & (data[:, 0] < 1500),
        (data[:, 0] > 1500) & (data[:, 0] < 2158),
        (data[:, 0] > 2158),
    ]

    outlier_mask = data[:, 1] > 1.015

    for i, component_mask in enumerate(component_masks):
        print(
            f"Fitting component {i + 1} of size {np.sum(component_mask & (~transit_mask))}"
        )
        mask = component_mask & ~outlier_mask
        component_data = data[mask]

        substract_activity(
            component_data,
            transit_mask[mask],
            f"{args.data_path}/ex1_filtered.txt",
            f"{args.out_path}/hyperparams.pkl",
            reset=i == 0,
        )
