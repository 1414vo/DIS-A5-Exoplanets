# Example Keplerian fit configuration file

# Required packages for setup
import pandas as pd
import numpy as np
import radvel

starname = "CB 01223"
nplanets = 1
instnames = ["INST1", "INST2-4"]
ntels = len(instnames)
fitting_basis = "logper tc secosw sesinw k"
bjd0 = 0.0
planet_letters = {1: "b"}

# Load radial velocity data, in this example the data is contained in
# an ASCII file, must have 'time', 'mnvel', 'errvel', and 'tel' keys
# the velocities are expected to be in m/s
data = pd.read_csv(
    "./data/ex2_RVs.txt",
    comment="#",
    names=["time", "mnvel", "errvel", "FWHM", "FWHM_err", "BIS", "BIS_err", "tel"],
)
data.tel = np.where(data.tel == "INST1", "INST1", "INST2-4")

# Define prior centers (initial guesses) in a basis of your choice (need not be in the fitting basis)
anybasis_params = radvel.Parameters(
    nplanets, basis="logper tc secosw sesinw k", planet_letters=planet_letters
)  # initialize Parameters object

anybasis_params["logper1"] = radvel.Parameter(value=1.8)  # period of 1st planet
anybasis_params["tc1"] = radvel.Parameter(
    value=6000.0
)  # time of inferior conjunction of 1st planet
anybasis_params["secosw1"] = radvel.Parameter(value=0.0)  # eccentricity of 1st planet
anybasis_params["sesinw1"] = radvel.Parameter(
    value=0.0
)  # argument of periastron of the star's orbit for 1st planet
anybasis_params["k1"] = radvel.Parameter(
    value=1e-3
)  # velocity semi-amplitude for 1st planet


anybasis_params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
anybasis_params["curv"] = radvel.Parameter(value=0.0, vary=False)
time_base = 6000.0

gp_explength_mean = 31.0 * np.sqrt(2.0)  # sqrt(2)*tau in Dai+ 2017 [days]
gp_explength_unc = 10.0 * np.sqrt(2.0)
gp_perlength_mean = np.sqrt(1.0 / (2.0 * 0.01))  # sqrt(1/(2*gamma)) in Dai+ 2017
gp_perlength_unc = 1.0
gp_per_mean = 31.0
gp_per_unc = 10.0

hnames = {
    "INST1": [
        "gp_amp",  # GP variability amplitude
        "gp_per",  # GP variability period
        "gp_explength",  # GP non-periodic characteristic length
        "gp_perlength",
    ],  # GP periodic characteristic length
    "INST2-4": ["gp_amp", "gp_per", "gp_explength", "gp_perlength"],
}

anybasis_params["gp_amp"] = radvel.Parameter(value=0.1)
anybasis_params["gp_explength"] = radvel.Parameter(value=gp_explength_mean)
anybasis_params["gp_per"] = radvel.Parameter(value=gp_per_mean)
anybasis_params["gp_perlength"] = radvel.Parameter(value=gp_perlength_mean)

jit_guesses = {"INST1": 1e-5, "INST2-4": 1e-6}


def initialize_instparams(tel_suffix):
    indices = data.tel == tel_suffix

    anybasis_params["gamma_" + tel_suffix] = radvel.Parameter(
        value=np.mean(data.mnvel[indices]), vary=False
    )
    anybasis_params["jit_" + tel_suffix] = radvel.Parameter(
        value=jit_guesses[tel_suffix]
    )


for tel in instnames:
    initialize_instparams(tel)

# Convert input orbital parameters into the fitting basis
params = anybasis_params.basis.to_any_basis(anybasis_params, fitting_basis)


# Define prior shapes and widths here.
priors = []
priors += [radvel.prior.Jeffreys("gp_amp", 0.001, 1.0)]
priors += [radvel.prior.Gaussian("gp_explength", gp_explength_mean, gp_explength_unc)]
priors += [radvel.prior.Gaussian("gp_per", gp_per_mean, gp_per_unc)]
priors += [radvel.prior.Gaussian("gp_perlength", gp_perlength_mean, gp_perlength_unc)]
priors += [radvel.prior.Gaussian("logper1", 1.0, 1.0)]
priors += [radvel.prior.Gaussian("tc1", 6000.0, 3.0)]
priors += [radvel.prior.HardBounds("k1", 1e-5, 0.1)]
priors += [radvel.prior.HardBounds("secosw1", 0.0, 0.2)]
priors += [radvel.prior.HardBounds("sesinw1", 0.0, 0.2)]
priors += [radvel.prior.Jeffreys("jit_INST1", 1e-7, 1.0)]
priors += [radvel.prior.Jeffreys("jit_INST2-4", 1e-7, 1.0)]

# optional argument that can contain stellar mass in solar units (mstar) and
# uncertainty (mstar_err). If not set, mstar will be set to nan.
stellar = dict(mstar=0.69, mstar_err=0.01)
