import numpy as np
import radvel.kepler as kepler
import radvel
from .utils import Prior

gp_explength_mean = 31.0 * np.sqrt(2.0)
gp_explength_unc = 10.0 * np.sqrt(2.0)
gp_perlength_mean = np.sqrt(1.0 / (2.0 * 0.01))
gp_perlength_unc = 1.0
gp_per_mean = 31.0
gp_per_unc = 10.0


def multiplanet_rv(t, params, vector):
    vel = np.zeros(len(t))
    params_synth = params.basis.v_to_synth(vector)

    for num_planet in range(params.num_planets):
        per = params_synth[5 * num_planet][0]
        tp = params_synth[5 * num_planet + 1][0]
        e = params_synth[5 * num_planet + 2][0]
        w = params_synth[5 * num_planet + 3][0]
        k = params_synth[5 * num_planet + 4][0]
        orbel_synth = np.array([per, tp, e, w, k])
        vel += kepler.rv_drive(t, orbel_synth)
    return vel


def setup_model(n_planets):
    params = radvel.Parameters(n_planets, basis="logper tc secosw sesinw k")
    for i in range(n_planets):
        params[f"logper{i + 1}"] = radvel.Parameter(value=1.8)
        params[f"tc{i + 1}"] = radvel.Parameter(value=6000.0)
        params[f"secosw{i + 1}"] = radvel.Parameter(value=0.0)
        params[f"sesinw{i + 1}"] = radvel.Parameter(value=0.0)
        params[f"k{i + 1}"] = radvel.Parameter(value=0.001)
    params["dvdt"] = radvel.Parameter(value=0.0, vary=False)
    params["curv"] = radvel.Parameter(value=0.0, vary=False)
    params["gp_amp"] = radvel.Parameter(value=0.1)
    params["gp_explength"] = radvel.Parameter(value=gp_explength_mean)
    params["gp_per"] = radvel.Parameter(value=gp_per_mean)
    params["gp_perlength"] = radvel.Parameter(value=gp_perlength_mean)
    return params


def setup_instances(t, v, v_err, tel, model):
    likes = []
    for tel_suffix in tel.unique():
        mask = tel == tel_suffix
        like = radvel.likelihood.GPLikelihood(
            model,
            t[mask],
            v[mask],
            v_err[mask],
            hnames=[
                "gp_amp",  # eta_1; GP variability amplitude
                "gp_explength",  # eta_2; GP non-periodic characteristic length
                "gp_per",  # eta_3; GP variability period
                "gp_perlength",  # eta_4; GP periodic characteristic length
            ],
            suffix="_" + tel_suffix,
            kernel_name="QuasiPer",
        )
        # Add in instrument parameters
        like.params["gamma_" + tel_suffix] = radvel.Parameter(
            value=np.mean(v[mask]), vary=False, linear=True
        )
        like.params["jit_" + tel_suffix] = radvel.Parameter(
            value=np.mean(abs(v_err[mask])), vary=True
        )
        likes.append(like)
    return radvel.CompositeLikelihood(likes)


def rv_priors(x, n_planets=1, n_tel=2):
    params = []
    # Planet params
    for i in range(n_planets):
        params.append(Prior("gaussian", (1.0, 1.0))(x[i * 5]))  # log_per_i
        params.append(Prior("gaussian", (6000.0, 3.0))(x[i * 5 + 1]))  # Tc
        params.append(Prior("uniform", (0.0, 0.2))(x[i * 5 + 2]))  # sqrt(e)cos(w)
        params.append(Prior("uniform", (0.0, 0.2))(x[i * 5 + 3]))  # sqrt(e)sin(w)
        params.append(Prior("uniform", (1e-5, 0.1))(x[i * 5 + 4]))  # k

    # GP parameters
    params.append(Prior("log-uniform", (1e-3, 1.0))(x[5 * n_planets]))  # gp amp
    params.append(
        Prior("gaussian", (gp_explength_mean, gp_explength_unc))(x[5 * n_planets + 1])
    )
    params.append(Prior("gaussian", (gp_per_mean, gp_per_unc))(x[5 * n_planets + 2]))
    params.append(
        Prior("gaussian", (gp_perlength_mean, gp_perlength_unc))(x[5 * n_planets + 3])
    )
    # Jitters
    for i in range(n_tel):
        params.append(
            Prior("log-uniform", (1e-7, 1.0))(x[5 * n_planets + 4 + i])
        )  # jitter 1
    return np.array(params)
