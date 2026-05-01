import numpy as np
import cmdstanpy
import pandas as pd
from simulator import generate_hierarchical, E_REF, SECONDS_PER_YEAR
from load_config import load_config

config = load_config()
hier_cfg = config["hierarchical"]
atmo_cfg = config["atmospheric_realism"]
det_cfg = config["detector_response"]
phys_prior_cfg = config["priors"]["physics"]
test_cfg = config["testing"]

rng = np.random.default_rng(int(test_cfg["seed_hierarchical_samplers"]))

hds = generate_hierarchical(
    K=int(hier_cfg["K"]),
    mu_phi=float(hier_cfg["mu_phi"]),
    sigma_phi=float(hier_cfg["sigma_phi"]),
    mu_eta=float(hier_cfg["mu_eta"]),
    sigma_eta=float(hier_cfg["sigma_eta"]),
    gamma=float(hier_cfg["gamma"]),
    delta=float(hier_cfg["delta"]),
    truth_model=hier_cfg["truth_model"],
    atmo_model=atmo_cfg["model"],
    prompt_fraction=float(atmo_cfg["prompt_fraction"]),
    eta_prompt=float(config["background"].get("eta_prompt", float(config["background"]["eta"]) * float(atmo_cfg["prompt_fraction"]))),
    delta_prompt=float(atmo_cfg["delta_prompt"]),
    E_knee=float(atmo_cfg["E_knee"]),
    knee_sharpness=float(atmo_cfg["knee_sharpness"]),
    prompt_prior_log_mean=float(atmo_cfg.get("prompt_prior_log_mean", np.log(1e-6))),
    prompt_prior_log_sd=float(atmo_cfg.get("prompt_prior_log_sd", 1.0)),
    prior_phi_log_mean=float(phys_prior_cfg["phi_log_mean"]),
    prior_phi_log_sd=float(phys_prior_cfg["phi_log_sd"]),
    prior_gamma_mean=float(phys_prior_cfg["gamma_mean"]),
    prior_gamma_sd=float(phys_prior_cfg["gamma_sd"]),
    prior_eta_log_mean=float(phys_prior_cfg["eta_log_mean"]),
    prior_eta_log_sd=float(phys_prior_cfg["eta_log_sd"]),
    prior_delta_mean=float(phys_prior_cfg["delta_mean"]),
    prior_delta_sd=float(phys_prior_cfg["delta_sd"]),
    detector_mode=det_cfg["mode"],
    livetime_years=float(det_cfg["livetime_years"]),
    sigma_log10=float(det_cfg["sigma_log10"]),
    physical_flux_units=bool(det_cfg.get("physical_flux_units", False)),
    hese75_aeff_allsky_path=det_cfg.get("hese75_aeff_allsky_path", None),
    hese75_migration_path=det_cfg.get("hese75_migration_path", None),
    hese75_sky_factor_sr=float(det_cfg.get("hese75_sky_factor_sr", 4.0 * np.pi)),
    rng=rng,
)
K = len(hds.seasons)
B = len(hds.seasons[0].counts)

log_E_norm = np.log(hds.seasons[0].E_centers / E_REF)
log_E_mean = float(np.mean(log_E_norm))

# building K x B counts array
counts_array = np.array([hds.seasons[k].counts for k in range(K)])

# build M_eff matching the Python posterior: M_eff = smearing * (acceptance * exposure)
mo = hds.seasons[0].model_options
acceptance = mo["detector_acceptance"]
smearing_matrix = mo["detector_smearing"]
_livetime = mo["livetime_years"]
_physical = mo["physical_flux_units"]
exposure = _livetime * (SECONDS_PER_YEAR if _physical else 1.0)
M_eff = smearing_matrix * (acceptance * exposure)

# prior hyperparameters from config (mu_eta_at_mean shifts by -delta*log_E_mean vs mu_eta)
_delta_prior_mean = float(phys_prior_cfg["delta_mean"])
mu_eta_at_mean_prior_mean = float(phys_prior_cfg["eta_log_mean"]) - _delta_prior_mean * log_E_mean

stan_data = {
    "K": K,
    "B": B,
    "counts": counts_array.tolist(),
    "log_E_norm": log_E_norm.tolist(),
    "E_widths": hds.seasons[0].E_widths.tolist(),
    "log_E_mean": log_E_mean,
    "M_eff": M_eff.tolist(),
    "prior_mu_phi_mean": float(phys_prior_cfg["phi_log_mean"]),
    "prior_mu_phi_sd": float(phys_prior_cfg["phi_log_sd"]),
    "prior_mu_eta_at_mean_mean": mu_eta_at_mean_prior_mean,
    "prior_mu_eta_at_mean_sd": float(phys_prior_cfg["eta_log_sd"]),
    "prior_gamma_mean": float(phys_prior_cfg["gamma_mean"]),
    "prior_gamma_sd": float(phys_prior_cfg["gamma_sd"]),
    "prior_delta_mean": _delta_prior_mean,
    "prior_delta_sd": float(phys_prior_cfg["delta_sd"]),
}

model = cmdstanpy.CmdStanModel(
    stan_file="hierarchical_model.stan",
    force_compile=True,
)

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=5000,
    iter_sampling=10000,
    seed=221,
    adapt_delta=0.95,
    show_progress=True,
)

pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 120)

summary = fit.summary()
main_params = (
    ["lp__", "mu_phi", "sigma_phi", "mu_eta_at_mean", "sigma_eta", "gamma", "delta"]
    + [f"log_phi_k[{k+1}]" for k in range(K)]
    + [f"log_eta_at_mean_k[{k+1}]" for k in range(K)]
)
print("MAIN PARAMETERS")
print(summary.loc[main_params].to_string())

print(f"\nDivergences per chain:")
for i, chain in enumerate(fit.method_variables()['divergent__'].T):
    print(f"  Chain {i+1}: {int(chain.sum())}")

# Stan's mu_eta_at_mean is the population mean of log_eta evaluated at E_mean,
# whereas the Python model's mu_eta is at E_ref. True value shifts by -delta*log_E_mean.
true_delta = hds.true_params["delta"]
true_mu_eta_at_mean = hds.true_params["mu_eta"] - true_delta * log_E_mean

print("\nSHARED PARAMETER COMPARISON")
print(f"{'Param':<18} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 74)

shared = {
    "gamma":          (hds.true_params["gamma"],    fit.stan_variable("gamma")),
    "delta":          (hds.true_params["delta"],    fit.stan_variable("delta")),
    "mu_phi":         (hds.true_params["mu_phi"],   fit.stan_variable("mu_phi")),
    "sigma_phi":      (hds.true_params["sigma_phi"],fit.stan_variable("sigma_phi")),
    "mu_eta_at_mean": (true_mu_eta_at_mean,          fit.stan_variable("mu_eta_at_mean")),
    "sigma_eta":      (hds.true_params["sigma_eta"], fit.stan_variable("sigma_eta")),
}

for name, (true_val, samples) in shared.items():
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{name:<18} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")

print("\nSEASON-SPECIFIC log_phi_k COMPARISON")
print(f"{'Season':<8} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 65)

log_phi_k_samples = fit.stan_variable("log_phi_k")
for k in range(K):
    true_val = np.log(hds.phi_k[k])
    samples  = log_phi_k_samples[:, k]
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{k+1:<8} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")

print("\nSEASON-SPECIFIC log_eta_at_mean_k COMPARISON")
print(f"{'Season':<8} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 65)

# true log_eta_at_mean_k[k] = log(eta_k) - delta * log_E_mean (same energy-reference shift)
log_eta_at_mean_k_samples = fit.stan_variable("log_eta_at_mean_k")
for k in range(K):
    true_val = np.log(hds.eta_k[k]) - true_delta * log_E_mean
    samples  = log_eta_at_mean_k_samples[:, k]
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{k+1:<8} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")
