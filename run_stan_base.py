import numpy as np
import cmdstanpy
import pandas as pd
from simulator import generate_single_season, E_REF, SECONDS_PER_YEAR
from load_config import load_config

config = load_config()
test_cfg = config["testing"]
signal_cfg = config["signal"]
background_cfg = config["background"]
bins_cfg = config["energy_bins"]
atmo_cfg = config["atmospheric_realism"]
det_cfg = config["detector_response"]
phys_prior_cfg = config["priors"]["physics"]

rng = np.random.default_rng(int(test_cfg["seed_samplers"]))

ds = generate_single_season(
    phi=float(signal_cfg["phi"]),
    gamma=float(signal_cfg["gamma"]),
    eta=float(background_cfg["eta"]),
    delta=float(background_cfg["delta"]),
    truth_model=test_cfg["truth_model"],
    n_bins=int(bins_cfg["n_bins"]),
    E_min=float(bins_cfg["E_min"]),
    E_max=float(bins_cfg["E_max"]),
    gamma2=float(signal_cfg["gamma2"]),
    E_break=float(signal_cfg["E_break"]),
    E_cut=float(signal_cfg["E_cut"]),
    atmo_model=atmo_cfg["model"],
    prompt_fraction=float(atmo_cfg["prompt_fraction"]),
    eta_prompt=float(background_cfg.get("eta_prompt", float(background_cfg["eta"]) * float(atmo_cfg["prompt_fraction"]))),
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

log_E_norm = np.log(ds.E_centers / E_REF)
log_E_mean = float(np.mean(log_E_norm))

# build M_eff matching the Python posterior: M_eff = smearing * (acceptance * exposure)
mo = ds.model_options
acceptance = mo["detector_acceptance"]
smearing_matrix = mo["detector_smearing"]
_livetime = mo["livetime_years"]
_physical = mo["physical_flux_units"]
exposure = _livetime * (SECONDS_PER_YEAR if _physical else 1.0)
M_eff = smearing_matrix * (acceptance * exposure)

# prior hyperparameters from config (log_eta_at_mean shifts by -delta*log_E_mean vs log_eta)
_delta_prior_mean = float(phys_prior_cfg["delta_mean"])
log_eta_at_mean_prior_mean = float(phys_prior_cfg["eta_log_mean"]) - _delta_prior_mean * log_E_mean

stan_data = {
    "B": len(ds.counts),
    "counts": ds.counts.tolist(),
    "log_E_norm": log_E_norm.tolist(),
    "E_widths": ds.E_widths.tolist(),
    "log_E_mean": log_E_mean,
    "M_eff": M_eff.tolist(),
    "prior_log_phi_mean": float(phys_prior_cfg["phi_log_mean"]),
    "prior_log_phi_sd": float(phys_prior_cfg["phi_log_sd"]),
    "prior_log_eta_at_mean_mean": log_eta_at_mean_prior_mean,
    "prior_log_eta_at_mean_sd": float(phys_prior_cfg["eta_log_sd"]),
    "prior_gamma_mean": float(phys_prior_cfg["gamma_mean"]),
    "prior_gamma_sd": float(phys_prior_cfg["gamma_sd"]),
    "prior_delta_mean": _delta_prior_mean,
    "prior_delta_sd": float(phys_prior_cfg["delta_sd"]),
}

model = cmdstanpy.CmdStanModel(stan_file="base_model.stan", force_compile=True)
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=5000,
    iter_sampling=10000,
    seed=int(test_cfg["seed_samplers"]),
    show_progress=True,
    adapt_delta=0.9
)

pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 120)

summary = fit.summary()
main_params = ["lp__", "log_phi", "gamma", "log_eta_at_mean", "log_eta", "delta"]
print("MAIN PARAMETERS")
print(summary.loc[main_params].to_string())

print(f"\nDivergences per chain:")
for i, chain in enumerate(fit.method_variables()['divergent__'].T):
    print(f"  Chain {i+1}: {int(chain.sum())}")

print("\nPARAMETER COMPARISON")
print(f"{'Param':<12} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 68)

params = {
    "log_phi": (np.log(ds.true_params["phi"]), fit.stan_variable("log_phi")),
    "gamma":   (ds.true_params["gamma"],        fit.stan_variable("gamma")),
    "log_eta": (np.log(ds.true_params["eta"]),  fit.stan_variable("log_eta")),
    "delta":   (ds.true_params["delta"],         fit.stan_variable("delta")),
}

for name, (true_val, samples) in params.items():
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{name:<12} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")

counts_rep = fit.stan_variable("counts_rep")
rep_mean = counts_rep.mean(axis=0)
rep_lo = np.percentile(counts_rep, 5,  axis=0)
rep_hi = np.percentile(counts_rep, 95, axis=0)

print("\nPOSTERIOR PREDICTIVE CHECK")
print(f"{'Bin':>4} {'Observed':>10} {'Rep mean':>10} {'Rep 5%':>8} {'Rep 95%':>9}")
for b in range(len(ds.counts)):
    flag = " *" if ds.counts[b] < rep_lo[b] or ds.counts[b] > rep_hi[b] else ""
    print(f"  {b+1:>2}   {ds.counts[b]:>8}   {rep_mean[b]:>9.1f}   {rep_lo[b]:>7.1f}   {rep_hi[b]:>8.1f}{flag}")
