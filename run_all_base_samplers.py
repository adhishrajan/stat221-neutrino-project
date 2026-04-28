"""
run_all_base_samplers.py

Runs RWMH and MALA for every combination of:
  - truth_model: power_law, broken_power_law, cutoff
  - prior_type:  weakly, flat, lognormal_gamma

Saves posterior summary CSVs to:
  Stats/posterior_summary_base_{truth_model}_{prior_type}_{sampler}.csv

And diagnostics to:
  Stats/diagnostics_{table}_{truth_model}_{prior_type}.csv
"""

import os
import numpy as np
import pandas as pd
from itertools import product

from simulator import generate_single_season
from posteriors import log_posterior_base, grad_log_posterior_base
from samplers import (
    run_rwmh, run_mala,
    run_multiple_chains, estimate_dense_precond_from_rwmh,
    posterior_summary_multi, diagnostics_dataframe,
)
from load_config import load_config

os.makedirs("Stats", exist_ok=True)

config        = load_config()
signal_cfg    = config["signal"]
background_cfg= config["background"]
bins_cfg      = config["energy_bins"]
atmo_cfg      = config["atmospheric_realism"]
det_cfg       = config["detector_response"]
phys_prior_cfg= config["priors"]["physics"]
rwmh_cfg      = config["samplers"]["RWMH"]
mala_cfg      = config["samplers"]["MALA"]
test_cfg      = config["testing"]

TRUTH_MODELS  = ["power_law", "broken_power_law", "cutoff"]
PRIOR_TYPES   = ["weakly", "flat", "lognormal_gamma"]

# ── Main loop ─────────────────────────────────────────────────────────────────
for truth_model, prior_type in product(TRUTH_MODELS, PRIOR_TYPES):
    print(f"\n{'='*60}")
    print(f"  truth_model={truth_model}  prior_type={prior_type}")
    print(f"{'='*60}")

    # Check if already done — skip if both CSVs exist
    path_rwmh = f"Stats/posterior_summary_base_{truth_model}_{prior_type}_rwmh.csv"
    path_mala = f"Stats/posterior_summary_base_{truth_model}_{prior_type}_mala.csv"
    if os.path.exists(path_rwmh) and os.path.exists(path_mala):
        print("  Already exists, skipping.")
        continue

    # ── Generate data ──────────────────────────────────────────────────────
    rng = np.random.default_rng(int(test_cfg["seed_samplers"]))

    ds = generate_single_season(
        phi=float(signal_cfg["phi"]),
        gamma=float(signal_cfg["gamma"]),
        eta=float(background_cfg["eta"]),
        delta=float(background_cfg["delta"]),
        truth_model=truth_model,
        n_bins=int(bins_cfg["n_bins"]),
        E_min=float(bins_cfg["E_min"]),
        E_max=float(bins_cfg["E_max"]),
        gamma2=float(signal_cfg["gamma2"]),
        E_break=float(signal_cfg["E_break"]),
        E_cut=float(signal_cfg["E_cut"]),
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(background_cfg.get(
            "eta_prompt",
            float(background_cfg["eta"]) * float(atmo_cfg["prompt_fraction"])
        )),
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

    print(f"  Counts: {ds.counts.sum()}  "
          f"Signal: {ds.mu_signal.sum():.1f}  "
          f"Background: {ds.mu_background.sum():.1f}")

    infer_prompt_eta = bool(ds.model_options.get("infer_prompt_eta", False))
    if infer_prompt_eta:
        param_names = ["log_phi", "gamma", "log_eta", "delta", "log_eta_prompt"]
        theta_true = np.array([
            np.log(ds.true_params["phi"]),
            ds.true_params["gamma"],
            np.log(ds.true_params["eta"]),
            ds.true_params["delta"],
            np.log(ds.true_params["eta_prompt"]),
        ])
    else:
        param_names = ["log_phi", "gamma", "log_eta", "delta"]
        theta_true = np.array([
            np.log(ds.true_params["phi"]),
            ds.true_params["gamma"],
            np.log(ds.true_params["eta"]),
            ds.true_params["delta"],
        ])

    true_values = {
        "phi":   ds.true_params["phi"],
        "gamma": ds.true_params["gamma"],
        "eta":   ds.true_params["eta"],
        "delta": ds.true_params["delta"],
    }
    if infer_prompt_eta:
        true_values["eta_prompt"] = ds.true_params["eta_prompt"]

    # ── Closure over current ds and prior_type ────────────────────────────
    def log_post(theta, _ds=ds, _pt=prior_type):
        return log_posterior_base(
            theta, _ds.counts, _ds.E_centers, _ds.E_widths,
            _pt, model_options=_ds.model_options,
        )

    def grad_log_post(theta, _ds=ds, _pt=prior_type):
        return grad_log_posterior_base(
            theta, _ds.counts, _ds.E_centers, _ds.E_widths,
            _pt, model_options=_ds.model_options,
        )

    theta_init = theta_true.copy()

    # ── RWMH ──────────────────────────────────────────────────────────────
    print("  Running RWMH...")
    rwmh_results = run_multiple_chains(
        run_rwmh,
        theta_init=theta_init,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_base"]),
        rng=rng,
        log_posterior_fn=log_post,
        n_iterations=int(rwmh_cfg["n_iterations"]),
        n_burnin=int(rwmh_cfg["n_burnin"]),
        adapt_proposal=bool(rwmh_cfg["adapt_proposal"]),
        adapt_until=int(rwmh_cfg["adapt_until"]),
        adapt_interval=int(rwmh_cfg["adapt_interval"]),
        target_accept=float(rwmh_cfg["target_accept"]),
        param_names=param_names,
    )

    # ── MALA (preconditioned by RWMH posterior covariance) ────────────────
    print("  Running MALA...")
    rwmh_cov = estimate_dense_precond_from_rwmh(
        rwmh_results, ridge=float(test_cfg["preconditioner_ridge"])
    )
    mala_results = run_multiple_chains(
        run_mala,
        theta_init=theta_init,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_base"]),
        rng=rng,
        log_posterior_fn=log_post,
        grad_log_posterior_fn=grad_log_post,
        n_iterations=int(mala_cfg["n_iterations"]),
        n_burnin=int(mala_cfg["n_burnin"]),
        step_size=float(mala_cfg["step_size"]),
        adapt_step=bool(mala_cfg["adapt_step"]),
        adapt_until=int(mala_cfg["adapt_until"]),
        adapt_interval=int(mala_cfg["adapt_interval"]),
        target_accept=float(mala_cfg["target_accept"]),
        param_names=param_names,
        precond=rwmh_cov,
        adapt_precond=bool(mala_cfg["adapt_precond"]),
        precond_type=mala_cfg["precond_type"],
        normalize_precond=bool(mala_cfg["normalize_precond"]),
    )

    # ── Save posterior summaries ──────────────────────────────────────────
    for sampler_name, results in [("rwmh", rwmh_results), ("mala", mala_results)]:
        summary = posterior_summary_multi(results, true_values=true_values)
        df = pd.DataFrame(summary).T
        path = f"Stats/posterior_summary_base_{truth_model}_{prior_type}_{sampler_name}.csv"
        df.to_csv(path)
        print(f"  Saved {path}")

    # ── Save diagnostics ──────────────────────────────────────────────────
    diag = diagnostics_dataframe({"RWMH": rwmh_results, "MALA": mala_results})
    for table_name, df in diag.items():
        path = f"Stats/diagnostics_{table_name}_{truth_model}_{prior_type}.csv"
        df.to_csv(path)
        print(f"  Saved {path}")

print("\nAll done. Generated CSVs:")
for truth_model, prior_type in product(TRUTH_MODELS, PRIOR_TYPES):
    for sampler in ["rwmh", "mala"]:
        p = f"Stats/posterior_summary_base_{truth_model}_{prior_type}_{sampler}.csv"
        status = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{status}] {p}")