"""
run_all_hierarchical_samplers.py

Runs RWMH and MALA for every combination of:
  - truth_model:       power_law, broken_power_law, cutoff
  - prior_type:        weakly, flat, lognormal_gamma
  - parameterization:  centered, noncentered

Saves posterior summary CSVs to:
  Stats/posterior_summary_hierarchical_{truth_model}_{prior_type}_{parameterization}_{sampler}.csv

And diagnostics to:
  Stats/diagnostics_hierarchical_{table}_{truth_model}_{prior_type}_{parameterization}.csv
"""

import os
import numpy as np
import pandas as pd
from itertools import product

from simulator import generate_hierarchical
from posteriors import make_hierarchical_fns
from samplers_hierarchical import (
    run_rwmh, run_mala,
    run_multiple_chains, estimate_dense_precond_from_rwmh,
    posterior_summary_dataframe_hierarchical, diagnostics_dataframe_hierarchical,
)
from load_config import load_config

os.makedirs("Stats", exist_ok=True)

config         = load_config()
hier_cfg       = config["hierarchical"]
atmo_cfg       = config["atmospheric_realism"]
det_cfg        = config["detector_response"]
phys_prior_cfg = config["priors"]["physics"]
rwmh_cfg       = config["samplers_hierarchical"]["RWMH"]
mala_cfg       = config["samplers_hierarchical"]["MALA"]
test_cfg       = config["testing"]

TRUTH_MODELS      = ["power_law", "broken_power_law", "cutoff"]
PRIOR_TYPES       = ["weakly", "flat", "lognormal_gamma"]
PARAMETERIZATIONS = ["centered", "noncentered"]

# ── Main loop ──────────────────────────────────────────────────────────────────
for truth_model, prior_type, parameterization in product(TRUTH_MODELS, PRIOR_TYPES, PARAMETERIZATIONS):
    print(f"\n{'='*70}")
    print(f"  truth_model={truth_model}  prior_type={prior_type}  parameterization={parameterization}")
    print(f"{'='*70}")

    path_rwmh = f"Stats/posterior_summary_hierarchical_{truth_model}_{prior_type}_{parameterization}_rwmh.csv"
    path_mala = f"Stats/posterior_summary_hierarchical_{truth_model}_{prior_type}_{parameterization}_mala.csv"
    if os.path.exists(path_rwmh) and os.path.exists(path_mala):
        print("  Already exists, skipping.")
        continue

    # ── Generate data ───────────────────────────────────────────────────────
    rng = np.random.default_rng(int(test_cfg["seed_hierarchical_samplers"]))

    ds = generate_hierarchical(
        K=int(hier_cfg["K"]),
        mu_phi=float(hier_cfg["mu_phi"]),
        sigma_phi=float(hier_cfg["sigma_phi"]),
        mu_eta=float(hier_cfg["mu_eta"]),
        sigma_eta=float(hier_cfg["sigma_eta"]),
        gamma=float(hier_cfg["gamma"]),
        delta=float(hier_cfg["delta"]),
        truth_model=truth_model,
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(config["background"].get(
            "eta_prompt",
            float(config["background"]["eta"]) * float(atmo_cfg["prompt_fraction"])
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
    K = len(ds.seasons)
    total_counts = sum(s.counts.sum() for s in ds.seasons)
    print(f"  K={K}  Total counts: {total_counts}")

    mu_phi_true    = ds.true_params["mu_phi"]
    sigma_phi_true = ds.true_params["sigma_phi"]
    mu_eta_true    = ds.true_params["mu_eta"]
    sigma_eta_true = ds.true_params["sigma_eta"]

    # ── Build theta_true, param_names, true_values ─────────────────────────
    if parameterization == "centered":
        param_names  = [f"log_phi_{i+1}" for i in range(K)]
        param_names += [f"log_eta_{i+1}" for i in range(K)]
        theta_true   = np.log(ds.phi_k).tolist() + np.log(ds.eta_k).tolist()
        latent_truth = {f"log_phi_{k+1}": np.log(ds.phi_k[k]) for k in range(K)}
        latent_truth.update({f"log_eta_{k+1}": np.log(ds.eta_k[k]) for k in range(K)})
    else:  # noncentered
        param_names  = [f"z_phi_{i+1}" for i in range(K)]
        param_names += [f"z_eta_{i+1}" for i in range(K)]
        z_phi_true   = (np.log(ds.phi_k) - mu_phi_true) / sigma_phi_true
        z_eta_true   = (np.log(ds.eta_k) - mu_eta_true) / sigma_eta_true
        theta_true   = z_phi_true.tolist() + z_eta_true.tolist()
        latent_truth = {f"z_phi_{k+1}": z_phi_true[k] for k in range(K)}
        latent_truth.update({f"z_eta_{k+1}": z_eta_true[k] for k in range(K)})

    infer_prompt_eta = bool(ds.seasons[0].model_options.get("infer_prompt_eta", False))
    param_names.extend(["gamma", "delta"])
    if infer_prompt_eta:
        param_names.append("log_eta_prompt")
    param_names.extend(["mu_phi", "log_sigma_phi", "mu_eta", "log_sigma_eta"])

    theta_true.extend([ds.true_params["gamma"], ds.true_params["delta"]])
    if infer_prompt_eta:
        theta_true.append(np.log(ds.seasons[0].true_params["eta_prompt"]))
    theta_true.extend([mu_phi_true, np.log(sigma_phi_true), mu_eta_true, np.log(sigma_eta_true)])
    theta_true = np.array(theta_true)

    true_values = {
        **latent_truth,
        **{f"log_phi_{k+1}": np.log(ds.phi_k[k]) for k in range(K)},
        **{f"log_eta_{k+1}": np.log(ds.eta_k[k]) for k in range(K)},
        "gamma":         ds.true_params["gamma"],
        "delta":         ds.true_params["delta"],
        **({"log_eta_prompt": np.log(ds.seasons[0].true_params["eta_prompt"])} if infer_prompt_eta else {}),
        "mu_phi":        mu_phi_true,
        "log_sigma_phi": np.log(sigma_phi_true),
        "mu_eta":        mu_eta_true,
        "log_sigma_eta": np.log(sigma_eta_true),
    }

    # ── Posterior functions ─────────────────────────────────────────────────
    log_post, grad_log_post, log_post_and_grad = make_hierarchical_fns(
        ds.seasons, parameterization, prior_type
    )

    # ── RWMH ────────────────────────────────────────────────────────────────
    print("  Running RWMH...")
    rwmh_results = run_multiple_chains(
        run_rwmh,
        theta_init=theta_true,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_hierarchical"]),
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

    # ── MALA (preconditioned by RWMH posterior covariance) ──────────────────
    print("  Running MALA...")
    rwmh_cov = estimate_dense_precond_from_rwmh(
        rwmh_results, ridge=float(test_cfg["preconditioner_ridge"])
    )
    mala_results = run_multiple_chains(
        run_mala,
        theta_init=theta_true,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_hierarchical"]),
        rng=rng,
        log_posterior_fn=log_post,
        grad_log_posterior_fn=grad_log_post,
        log_posterior_and_grad_fn=log_post_and_grad,
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

    # ── Save posterior summaries ────────────────────────────────────────────
    latent_disp        = test_cfg["latent_display_diagnostics"]
    results_by_sampler = {"RWMH": rwmh_results, "MALA": mala_results}

    for sampler_name, results in results_by_sampler.items():
        df = posterior_summary_dataframe_hierarchical(
            results,
            parameterization=parameterization,
            K=K,
            latent_display=latent_disp,
            true_values=true_values,
        )
        path = f"Stats/posterior_summary_hierarchical_{truth_model}_{prior_type}_{parameterization}_{sampler_name.lower()}.csv"
        df.to_csv(path)
        print(f"  Saved {path}")

    # ── Save diagnostics ────────────────────────────────────────────────────
    diag = diagnostics_dataframe_hierarchical(
        results_by_sampler,
        parameterization=parameterization,
        K=K,
        latent_display=latent_disp,
    )
    for table_name, df in diag.items():
        path = f"Stats/diagnostics_hierarchical_{table_name}_{truth_model}_{prior_type}_{parameterization}.csv"
        df.to_csv(path)
        print(f"  Saved {path}")

print("\nAll done. Generated CSVs:")
for truth_model, prior_type, parameterization in product(TRUTH_MODELS, PRIOR_TYPES, PARAMETERIZATIONS):
    for sampler in ["rwmh", "mala"]:
        p = f"Stats/posterior_summary_hierarchical_{truth_model}_{prior_type}_{parameterization}_{sampler}.csv"
        status = "OK" if os.path.exists(p) else "MISSING"
        print(f"  [{status}] {p}")
