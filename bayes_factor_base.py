"""
Computes pairwise Bayes Factors across prior types (weakly / flat / lognormal_gamma)
for each truth model (power_law, broken_power_law, cutoff), using the base SPL
inference model.

Generates:
    - Stats/log_ml_base.csv: log Z and SE for all combinations of (truth_model, prior_type)
    - Stats/bayes_factors_base.csv  
    - Figures/bayes_factors_base.png 
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from itertools import product
from scipy.special import gammaln

from simulator import generate_single_season
from posteriors import log_posterior_base, grad_log_posterior_base
from samplers import (
    run_rwmh, run_mala,
    run_multiple_chains, estimate_dense_precond_from_rwmh,
)
from bridge_sampling import estimate_log_ml
from load_config import load_config


os.makedirs("Stats", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

config         = load_config()
signal_cfg     = config["signal"]
background_cfg = config["background"]
bins_cfg       = config["energy_bins"]
atmo_cfg       = config["atmospheric_realism"]
det_cfg        = config["detector_response"]
phys_prior_cfg = config["priors"]["physics"]
rwmh_cfg       = config["samplers"]["RWMH"]
mala_cfg       = config["samplers"]["MALA"]
test_cfg       = config["testing"]

TRUTH_MODELS = ["power_law", "broken_power_law", "cutoff"]
PRIOR_TYPES  = ["weakly", "flat", "lognormal_gamma"]

TRUTH_LABELS = {
    "power_law":       "SPL",
    "broken_power_law":"BPL",
    "cutoff":          "Cutoff",
}
PRIOR_LABELS = {
    "weakly":          "Weakly\nInformative",
    "flat":            "Flat",
    "lognormal_gamma": "Log-Normal γ",
}
IMPROPER_PRIORS = {"flat", "lognormal_gamma"}

# Helper Functions
def _generate_data(truth_model):
    rng = np.random.default_rng(int(test_cfg["seed_samplers"]))
    return generate_single_season(
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


def _get_samples(ds, truth_model, prior_type):
    """Load cached samples or run RWMH+MALA to get posterior samples."""
    samples_path = f"Stats/samples_base_{truth_model}_{prior_type}_mala.npy"
    if os.path.exists(samples_path):
        print(f"    Loading cached samples: {samples_path}")
        return np.load(samples_path)

    infer_prompt = bool(ds.model_options.get("infer_prompt_eta", False))
    if infer_prompt:
        param_names = ["log_phi", "gamma", "log_eta", "delta", "log_eta_prompt"]
        theta_true = np.array([
            np.log(ds.true_params["phi"]), ds.true_params["gamma"],
            np.log(ds.true_params["eta"]), ds.true_params["delta"],
            np.log(ds.true_params["eta_prompt"]),
        ])
    else:
        param_names = ["log_phi", "gamma", "log_eta", "delta"]
        theta_true = np.array([
            np.log(ds.true_params["phi"]), ds.true_params["gamma"],
            np.log(ds.true_params["eta"]), ds.true_params["delta"],
        ])

    def log_post(theta, _ds=ds, _pt=prior_type):
        return log_posterior_base(theta, _ds.counts, _ds.E_centers, _ds.E_widths,
                                  _pt, model_options=_ds.model_options)

    def grad_log_post(theta, _ds=ds, _pt=prior_type):
        return grad_log_posterior_base(theta, _ds.counts, _ds.E_centers, _ds.E_widths,
                                       _pt, model_options=_ds.model_options)

    rng = np.random.default_rng(int(test_cfg["seed_samplers"]) + 1)

    print("    Running RWMH...")
    rwmh_results = run_multiple_chains(
        run_rwmh, theta_init=theta_true,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_base"]),
        rng=rng, log_posterior_fn=log_post,
        n_iterations=int(rwmh_cfg["n_iterations"]),
        n_burnin=int(rwmh_cfg["n_burnin"]),
        adapt_proposal=bool(rwmh_cfg["adapt_proposal"]),
        adapt_until=int(rwmh_cfg["adapt_until"]),
        adapt_interval=int(rwmh_cfg["adapt_interval"]),
        target_accept=float(rwmh_cfg["target_accept"]),
        param_names=param_names,
    )

    print("    Running MALA...")
    rwmh_cov = estimate_dense_precond_from_rwmh(
        rwmh_results, ridge=float(test_cfg["preconditioner_ridge"])
    )
    mala_results = run_multiple_chains(
        run_mala, theta_init=theta_true,
        n_chains=int(test_cfg["n_chains"]),
        init_strategy=test_cfg["init_strategy"],
        init_scale=float(test_cfg["init_scale_base"]),
        rng=rng, log_posterior_fn=log_post,
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

    samples = np.vstack([r.samples for r in mala_results])
    np.save(samples_path, samples)
    print(f"    Saved {samples_path}")
    return samples


# Main

log_ml_cache_path = "Stats/log_ml_base_cache.json"
if os.path.exists(log_ml_cache_path):
    with open(log_ml_cache_path) as f:
        log_ml_cache = json.load(f)
else:
    log_ml_cache = {}

for truth_model in TRUTH_MODELS:
    print(f"\n{'='*60}")
    print(f"  Truth model: {truth_model}")
    print(f"{'='*60}")

    ds = _generate_data(truth_model)

    for prior_type in PRIOR_TYPES:
        key = f"{truth_model}_{prior_type}"
        if key in log_ml_cache:
            print(f"  [{prior_type}] log Z = {log_ml_cache[key]['log_Z']:.3f}  (cached)")
            continue

        print(f"  [{prior_type}]")

        samples = _get_samples(ds, truth_model, prior_type)

        def log_post_fn(theta, _ds=ds, _pt=prior_type):
            return log_posterior_base(theta, _ds.counts, _ds.E_centers, _ds.E_widths,
                                      _pt, model_options=_ds.model_options)

        print(f"    Running bridge sampling ({len(samples)} posterior samples)...")
        rng_bs = np.random.default_rng(int(test_cfg["seed_samplers"]) + 42)
        log_Z, log_Z_se = estimate_log_ml(
            log_post_fn,
            samples,
            n_proposal=5000,
            n_iter=500,
            tol=1e-6,
            rng=rng_bs,
            n_bootstrap=200,
            ridge=1e-6,
        )

        # Add back the Σ log(n_i!) term dropped by the Poisson log-likelihood,
        # so that log Z values are comparable across different datasets (truth models).
        log_Z += float(np.sum(gammaln(ds.counts + 1)))
        # SE is unaffected (additive constant has no variance)

        log_ml_cache[key] = {"log_Z": log_Z, "log_Z_se": log_Z_se}
        with open(log_ml_cache_path, "w") as f:
            json.dump(log_ml_cache, f, indent=2)
        print(f"    log Z = {log_Z:.3f}  ± {log_Z_se:.3f}")

# Build output table

rows = []
for truth_model in TRUTH_MODELS:
    for prior_type in PRIOR_TYPES:
        key = f"{truth_model}_{prior_type}"
        entry = log_ml_cache[key]
        rows.append({
            "truth_model": truth_model,
            "prior_type":  prior_type,
            "log_Z":       entry["log_Z"],
            "log_Z_se":    entry["log_Z_se"],
        })

results = pd.DataFrame(rows)
results.to_csv("Stats/log_ml_base.csv", index=False)
print("\nSaved Stats/log_ml_base.csv")

# Build pairwise comparison table
bf_rows = []
for truth_model in TRUTH_MODELS:
    sub = results[results["truth_model"] == truth_model].set_index("prior_type")
    for p1 in PRIOR_TYPES:
        for p2 in PRIOR_TYPES:
            if p1 == p2:
                continue
            log_BF = sub.loc[p1, "log_Z"] - sub.loc[p2, "log_Z"]
            bf_rows.append({
                "Truth Model": TRUTH_LABELS[truth_model],
                "Prior (M1)":  p1,
                "Prior (M2)":  p2,
                "log BF(M1/M2)": round(log_BF, 3),
                "BF(M1/M2)":     round(np.exp(log_BF), 3),
            })

bf_df = pd.DataFrame(bf_rows)
bf_df.to_csv("Stats/bayes_factors_base.csv", index=False)
print("Saved Stats/bayes_factors_base.csv")

# Figure generation
PRIOR_DISPLAY = {
    "weakly":          "Weakly",
    "flat":            "Flat",
    "lognormal_gamma": "LN-γ",
}

# Figure 1: merged log Z table (rows = priors, cols = truth models)
ALT   = to_rgba("#f8f9fa")
WHITE = to_rgba("white")
GREEN = to_rgba("#d4edda")

col_labels = ["Prior"] + [TRUTH_LABELS[tm] for tm in TRUTH_MODELS]
n_cols = len(col_labels)
n_rows = len(PRIOR_TYPES)

# best prior per truth model (for green highlighting)
best_prior_per_tm = {
    tm: results[results["truth_model"] == tm].set_index("prior_type")["log_Z"].idxmax()
    for tm in TRUTH_MODELS
}

cell_text = []
cell_colors = []
for row_idx, prior_type in enumerate(PRIOR_TYPES):
    base = ALT if row_idx % 2 == 0 else WHITE
    row_vals = [PRIOR_DISPLAY[prior_type]]
    row_cols = [base]
    for tm in TRUTH_MODELS:
        log_Z_val = results.loc[
            (results["truth_model"] == tm) & (results["prior_type"] == prior_type),
            "log_Z"
        ].iloc[0]
        row_vals.append(f"{log_Z_val:.2f}")
        row_cols.append(GREEN if best_prior_per_tm[tm] == prior_type else base)
    cell_text.append(row_vals)
    cell_colors.append(row_cols)

fig, ax = plt.subplots(figsize=(7, 2.8))
ax.axis("off")

col_widths = [0.22] + [0.19] * len(TRUTH_MODELS)
tbl = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellColours=cell_colors,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.8)

for j in range(n_cols):
    tbl[0, j].set_facecolor("#1a1a2e")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(n_rows + 1):
    for j in range(n_cols):
        tbl[i, j].set_edgecolor("#dee2e6")

fig.suptitle(
    "Log Marginal Likelihoods (SPL Model) by Prior and Truth Model",
    fontsize=9, fontweight="bold", y=1.04,
)

plt.tight_layout()
fig.savefig("Figures/bayes_factors_base.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved Figures/bayes_factors_base.png")

# Figure 2: pairwise log Bayes Factor heatmaps
prior_short = [PRIOR_DISPLAY[p] for p in PRIOR_TYPES]
n_priors = len(PRIOR_TYPES)

fig2, axes2 = plt.subplots(1, len(TRUTH_MODELS), figsize=(14, 4))
if len(TRUTH_MODELS) == 1:
    axes2 = [axes2]

for ax, truth_model in zip(axes2, TRUTH_MODELS):
    sub = (
        results[results["truth_model"] == truth_model]
        .set_index("prior_type")
        .loc[PRIOR_TYPES]
    )

    log_bf = np.array([
        [sub.loc[p1, "log_Z"] - sub.loc[p2, "log_Z"] for p2 in PRIOR_TYPES]
        for p1 in PRIOR_TYPES
    ])

    vmax = max(np.abs(log_bf[~np.eye(n_priors, dtype=bool)]).max(), 0.1)
    im = ax.imshow(log_bf, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n_priors))
    ax.set_yticks(range(n_priors))
    ax.set_xticklabels(prior_short, fontsize=9)
    ax.set_yticklabels(prior_short, fontsize=9)
    ax.set_xlabel("Model 2", fontsize=9)
    ax.set_ylabel("Model 1", fontsize=9)
    ax.set_title(f"Truth: {TRUTH_LABELS[truth_model]}", fontsize=10, fontweight="bold")

    for i in range(n_priors):
        for j in range(n_priors):
            ax.text(j, i, f"{log_bf[i, j]:.2f}",
                    ha="center", va="center", fontsize=9,
                    color="black" if abs(log_bf[i, j]) < 0.6 * vmax else "white")

    plt.colorbar(im, ax=ax, label="log BF(Model 1/Model 2)", shrink=0.85)

fig2.suptitle(
    "Pairwise Log Bayes Factors (SPL Model)",
    fontsize=9, fontweight="bold",
)

plt.tight_layout()
fig2.savefig("Figures/bayes_factors_comparisons.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved Figures/bayes_factors_comparisons.png")
