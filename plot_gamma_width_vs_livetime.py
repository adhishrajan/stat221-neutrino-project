"""
Scan livetime for SPL-only inference and plot normalized posterior width of gamma.

Normalized width:
    sigma_gamma / |gamma_true|
"""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from load_config import load_config
from simulator import generate_single_season
from posteriors import log_posterior_base, grad_log_posterior_base
from samplers import run_mala, run_multiple_chains, _combined_samples

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    }
)


# 1-100 years, logarithmically spaced.
LIVETIMES_YEARS = np.logspace(0.0, 2.0, 12)
N_TRIALS = 6
N_CHAINS_OVERRIDE = None
OUTPUT_PATH = "results/gamma_width_vs_livetime_spl.png"
RESULTS_TABLE_PATH = "results/gamma_width_vs_livetime_spl.csv"
PLOT_ONLY = True

cfg = copy.deepcopy(load_config())

if N_CHAINS_OVERRIDE is not None:
    cfg["testing"]["n_chains"] = int(N_CHAINS_OVERRIDE)

signal_cfg = cfg["signal"]
bg_cfg = cfg["background"]
bins_cfg = cfg["energy_bins"]
atmo_cfg = cfg["atmospheric_realism"]
det_cfg = cfg["detector_response"]
prior_phys_cfg = cfg["priors"]["physics"]
prior_type = cfg["priors"]["default"]
test_cfg = cfg["testing"]
mala_cfg = cfg["samplers"]["MALA"]

gamma_true = float(signal_cfg["gamma"])
eta_prompt_default = float(
    bg_cfg.get("eta_prompt", float(bg_cfg["eta"]) * float(atmo_cfg.get("prompt_fraction", 0.0)))
)

rng_master = np.random.default_rng(int(test_cfg["seed_samplers"]))
rows = []
if PLOT_ONLY and Path(RESULTS_TABLE_PATH).exists():
    table = np.loadtxt(RESULTS_TABLE_PATH, delimiter=",", skiprows=1)
    if table.ndim == 1:
        table = table[None, :]
    for row in table:
        rows.append(
            {
                "livetime_years": float(row[0]),
                "norm_width_mean": float(row[1]),
                "norm_width_q16": float(row[2]),
                "norm_width_q84": float(row[3]),
                "gamma_mean_over_trials": float(row[4]),
            }
        )
else:
    livetimes = [float(x) for x in LIVETIMES_YEARS]
    for t_years in livetimes:
        widths = []
        means = []
        print(str(t_years) + ":")
        for _ in range(int(N_TRIALS)):
            print(f"  Trial {_ + 1}/{N_TRIALS}...")
            trial_rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))

            ds = generate_single_season(
                phi=float(signal_cfg["phi"]),
                gamma=float(signal_cfg["gamma"]),
                eta=float(bg_cfg["eta"]),
                delta=float(bg_cfg["delta"]),
                truth_model="power_law",
                n_bins=int(bins_cfg["n_bins"]),
                E_min=float(bins_cfg["E_min"]),
                E_max=float(bins_cfg["E_max"]),
                gamma2=float(signal_cfg["gamma2"]),
                E_break=float(signal_cfg["E_break"]),
                E_cut=float(signal_cfg["E_cut"]),
                atmo_model=atmo_cfg["model"],
                prompt_fraction=float(atmo_cfg["prompt_fraction"]),
                eta_prompt=eta_prompt_default,
                delta_prompt=float(atmo_cfg["delta_prompt"]),
                E_knee=float(atmo_cfg["E_knee"]),
                knee_sharpness=float(atmo_cfg["knee_sharpness"]),
                prompt_prior_log_mean=float(atmo_cfg.get("prompt_prior_log_mean", np.log(1e-6))),
                prompt_prior_log_sd=float(atmo_cfg.get("prompt_prior_log_sd", 1.0)),
                prior_phi_log_mean=float(prior_phys_cfg["phi_log_mean"]),
                prior_phi_log_sd=float(prior_phys_cfg["phi_log_sd"]),
                prior_gamma_mean=float(prior_phys_cfg["gamma_mean"]),
                prior_gamma_sd=float(prior_phys_cfg["gamma_sd"]),
                prior_eta_log_mean=float(prior_phys_cfg["eta_log_mean"]),
                prior_eta_log_sd=float(prior_phys_cfg["eta_log_sd"]),
                prior_delta_mean=float(prior_phys_cfg["delta_mean"]),
                prior_delta_sd=float(prior_phys_cfg["delta_sd"]),
                detector_mode=det_cfg["mode"],
                livetime_years=float(t_years),
                sigma_log10=float(det_cfg["sigma_log10"]),
                physical_flux_units=bool(det_cfg.get("physical_flux_units", False)),
                hese75_aeff_allsky_path=det_cfg.get("hese75_aeff_allsky_path", None),
                hese75_migration_path=det_cfg.get("hese75_migration_path", None),
                hese75_sky_factor_sr=float(det_cfg.get("hese75_sky_factor_sr", 4.0 * np.pi)),
                rng=trial_rng,
            )

            if ds.model_options.get("infer_prompt_eta", False):
                param_names = ["log_phi", "gamma", "log_eta", "delta", "log_eta_prompt"]
                theta_true = np.array(
                    [
                        np.log(ds.true_params["phi"]),
                        ds.true_params["gamma"],
                        np.log(ds.true_params["eta"]),
                        ds.true_params["delta"],
                        np.log(ds.true_params["eta_prompt"]),
                    ],
                    dtype=float,
                )
            else:
                param_names = ["log_phi", "gamma", "log_eta", "delta"]
                theta_true = np.array(
                    [
                        np.log(ds.true_params["phi"]),
                        ds.true_params["gamma"],
                        np.log(ds.true_params["eta"]),
                        ds.true_params["delta"],
                    ],
                    dtype=float,
                )

            chain_results = run_multiple_chains(
                run_mala,
                theta_init=theta_true,
                n_chains=int(test_cfg["n_chains"]),
                init_strategy=test_cfg["init_strategy"],
                init_scale=float(test_cfg["init_scale_base"]),
                rng=trial_rng,
                log_posterior_fn=lambda theta, _ds=ds: log_posterior_base(
                    theta,
                    _ds.counts,
                    _ds.E_centers,
                    _ds.E_widths,
                    prior_type,
                    model_options=_ds.model_options,
                ),
                grad_log_posterior_fn=lambda theta, _ds=ds: grad_log_posterior_base(
                    theta,
                    _ds.counts,
                    _ds.E_centers,
                    _ds.E_widths,
                    prior_type,
                    model_options=_ds.model_options,
                ),
                n_iterations=int(mala_cfg["n_iterations"]),
                n_burnin=int(mala_cfg["n_burnin"]),
                step_size=float(mala_cfg["step_size"]),
                adapt_step=bool(mala_cfg["adapt_step"]),
                adapt_until=int(mala_cfg["adapt_until"]),
                adapt_interval=int(mala_cfg["adapt_interval"]),
                target_accept=float(mala_cfg["target_accept"]),
                param_names=param_names,
                precond_type=mala_cfg["precond_type"],
                normalize_precond=bool(mala_cfg["normalize_precond"]),
                adapt_precond=bool(mala_cfg["adapt_precond"]),
                verbose=False,
            )

            combined = _combined_samples(chain_results)
            gamma_idx = param_names.index("gamma")
            gamma_samples = combined[:, gamma_idx]

            widths.append(float(np.std(gamma_samples, ddof=1)) / abs(gamma_true))
            means.append(float(np.mean(gamma_samples)))

        widths = np.asarray(widths, dtype=float)
        rows.append(
            {
                "livetime_years": t_years,
                "norm_width_mean": float(np.mean(widths)),
                "norm_width_q16": float(np.quantile(widths, 0.16)),
                "norm_width_q84": float(np.quantile(widths, 0.84)),
                "gamma_mean_over_trials": float(np.mean(means)),
            }
        )

    csv_out = Path(RESULTS_TABLE_PATH)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.array(
        [
            [
                r["livetime_years"],
                r["norm_width_mean"],
                r["norm_width_q16"],
                r["norm_width_q84"],
                r["gamma_mean_over_trials"],
            ]
            for r in rows
        ],
        dtype=float,
    )
    np.savetxt(
        csv_out,
        arr,
        delimiter=",",
        header="livetime_years,norm_width_mean,norm_width_q16,norm_width_q84,gamma_mean_over_trials",
        comments="",
    )
    print("Saved table:", csv_out)

rows = sorted(rows, key=lambda r: r["livetime_years"])
t = np.array([r["livetime_years"] for r in rows])
y = np.array([r["norm_width_mean"] for r in rows])
y16 = np.array([r["norm_width_q16"] for r in rows])
y84 = np.array([r["norm_width_q84"] for r in rows])

plt.figure(figsize=(9, 6))
plt.plot(t, y, marker="o", lw=2.2, color="black")
plt.fill_between(t, y16, y84, alpha=0.2, color="0.5")

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Livetime [years]")
plt.ylabel(r"Normalized posterior width of $\gamma$")
plt.grid(True, which="both", alpha=0.2, linewidth=0.7)
plt.tight_layout()

out = Path(OUTPUT_PATH)
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")

print("Saved figure:", out)
print("\nTable:")
print("livetime_years  norm_width_mean  q16  q84")
for r in rows:
    print(
        f"{r['livetime_years']:13.3f}  "
        f"{r['norm_width_mean']:15.5e}  "
        f"{r['norm_width_q16']:5.5e}  "
        f"{r['norm_width_q84']:5.5e}"
    )
