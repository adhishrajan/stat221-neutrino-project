"""
Here we compute MLE for the base poisson model and compare
against bayesian posterior estimates for RWMH and MALA

For each truth model, we
  1. Generate the same synthetic dataset used in the Bayesian analysis
  2. Find theta_MLE by minimizing the negative log likelihood
  3. Compute approximate 90% confidence intervals with the observed Fisher information
  4. Compare MLE point estimates and CIs to Bayesian posterior means and CIs
     from the normal/lognormal prior
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
from scipy.stats import norm

from simulator import generate_single_season
from posteriors import (
    expected_counts_base,
    poisson_log_likelihood,
    unpack_base_params,
)
from load_config import load_config

os.makedirs("Stats", exist_ok=True)
os.makedirs("Figures", exist_ok=True)

config = load_config()
signal_cfg = config["signal"]
background_cfg = config["background"]
bins_cfg = config["energy_bins"]
atmo_cfg = config["atmospheric_realism"]
det_cfg = config["detector_response"]
phys_prior_cfg = config["priors"]["physics"]
test_cfg = config["testing"]

TRUTH_MODELS = ["power_law", "broken_power_law", "cutoff"]
TRUTH_LABELS = {
    "power_law": "SPL Truth",
    "broken_power_law": "BPL Truth",
    "cutoff": "Cutoff Truth",
}
TRUE_GAMMA = {
    "power_law": 2.5,
    "broken_power_law": 2.5,
    "cutoff": 2.5,
}

Z90 = norm.ppf(0.95)

COLORS = {
    "MLE": "#E67E22",
    "Bayes": "#4878CF",
}



def make_dataset(truth_model):
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


def neg_log_likelihood(theta_transformed, counts, E, E_widths, model_options):
    p = unpack_base_params(theta_transformed, model_options=model_options)
    phi = p["phi"]
    gamma = p["gamma"]
    eta = p["eta"]
    delta = p["delta"]

    if phi <= 0 or eta <= 0:
        return 1e10

    mu = expected_counts_base(E, E_widths, phi, gamma, eta, delta, model_options=model_options)
    if np.any(mu <= 0):
        return 1e10

    ll = poisson_log_likelihood(counts, mu)
    if not np.isfinite(ll):
        return 1e10

    return -ll


def compute_mle(ds):
    theta_init = np.array([
        np.log(ds.true_params["phi"]),
        ds.true_params["gamma"],
        np.log(ds.true_params["eta"]),
        ds.true_params["delta"],
    ])

    def obj(theta):
        return neg_log_likelihood(
            theta, ds.counts, ds.E_centers, ds.E_widths,
            model_options=ds.model_options
        )

    # multiple starts
    best_result = None
    best_val = np.inf
    rng_init = np.random.default_rng(42)
    starts = [theta_init.copy()]
    for _ in range(4):
        starts.append(theta_init + rng_init.normal(0, 0.1, size=4))

    for theta_start in starts:
        result = minimize(
            obj, theta_start,
            method="L-BFGS-B",
            options={"maxiter": 10000, "ftol": 1e-12, "gtol": 1e-8},
        )
        if result.fun < best_val:
            best_val = result.fun
            best_result = result

    theta_mle = best_result.x

    eps = 1e-5
    d = len(theta_mle)
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            ei = np.zeros(d); ei[i] = eps
            ej = np.zeros(d); ej[j] = eps
            fpp = obj(theta_mle + ei + ej)
            fpm = obj(theta_mle + ei - ej)
            fmp = obj(theta_mle - ei + ej)
            fmm = obj(theta_mle - ei - ej)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps**2)
            H[j, i] = H[i, j]

    H += 1e-8 * np.eye(d)

    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        print("Hessian inversion failed. using fallback SE=0.2")
        se = np.ones(d) * 0.2

    param_names = ["log_phi", "gamma", "log_eta", "delta"]
    results = {}
    for k, name in enumerate(param_names):
        mle_val = theta_mle[k]
        se_val = se[k]
        ci_lo = mle_val - Z90 * se_val
        ci_hi = mle_val + Z90 * se_val

        if name == "log_phi":
            results["phi"] = {
                "mle": np.exp(mle_val),
                "ci_lo": np.exp(ci_lo),
                "ci_hi": np.exp(ci_hi),
                "se": se_val,
            }
        elif name == "log_eta":
            results["eta"] = {
                "mle": np.exp(mle_val),
                "ci_lo": np.exp(ci_lo),
                "ci_hi": np.exp(ci_hi),
                "se": se_val,
            }
        else:
            results[name] = {
                "mle": mle_val,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
                "se": se_val,
            }

    results["_nll"]= best_val
    results["_converged"] = best_result.success
    return results


def load_bayesian(truth_model, sampler="mala", prior="weakly"):
    path = f"stats/posterior_summary_base_{truth_model}_{prior}_{sampler}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    out = {}
    param_map = {
        "log_phi": "phi",
        "gamma": "gamma",
        "log_eta": "eta",
        "delta": "delta",
    }
    for raw, display in param_map.items():
        if raw not in df.index:
            continue
        row = df.loc[raw]
        out[display] = {
            "mean": float(row["mean"]),
            "ci_lo": float(row["ci_lower"]),
            "ci_hi": float(row["ci_upper"]),
        }
    return out



mle_results = {}
bayes_results = {}

for truth_model in TRUTH_MODELS:
    print(f"\n{'='*50}")
    print(f"  {truth_model}")
    print(f"{'='*50}")
    ds = make_dataset(truth_model)
    print(f"  Counts: {ds.counts.sum()}  "
          f"Signal: {ds.mu_signal.sum():.1f}  "
          f"Background: {ds.mu_background.sum():.1f}")

    print("Computing MLE")
    mle = compute_mle(ds)
    mle_results[truth_model] = mle
    print(f"  MLE gamma  = {mle['gamma']['mle']:.4f}  "
          f"90% CI = [{mle['gamma']['ci_lo']:.4f}, {mle['gamma']['ci_hi']:.4f}]  "
          f"converged={mle['_converged']}")

    bayes = load_bayesian(truth_model)
    bayes_results[truth_model] = bayes
    if bayes:
        print(f"Bayes gamma = {bayes['gamma']['mean']:.4f}  "
              f"90% CI = [{bayes['gamma']['ci_lo']:.4f}, {bayes['gamma']['ci_hi']:.4f}]")



TRUE_VALS = {
    "phi": float(signal_cfg["phi"]),
    "gamma": 2.5,
    "eta": float(background_cfg["eta"]),
    "delta": float(background_cfg["delta"]),
}
PARAM_DISPLAY = {
    "phi": "φ",
    "gamma": "γ",
    "eta": "η",
    "delta": "δ",
}

table_rows = []
for truth_model in TRUTH_MODELS:
    mle = mle_results[truth_model]
    bayes = bayes_results.get(truth_model)

    for param in ["phi", "gamma", "eta", "delta"]:
        true_val = TRUE_VALS[param]
        mle_val = mle[param]["mle"]
        mle_lo = mle[param]["ci_lo"]
        mle_hi = mle[param]["ci_hi"]
        mle_cov = "YES" if mle_lo <= true_val <= mle_hi else "NO"
        mle_bias = mle_val - true_val

        b_mean = bayes[param]["mean"]  if bayes and param in bayes else None
        b_lo = bayes[param]["ci_lo"] if bayes and param in bayes else None
        b_hi = bayes[param]["ci_hi"] if bayes and param in bayes else None
        b_cov = ("YES" if b_lo <= true_val <= b_hi else "NO") if b_lo else "N/A"
        b_bias = (b_mean - true_val) if b_mean else None

        fmt = lambda x: f"{x:.4e}" if param in ("phi","eta") else f"{x:.4f}"

        table_rows.append({
            "Truth Model":  TRUTH_LABELS[truth_model],
            "Param": PARAM_DISPLAY[param],
            "True": fmt(true_val),
            "MLE": fmt(mle_val),
            "MLE CI Lo": fmt(mle_lo),
            "MLE CI Hi": fmt(mle_hi),
            "MLE Covered": mle_cov,
            "MLE Bias": f"{mle_bias:+.4f}",
            "Bayes Mean": fmt(b_mean) if b_mean else "N/A",
            "Bayes CI Lo": fmt(b_lo) if b_lo else "N/A",
            "Bayes CI Hi": fmt(b_hi) if b_hi else "N/A",
            "Bayes Covered": b_cov,
            "Bayes Bias": f"{b_bias:+.4f}" if b_bias else "N/A",
            "_mle_cov": mle_cov == "YES",
        })

df_table = pd.DataFrame(table_rows)
df_table.drop(columns=["_mle_cov"]).to_csv("stats/mle_comparison_table.csv", index=False)
print("\nsaved stats/mle_comparison_table.csv")



fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

for ax, truth_model in zip(axes, TRUTH_MODELS):
    true_gamma = TRUE_GAMMA[truth_model]
    ax.axvline(true_gamma, color="black", linestyle="--",
               linewidth=1.4, label="True γ", zorder=1)

    # MLE
    mle = mle_results[truth_model]
    mean = mle["gamma"]["mle"]
    lo = mle["gamma"]["ci_lo"]
    hi = mle["gamma"]["ci_hi"]
    ax.errorbar(mean, 0.15,
                xerr=[[mean - lo], [hi - mean]],
                fmt="s", color=COLORS["MLE"],
                capsize=6, markersize=8, linewidth=2,
                label="MLE (90% CI)", zorder=2)

    # Bayes
    bayes = bayes_results.get(truth_model)
    if bayes and "gamma" in bayes:
        bm = bayes["gamma"]["mean"]
        bl = bayes["gamma"]["ci_lo"]
        bh = bayes["gamma"]["ci_hi"]
        ax.errorbar(bm, -0.15,
                    xerr=[[bm - bl], [bh - bm]],
                    fmt="o", color=COLORS["Bayes"],
                    capsize=6, markersize=8, linewidth=2,
                    label="Bayesian — Normal/Log-Normal (90% CI)", zorder=2)

    ax.set_title(TRUTH_LABELS[truth_model], fontsize=13, fontweight="bold")
    ax.set_xlabel("Spectral index γ", fontsize=11)
    ax.set_yticks([])
    ax.set_ylim(-0.45, 0.45)

handles = [
    mpatches.Patch(color=COLORS["MLE"], label="MLE (90% CI via Fisher information)"),
    mpatches.Patch(color=COLORS["Bayes"], label="Bayesian — Normal/Log-Normal (90% CI)"),
    plt.Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.4, label="True γ"),
]
fig.legend(handles=handles, loc="lower center", ncol=3,
           fontsize=11, bbox_to_anchor=(0.5, -0.05), frameon=False)
fig.suptitle(
    "MLE vs Bayesian Posterior: Spectral Index γ\n"
    "(SPL inference model, 90% intervals)",
    fontsize=13, fontweight="bold", y=1.02,
)
plt.tight_layout()
fig.savefig("figures/mle_comparison_forest.png", dpi=200, bbox_inches="tight")
plt.close()
print("saved figures/mle_comparison_forest.png")



fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(TRUTH_MODELS))
width = 0.35

mle_biases = [mle_results[tm]["gamma"]["mle"] - TRUE_GAMMA[tm] for tm in TRUTH_MODELS]
bayes_biases = []
for tm in TRUTH_MODELS:
    b = bayes_results.get(tm)
    bayes_biases.append(b["gamma"]["mean"] - TRUE_GAMMA[tm] if b and "gamma" in b else np.nan)

ax.bar(x - width/2, mle_biases,   width, label="MLE",
       color=COLORS["MLE"],   alpha=0.85, edgecolor="white")
ax.bar(x + width/2, bayes_biases, width, label="Bayes (Normal/Log-Normal)",
       color=COLORS["Bayes"], alpha=0.85, edgecolor="white")

ax.axhline(0, color="black", linewidth=1.0)
ax.set_xticks(x)
ax.set_xticklabels([TRUTH_LABELS[tm] for tm in TRUTH_MODELS], fontsize=12)
ax.set_ylabel("Bias in γ (estimate − true)", fontsize=11)
ax.set_title("Bias Comparison: MLE vs Bayesian Posterior Mean for γ", fontsize=12, fontweight="bold")
ax.legend(fontsize=11, frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.3, linestyle="--")

for i, (mb, bb) in enumerate(zip(mle_biases, bayes_biases)):
    ax.text(i - width/2, mb + 0.01*np.sign(mb),
            f"{mb:+.3f}", ha="center",
            va="bottom" if mb >= 0 else "top",
            fontsize=9, color=COLORS["MLE"])
    if not np.isnan(bb):
        ax.text(i + width/2, bb + 0.01*np.sign(bb),
                f"{bb:+.3f}", ha="center",
                va="bottom" if bb >= 0 else "top",
                fontsize=9, color=COLORS["Bayes"])

plt.tight_layout()
fig.savefig("figures/mle_bias_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/mle_bias_comparison.png")


print("SUMMARY of MLE vs Bayesian for γ")
print(f"{'Truth':<16} {'True γ':>7} | "
      f"{'MLE':>7} {'MLE CI':>18} {'Cov':>4} | "
      f"{'Bayes':>7} {'Bayes CI':>18} {'Cov':>4}")
print("-"*80)
for tm in TRUTH_MODELS:
    tg = TRUE_GAMMA[tm]
    mle = mle_results[tm]
    bayes = bayes_results.get(tm)
    mm = mle["gamma"]["mle"]
    ml = mle["gamma"]["ci_lo"]
    mh = mle["gamma"]["ci_hi"]
    mc = "YES" if ml <= tg <= mh else "NO"
    bm = bayes["gamma"]["mean"]  if bayes and "gamma" in bayes else float("nan")
    bl = bayes["gamma"]["ci_lo"] if bayes and "gamma" in bayes else float("nan")
    bh = bayes["gamma"]["ci_hi"] if bayes and "gamma" in bayes else float("nan")
    bc = ("YES" if bl <= tg <= bh else "NO") if not np.isnan(bl) else "N/A"
    print(f"{TRUTH_LABELS[tm]:<16} {tg:>7.3f} | "
          f"{mm:>7.3f} [{ml:.3f}, {mh:.3f}] {mc:>4} | "
          f"{bm:>7.3f} [{bl:.3f}, {bh:.3f}] {bc:>4}")