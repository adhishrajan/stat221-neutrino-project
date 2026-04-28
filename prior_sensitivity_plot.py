"""
prior_sensitivity_plot.py

Produces a prior sensitivity figure for gamma showing how the posterior
shifts across the three prior schemes, for each truth model.
Reads from stats/posterior_summary_base_{truth_model}_{prior}_{sampler}.csv
Saves:
  - Figures/prior_sensitivity_gamma.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("Figures", exist_ok=True)

TRUTH_MODELS = ["power_law", "broken_power_law", "cutoff"]
PRIORS = ["weakly", "flat", "lognormal_gamma"]
SAMPLERS = ["rwmh", "mala"]

TRUTH_LABELS = {
    "power_law": "SPL Truth (γ = 2.5)",
    "broken_power_law": "BPL Truth (γ₁ = 2.5, γ₂ = 2.8)",
    "cutoff": "Cutoff Truth (γ = 2.5, E_cut = 1 PeV)",
}
PRIOR_LABELS = {
    "weakly": "Normal/Log-Normal",
    "flat": "Flat",
    "lognormal_gamma": "Log-Normal γ",
}
SAMPLER_LABELS = {"rwmh": "RWMH", "mala": "MALA"}

TRUE_GAMMA = {
    "power_law": 2.5,
    "broken_power_law": 2.5,
    "cutoff": 2.5,
}

COLORS = {
    "weakly": "#4878CF",
    "flat": "#D65F5F",
    "lognormal_gamma": "#3CB371",
}

# y offsets
OFFSETS = {
    "weakly": 0.18,
    "flat": 0.0,
    "lognormal_gamma": -0.18,
}

# load results
results = {}
missing = []

for truth_model in TRUTH_MODELS:
    results[truth_model] = {}
    for sampler in SAMPLERS:
        results[truth_model][sampler] = {}
        for prior in PRIORS:
            path = f"Stats/posterior_summary_base_{truth_model}_{prior}_{sampler}.csv"
            if not os.path.exists(path):
                missing.append(path)
                results[truth_model][sampler][prior] = None
                continue
            df = pd.read_csv(path, index_col=0)
            if "gamma" not in df.index:
                results[truth_model][sampler][prior] = None
                continue
            row = df.loc["gamma"]
            results[truth_model][sampler][prior] = {
                "mean":     float(row["mean"]),
                "ci_lower": float(row["ci_lower"]),
                "ci_upper": float(row["ci_upper"]),
            }

if missing:
    print("WARNING — missing files:")
    for p in missing:
        print(f"  {p}")
#plot
n_rows = len(TRUTH_MODELS)
n_cols = len(SAMPLERS)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(11, 3.8 * n_rows),
    sharey=False,
)

plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

for row_idx, truth_model in enumerate(TRUTH_MODELS):
    true_gamma = TRUE_GAMMA[truth_model]

    for col_idx, sampler in enumerate(SAMPLERS):
        ax = axes[row_idx, col_idx]

        # true value
        ax.axvline(
            true_gamma,
            color="black", linestyle="--", linewidth=1.4,
            label="True γ", zorder=1,
        )

        any_data = False
        for prior in PRIORS:
            res = results[truth_model][sampler][prior]
            if res is None:
                continue
            any_data = True

            mean = res["mean"]
            ci_lo = res["ci_lower"]
            ci_hi = res["ci_upper"]
            y_off = OFFSETS[prior]

            ax.errorbar(
                mean, y_off,
                xerr=[[mean - ci_lo], [ci_hi - mean]],
                fmt="o",
                color=COLORS[prior],
                capsize=6,
                markersize=8,
                linewidth=2,
                label=PRIOR_LABELS[prior],
                zorder=2,
            )

        ax.set_yticks([])
        ax.set_ylim(-0.45, 0.45)
        ax.set_xlabel("Spectral index γ", fontsize=11)


        if row_idx == 0:
            ax.set_title(SAMPLER_LABELS[sampler], fontsize=13, fontweight="bold", pad=10)


        if col_idx == 0:
            ax.set_ylabel(
                TRUTH_LABELS[truth_model],
                fontsize=10, labelpad=10,
                fontweight="bold",
            )

        if not any_data:
            ax.text(
                0.5, 0.5, "No data",
                transform=ax.transAxes,
                ha="center", va="center",
                color="gray", fontsize=10,
            )


patches = [
    mpatches.Patch(color=COLORS[p], label=PRIOR_LABELS[p])
    for p in PRIORS
]
patches.append(
    plt.Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.4, label="True γ")
)
fig.legend(
    handles=patches,
    loc="lower center",
    ncol=len(PRIORS) + 1,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.03),
    frameon=False,
)

fig.suptitle(
    "Prior Sensitivity: Posterior Mean and 90% CI for γ\n"
    "across Prior Schemes, Truth Models, and Samplers",
    fontsize=13, fontweight="bold", y=1.01,
)

plt.tight_layout()
fig.savefig("Figures/prior_sensitivity_gamma.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved Figures/prior_sensitivity_gamma.png")