"""
parameter_recovery_table_hierarchical.py

Produces a parameter recovery table for gamma across all combinations of:
  - truth model:      power_law, broken_power_law, cutoff
  - prior:            weakly, flat, lognormal_gamma
  - parameterization: centered, noncentered
  - sampler:          rwmh, mala

For each combination, reports:
  - true gamma
  - posterior mean
  - 90% CI lower and upper
  - coverage indicator
  - bias 
Outputs:
  - Figures/parameter_recovery_table_hierarchical.png  
  - Stats/parameter_recovery_table_hierarchical.csv    
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

os.makedirs("Figures", exist_ok=True)
os.makedirs("Stats", exist_ok=True)

TRUTH_MODELS = ["power_law", "broken_power_law", "cutoff"]
PRIORS = ["weakly", "flat", "lognormal_gamma"]
PARAMETERIZATIONS = ["centered", "noncentered"]
SAMPLERS = ["rwmh", "mala"]

TRUTH_LABELS = {
    "power_law":       "SPL",
    "broken_power_law":"BPL",
    "cutoff":          "Cutoff",
}
PRIOR_LABELS = {
    "weakly":          "Normal/Log-Normal",
    "flat":            "Flat",
    "lognormal_gamma": "Log-Normal γ",
}
PARAM_LABELS = {"centered": "Centered", "noncentered": "Non-centered"}
SAMPLER_LABELS = {"rwmh": "RWMH", "mala": "MALA"}

TRUE_GAMMA = {"power_law": 2.5, "broken_power_law": 2.5, "cutoff": 2.5}

rows = []

for truth_model in TRUTH_MODELS:
    for prior in PRIORS:
        for parameterization in PARAMETERIZATIONS:
            for sampler in SAMPLERS:
                path = (
                    f"Stats/posterior_summary_hierarchical_"
                    f"{truth_model}_{prior}_{parameterization}_{sampler}.csv"
                )
                if not os.path.exists(path):
                    print(f"  MISSING: {path}")
                    continue

                df = pd.read_csv(path, index_col=0)

                if "gamma" not in df.index:
                    print(f"  WARNING: 'gamma' not found in {path}")
                    continue

                row = df.loc["gamma"]
                true_gamma = TRUE_GAMMA[truth_model]
                mean_val = float(row["mean"])
                ci_lo = float(row["ci_lower"])
                ci_hi = float(row["ci_upper"])
                covered = ci_lo <= true_gamma <= ci_hi
                bias = mean_val - true_gamma

                rows.append({
                    "Truth Model":     TRUTH_LABELS[truth_model],
                    "Prior":           PRIOR_LABELS[prior],
                    "Param.":          PARAM_LABELS[parameterization],
                    "Sampler":         SAMPLER_LABELS[sampler],
                    "True γ":          true_gamma,
                    "Post. Mean":      round(mean_val, 3),
                    "90% CI Lo":       round(ci_lo, 3),
                    "90% CI Hi":       round(ci_hi, 3),
                    "Bias":            round(bias, 3),
                    "Covered":         "YES" if covered else "NO",
                    "_covered":        covered,
                })

if not rows:
    raise RuntimeError(
        "No CSV files found. Run run_all_hierarchical_samplers.py first."
    )

table_df = pd.DataFrame(rows)

csv_df = table_df.drop(columns=["_covered"])
csv_df.to_csv("Stats/parameter_recovery_table_hierarchical.csv", index=False)
print("Saved Stats/parameter_recovery_table_hierarchical.csv")

display_cols = [
    "Truth Model", "Prior", "Param.", "Sampler",
    "True γ", "Post. Mean", "90% CI Lo", "90% CI Hi", "Bias", "Covered",
]
plot_df = table_df[display_cols].copy()

n_rows = len(plot_df)
n_cols = len(display_cols)

fig_height = max(4.0, 0.35 * n_rows + 1.2)
fig, ax = plt.subplots(figsize=(16, fig_height))
ax.axis("off")

col_widths = [0.10, 0.15, 0.11, 0.08, 0.08, 0.09, 0.09, 0.09, 0.08, 0.08]

tbl = ax.table(
    cellText=plot_df.values,
    colLabels=display_cols,
    cellLoc="center",
    loc="center",
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.45)

# header row styling
for j in range(n_cols):
    cell = tbl[0, j]
    cell.set_facecolor("#1a1a2e")
    cell.set_text_props(color="white", fontweight="bold")

GREEN_LIGHT  = to_rgba("#d4edda")
RED_LIGHT    = to_rgba("#f8d7da")
ALT_LIGHT    = to_rgba("#f8f9fa")
WHITE        = to_rgba("white")

covered_col_idx = display_cols.index("Covered")

for i, row_data in enumerate(rows):
    bg      = ALT_LIGHT if i % 2 == 0 else WHITE
    covered = row_data["_covered"]

    for j in range(n_cols):
        cell = tbl[i + 1, j]
        if j == covered_col_idx:
            cell.set_facecolor(GREEN_LIGHT if covered else RED_LIGHT)
            cell.set_text_props(
                fontweight="bold",
                color="#155724" if covered else "#721c24",
            )
        else:
            cell.set_facecolor(bg)
        cell.set_edgecolor("#dee2e6")

# bold the first row of each new truth model group
prev_tm = None
for i, row_data in enumerate(rows):
    tm = row_data["Truth Model"]
    if tm != prev_tm:
        for j in range(n_cols):
            tbl[i + 1, j].set_text_props(fontweight="bold")
        prev_tm = tm

n_covered     = sum(r["_covered"] for r in rows)
n_not_covered = n_rows - n_covered

fig.suptitle(
    "Parameter Recovery (Hierarchical Model): Posterior Mean and 90% CI for γ\n"
    f"(SPL inference applied to all truth models — {n_covered}/{n_rows} covered)",
    fontsize=12, fontweight="bold", y=0.99,
)

plt.tight_layout()
fig.savefig("Figures/parameter_recovery_table_hierarchical.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved Figures/parameter_recovery_table_hierarchical.png")
print(f"\nTotal rows: {n_rows}  Covered: {n_covered}  Not covered: {n_not_covered}")
