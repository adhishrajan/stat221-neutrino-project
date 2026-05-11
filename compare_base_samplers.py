"""
compare_samplers_base.py

Runs RWMH, MALA, and Stan/NUTS on the same synthetic base-model dataset
and produces three publication-quality figures:

  figures/sampler_comparison_forest.png   -- forest plot (mean + 90% CI)
  figures/gamma_posterior_comparison.png  -- overlaid gamma KDE
  figures/ess_per_second.png              -- ESS/sec bar chart

Run from your project root:
    python compare_samplers_base.py
"""

import os
import numpy as np
import time
import cmdstanpy
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde

from simulator import generate_single_season, E_REF
from posteriors import log_posterior_base, grad_log_posterior_base
from samplers import (
    run_rwmh, run_mala,
    run_multiple_chains, estimate_dense_precond_from_rwmh,
    effective_sample_size_multi, _stack_chains,
)

os.makedirs("figures", exist_ok=True)

# ── single-chain ESS (not in your current samplers.py, defined here) ─────────
def effective_sample_size(chain: np.ndarray) -> float:
    n = len(chain)
    if n < 10:
        return float(n)
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30:
        return 1.0
    fft_x = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (var * n)
    T = 1
    for k in range(1, n):
        if acf[k] < 0:
            break
        T += 1
    tau = max(1.0 + 2.0 * np.sum(acf[1:T]), 1.0)
    return n / tau


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
rng = np.random.default_rng(42)
ds  = generate_single_season(rng=rng)

log_E_norm = np.log(ds.E_centers / E_REF)
log_E_mean = float(np.mean(log_E_norm))

param_names  = ["log_phi", "gamma", "log_eta", "delta"]
display_names = {
    "log_phi": r"$\log\,\phi$",
    "gamma":   r"$\gamma$",
    "log_eta": r"$\log\,\eta$",
    "delta":   r"$\delta$",
}
true_vals = {
    "log_phi": np.log(ds.true_params["phi"]),
    "gamma":   ds.true_params["gamma"],
    "log_eta": np.log(ds.true_params["eta"]),
    "delta":   ds.true_params["delta"],
}

theta_true = np.array([true_vals[n] for n in param_names])
theta_init = theta_true + rng.normal(0, 0.05, size=4)

prior_type = "lognormal_gamma"

def log_post(theta):
    return log_posterior_base(theta, ds.counts, ds.E_centers, ds.E_widths, prior_type)

def grad_log_post(theta):
    return grad_log_posterior_base(theta, ds.counts, ds.E_centers, ds.E_widths, prior_type)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RWMH  (4 chains, 50k iterations, 10k burnin)
# ─────────────────────────────────────────────────────────────────────────────
print("Running RWMH (4 chains)...")
rwmh_results = run_multiple_chains(
    run_rwmh,
    theta_init=theta_init,
    n_chains=4,
    init_strategy="jitter",
    init_scale=0.1,
    rng=np.random.default_rng(221),
    log_posterior_fn=log_post,
    n_iterations=50000,
    n_burnin=10000,
    adapt_proposal=True,
    adapt_until=10000,
    param_names=param_names
)
rwmh_stacked = _stack_chains(rwmh_results)   # (4, 40000, 4)
rwmh_combined = rwmh_stacked.reshape(-1, 4)  # (160000, 4)
rwmh_time = sum(r.wall_time for r in rwmh_results)
print(f"  Done. Total wall time: {rwmh_time:.1f}s, "
      f"mean accept: {np.mean([r.acceptance_rate for r in rwmh_results]):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MALA  (4 chains, 50k iterations, 10k burnin, RWMH preconditioner)
# ─────────────────────────────────────────────────────────────────────────────
print("Running MALA (4 chains)...")
rwmh_cov = estimate_dense_precond_from_rwmh(rwmh_results, ridge=1e-6)

mala_results = run_multiple_chains(
    run_mala,
    theta_init=theta_init,
    n_chains=4,
    init_strategy="jitter",
    init_scale=0.1,
    rng=np.random.default_rng(222),
    log_posterior_fn=log_post,
    grad_log_posterior_fn=grad_log_post,
    n_iterations=50000,
    n_burnin=10000,
    step_size=0.01,
    adapt_step=True,
    adapt_until=10000,
    target_accept=0.57,
    param_names=param_names,
    precond=rwmh_cov,
    precond_type="dense",
    normalize_precond=True,
)
mala_stacked = _stack_chains(mala_results)
mala_combined = mala_stacked.reshape(-1, 4)
mala_time = sum(r.wall_time for r in mala_results)
print(f"  Done. Total wall time: {mala_time:.1f}s, "
      f"mean accept: {np.mean([r.acceptance_rate for r in mala_results]):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Stan / NUTS  (4 chains, 10k warmup, 10k sampling)
# ─────────────────────────────────────────────────────────────────────────────
print("Running Stan/NUTS (4 chains)...")
stan_data = {
    "B":          len(ds.counts),
    "counts":     ds.counts.tolist(),
    "log_E_norm": log_E_norm.tolist(),
    "E_widths":   ds.E_widths.tolist(),
    "log_E_mean": log_E_mean,
}

model = cmdstanpy.CmdStanModel(stan_file="base_model.stan", force_compile=False)
t0 = time.time()
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=10000,
    iter_sampling=10000,
    seed=221,
    adapt_delta=0.95,
    show_progress=False,
    show_console=False,
)
stan_time = time.time() - t0

divs = sum(int(c.sum()) for c in fit.method_variables()['divergent__'].T)
print(f"  Done. Wall time: {stan_time:.1f}s, divergences: {divs}")

log_eta_stan = (fit.stan_variable("log_eta_at_mean")
                + fit.stan_variable("delta") * log_E_mean)

stan_vars = {
    "log_phi": fit.stan_variable("log_phi").flatten(),
    "gamma":   fit.stan_variable("gamma").flatten(),
    "log_eta": log_eta_stan.flatten(),
    "delta":   fit.stan_variable("delta").flatten(),
}


# ─────────────────────────────────────────────────────────────────────────────
# 5. EQUALIZE sample counts for fair plotting
# ─────────────────────────────────────────────────────────────────────────────
n_plot = min(
    len(rwmh_combined),
    len(mala_combined),
    len(stan_vars["gamma"]),
)
# subsample randomly so we don't always take the first n
rng_plot = np.random.default_rng(0)
idx_rwmh = rng_plot.choice(len(rwmh_combined), n_plot, replace=False)
idx_mala = rng_plot.choice(len(mala_combined), n_plot, replace=False)
idx_stan = rng_plot.choice(len(stan_vars["gamma"]), n_plot, replace=False)

sampler_data = {
    "RWMH": {n: rwmh_combined[idx_rwmh, i] for i, n in enumerate(param_names)},
    "MALA": {n: mala_combined[idx_mala, i] for i, n in enumerate(param_names)},
    "Stan": {n: stan_vars[n][idx_stan]      for n in param_names},
}
sampler_times = {"RWMH": rwmh_time, "MALA": mala_time, "Stan": stan_time}

print(f"\nPlotting with {n_plot:,} equalised samples per sampler.")


# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
COLORS   = {"RWMH": "#4878CF", "MALA": "#D65F5F", "Stan": "#3CB371"}
SAMPLERS = ["RWMH", "MALA", "Stan"]
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Forest plot
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
offsets = {"RWMH": 0.18, "MALA": 0.0, "Stan": -0.18}

for ax, name in zip(axes, param_names):
    true_val = true_vals[name]
    ax.axvline(true_val, color="black", linestyle="--", linewidth=1.4,
               label="True value", zorder=1)

    for sampler in SAMPLERS:
        chain = sampler_data[sampler][name]
        mean  = np.mean(chain)
        lo, hi = np.percentile(chain, [5, 95])
        y = offsets[sampler]
        ax.errorbar(mean, y,
                    xerr=[[mean - lo], [hi - mean]],
                    fmt="o", color=COLORS[sampler],
                    capsize=6, markersize=8, linewidth=2,
                    label=sampler, zorder=2)

    ax.set_title(display_names[name], fontsize=15, pad=6)
    ax.set_yticks([])
    ax.set_ylim(-0.45, 0.45)
    ax.set_xlabel("Posterior value", fontsize=10)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.4)

# legend below the figure
patches = [mpatches.Patch(color=COLORS[s], label=s) for s in SAMPLERS]
patches.append(plt.Line2D([0], [0], color="black", linestyle="--",
                           linewidth=1.4, label="True value"))
fig.subplots_adjust(wspace=0.4)
fig.legend(handles=patches, loc="lower center", ncol=4,
           fontsize=12, bbox_to_anchor=(0.5, -0.04),
           frameon=False)

fig.suptitle("Posterior comparison: RWMH vs MALA vs Stan/NUTS\n"
             "(dot = mean, bars = 90% credible interval)",
             fontsize=12, y=1.02)
plt.tight_layout()
fig.savefig("figures/sampler_comparison_forest.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/sampler_comparison_forest.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Overlaid gamma posteriors
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4.2))

bw = 0.13   # same bandwidth for all three → fair smoothness comparison
all_gamma = np.concatenate([sampler_data[s]["gamma"] for s in SAMPLERS])
x_min, x_max = all_gamma.min() - 0.1, all_gamma.max() + 0.1
x_range = np.linspace(x_min, x_max, 600)

for sampler in SAMPLERS:
    chain = sampler_data[sampler]["gamma"]
    kde   = gaussian_kde(chain, bw_method=bw)
    y     = kde(x_range)
    ax.plot(x_range, y, color=COLORS[sampler], linewidth=2.5, label=sampler)
    ax.fill_between(x_range, y, alpha=0.08, color=COLORS[sampler])

true_gamma = true_vals["gamma"]
ax.axvline(true_gamma, color="black", linestyle="--", linewidth=1.6,
           label=f"True $\\gamma$ = {true_gamma}")

ax.set_xlabel(r"Spectral index $\gamma$", fontsize=13)
ax.set_ylabel("Posterior density", fontsize=12)
ax.set_title(r"Posterior on $\gamma$: all three samplers agree", fontsize=13)
ax.legend(fontsize=11, frameon=False)
plt.tight_layout()
fig.savefig("figures/gamma_posterior_comparison.png",
            dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/gamma_posterior_comparison.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — ESS per second bar chart
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4.2))
x     = np.arange(len(param_names))
width = 0.25

for i, sampler in enumerate(SAMPLERS):
    ess_per_sec = []
    for name in param_names:
        chain = sampler_data[sampler][name]
        # use the full (not subsampled) chain for ESS
        if sampler == "RWMH":
            full = rwmh_combined[:, param_names.index(name)]
        elif sampler == "MALA":
            full = mala_combined[:, param_names.index(name)]
        else:
            full = stan_vars[name]
        ess  = effective_sample_size(full)
        ess_per_sec.append(ess / sampler_times[sampler])

    bars = ax.bar(x + (i - 1) * width, ess_per_sec, width,
                  label=sampler, color=COLORS[sampler],
                  alpha=0.88, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([display_names[n] for n in param_names], fontsize=13)
ax.set_ylabel("ESS / second", fontsize=12)
ax.set_title("Sampling efficiency: effective samples per second", fontsize=13)
ax.legend(fontsize=11, frameon=False)
plt.tight_layout()
fig.savefig("figures/ess_per_second.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved figures/ess_per_second.png")


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("SAMPLER SUMMARY")
print("="*80)
print(f"{'Sampler':<8} {'Chains':>6} {'Samples':>10} {'Wall time':>11} "
      f"{'Accept%':>9} {'Divergences':>12}")
print("-"*60)
print(f"{'RWMH':<8} {'4':>6} {len(rwmh_combined):>10,} "
      f"{rwmh_time:>10.1f}s "
      f"{np.mean([r.acceptance_rate for r in rwmh_results])*100:>8.1f}%"
      f"{'—':>13}")
print(f"{'MALA':<8} {'4':>6} {len(mala_combined):>10,} "
      f"{mala_time:>10.1f}s "
      f"{np.mean([r.acceptance_rate for r in mala_results])*100:>8.1f}%"
      f"{'—':>13}")
print(f"{'Stan'::<8} {'4':>6} {len(stan_vars['gamma']):>10,} "
      f"{stan_time:>10.1f}s {'—':>9}% {divs:>12}")

print(f"\n{'Param':<10} {'True':>9} | "
      f"{'RWMH mean':>10} {'RWMH SD':>8} | "
      f"{'MALA mean':>10} {'MALA SD':>8} | "
      f"{'Stan mean':>10} {'Stan SD':>8}")
print("-"*85)
for i, name in enumerate(param_names):
    tv = true_vals[name]
    rc = rwmh_combined[:, i]
    mc = mala_combined[:, i]
    sc = stan_vars[name]
    print(f"{name:<10} {tv:>9.4f} | "
          f"{np.mean(rc):>10.4f} {np.std(rc):>8.4f} | "
          f"{np.mean(mc):>10.4f} {np.std(mc):>8.4f} | "
          f"{np.mean(sc):>10.4f} {np.std(sc):>8.4f}")