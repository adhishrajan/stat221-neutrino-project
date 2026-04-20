import numpy as np
import cmdstanpy
import pandas as pd
from simulator import generate_single_season, E_REF

rng = np.random.default_rng(111)
ds = generate_single_season(rng=rng)

log_E_norm = np.log(ds.E_centers / E_REF)
log_E_mean = float(np.mean(log_E_norm))

stan_data = {
    "B": len(ds.counts),
    "counts": ds.counts.tolist(),
    "log_E_norm": log_E_norm.tolist(),
    "E_widths": ds.E_widths.tolist(),
    "log_E_mean": log_E_mean,
}

model = cmdstanpy.CmdStanModel(stan_file="base_model.stan", force_compile=True)
fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=5000,
    iter_sampling=10000,
    seed=221,
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
    "gamma": (ds.true_params["gamma"], fit.stan_variable("gamma")),
    "log_eta": (np.log(ds.true_params["eta"]), fit.stan_variable("log_eta")),
    "delta": (ds.true_params["delta"], fit.stan_variable("delta")),
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

print("\n=== POSTERIOR PREDICTIVE CHECK ===")
print(f"{'Bin':>4} {'Observed':>10} {'Rep mean':>10} {'Rep 5%':>8} {'Rep 95%':>9}")
for b in range(len(ds.counts)):
    flag = " *" if ds.counts[b] < rep_lo[b] or ds.counts[b] > rep_hi[b] else ""
    print(f"  {b+1:>2}   {ds.counts[b]:>8}   {rep_mean[b]:>9.1f}   {rep_lo[b]:>7.1f}   {rep_hi[b]:>8.1f}{flag}")