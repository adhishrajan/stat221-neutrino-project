import numpy as np
import cmdstanpy
import pandas as pd
from simulator import generate_hierarchical, E_REF

rng = np.random.default_rng(221)
hds = generate_hierarchical(rng=rng)
K = len(hds.seasons)
B = len(hds.seasons[0].counts)

log_E_norm = np.log(hds.seasons[0].E_centers / E_REF)
log_E_mean = float(np.mean(log_E_norm))

# building K x B counts array
counts_array = np.array([hds.seasons[k].counts for k in range(K)])

stan_data = {
    "K": K,
    "B": B,
    "counts": counts_array.tolist(),
    "log_E_norm": log_E_norm.tolist(),
    "E_widths": hds.seasons[0].E_widths.tolist(),
    "log_E_mean": log_E_mean,
}

model = cmdstanpy.CmdStanModel(
    stan_file="hierarchical_model.stan",
    force_compile=True,
)

fit = model.sample(
    data=stan_data,
    chains=4,
    iter_warmup=5000,
    iter_sampling=10000,
    seed=221,
    adapt_delta=0.95,
    show_progress=True,
)

pd.set_option('display.max_rows', 200)
pd.set_option('display.width', 120)

summary = fit.summary()
main_params = (["lp__", "mu_phi", "sigma_phi", "gamma", "log_eta", "delta"]
               + [f"log_phi_k[{k+1}]" for k in range(K)])
print("MAIN PARAMETERS")
print(summary.loc[main_params].to_string())

print(f"\nDivergences per chain:")
for i, chain in enumerate(fit.method_variables()['divergent__'].T):
    print(f"  Chain {i+1}: {int(chain.sum())}")

print("\nSHARED PARAMETER COMPARISON")
print(f"{'Param':<14} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 70)

shared = {
    "gamma":   (hds.true_params["gamma"], fit.stan_variable("gamma")),
    "log_eta": (np.log(hds.true_params["eta"]), fit.stan_variable("log_eta")),
    "delta":   (hds.true_params["delta"], fit.stan_variable("delta")),
    "mu_phi":  (hds.true_params["mu_phi"], fit.stan_variable("mu_phi")),
    "sigma_phi":(hds.true_params["sigma_phi"], fit.stan_variable("sigma_phi")),
}

for name, (true_val, samples) in shared.items():
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{name:<14} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")

print("\nSEASON SPECIFIC log_phi_k COMPARISON")
print(f"{'Season':<8} {'True':>10} {'Stan mean':>12} {'Stan SD':>10} {'Stan 5%':>10} {'Stan 95%':>10}")
print("-" * 65)

log_phi_k_samples = fit.stan_variable("log_phi_k")
for k in range(K):
    true_val = np.log(hds.phi_k[k])
    samples  = log_phi_k_samples[:, k]
    mean = np.mean(samples)
    sd = np.std(samples)
    lo = np.percentile(samples, 5)
    hi = np.percentile(samples, 95)
    flag = "OK" if lo <= true_val <= hi else "outside 90% CI"
    print(f"{k+1:<8} {true_val:>10.4f} {mean:>12.4f} {sd:>10.4f} {lo:>10.4f} {hi:>10.4f}  {flag}")