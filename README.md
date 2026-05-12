# stat221-neutrino-project

Bayesian inference for astrophysical neutrino spectra, motivated by IceCube-style binned energy observations. Observed bin counts are modeled as Poisson draws from a mixture of astrophysical signal and atmospheric background, fit spectral models via custom MCMC, and compare models using Bayes factors.

---

## Key files

The files below contain the core statistical implementation. Everything else (notebooks, plotting scripts, simulation study runners) builds on these.

### `simulator.py`

Generates synthetic Poisson-binned datasets for single-season and multi-season (hierarchical) settings.

**Key Components:**
- `signal_power_law`, `signal_broken_power_law`, `signal_cutoff`: three astrophysical signal models
- `atmospheric_shape` / `background_atmospheric`: atmospheric neutrino background (simple power law or conventional + prompt)
- `generate_single_season`: draws Poisson counts from signal + background; returns a `Dataset` with true parameters and expected counts
- `generate_hierarchical`: draws K seasons with per-season flux normalizations from a log-normal population; returns a `HierarchicalDataset`
- `Dataset` / `HierarchicalDataset`: dataclasses that carry observed counts, energy bins, true parameters, and model options through the rest of the pipeline

---

### `posteriors.py`

Log posterior and analytic gradient functions for both the single-season base model and the multi-season hierarchical model. For strictly positive parameters, sampling is done in unconstrained log-transformed space.

**Parameter transformations (base model):**

| Natural | Transformed |
|---------|-------------|
| $\phi$  | $\log \phi$ |
| $\gamma$| $\gamma$    |
| $\eta$  | $\log \eta$ |
| $\delta$| $\delta$    |

**Key Components:**
- `log_prior_base` / `log_prior_spectral`: encode three prior types: `weakly` (log-normal on $\phi$, $\eta$; normal on $\gamma$, $\delta$), `flat` (bounded), `lognormal_gamma` (log-normal on $\gamma$, bounded flat priors on the others)
- `log_posterior_base`: Poisson log-likelihood + prior + change-of-variables Jacobian for the base model
- `grad_log_posterior_base`: analytic gradient of the above (used by MALA); includes likelihood ratio terms and prior gradients in transformed space
- `verify_gradient`: finite-difference check confirming analytic == numerical gradient
- `log_posterior_hierarchical` / `grad_log_posterior_hierarchical`: extends the above to K seasons sharing $\gamma$, $\delta$ with per-season $\phi_k$'s drawn from $N(\mu_\phi, \sigma_\phi^2)$ and per-season $\eta_k$'s drawn from $N(\mu_\eta, \sigma_\eta^2)$ ; supports both centered and non-centered parameterizations

---

### `samplers.py`

Custom MCMC samplers built from scratch: Random-Walk Metropolis–Hastings (RWMH) and Metropolis-Adjusted Langevin Algorithm (MALA)

**Key Components:**
- `run_rwmh`: adaptive RWMH; updates proposal covariance as (2.38²/d) x empirical_cov during a burn-in window (following Haario et al. 2001)
- `run_mala`: MALA with optional dense preconditioning; adapts step size toward a target acceptance rate (≈0.60); gradient supplied by `posteriors.py`
- `run_multiple_chains`: launches N independent chains with jittered initialization; used for all inference pipelines (default N=4)
- `split_rhat`: split-chain R-hat convergence diagnostic
- `effective_sample_size`: ESS estimate from autocorrelation
- `plot_traceplots` / `plot_acf`: visual diagnostics
- `MCMCResult`: dataclass returned by each sampler; carries samples, log-posteriors, acceptance rate, wall time

---

### `base_model.stan` + `run_stan_base.py`

Stan implementation of the single-season SPL model, used as a NUTS baseline against the hand-coded samplers.

**`base_model.stan` — Key Components:**
- **Reparameterized $\eta$**: the model samples `log_eta_at_mean` = $\log \eta$ − $\delta$ · mean(log E/E_ref) rather than log $\eta$ directly; this decorrelates $\eta$ from $\delta$ and substantially improves geometry for NUTS
- **Three prior types via integer flag**: `prior_type` is passed as data (0 = flat, 1 = weakly, 2 = lognormal_gamma), keeping the model in a single file that matches all Python configurations exactly
- **Jacobian correction for flat/lognormal_gamma**: `target += log_phi + log_eta` is added explicitly when `prior_type != 1`, matching the change-of-variables term in `posteriors.py`
- **`generated quantities` block**: produces posterior predictive replications `counts_rep` for model checking

**`run_stan_base.py` — Key Components:**
- Assembles the effective response matrix `M_eff = smearing x (acceptance x exposure)` from the `Dataset.model_options` dict, matching exactly what `posteriors.py` passes through `_apply_response`
- Loops over all `truth_model x prior_type` combinations 
- Saves posterior summary and diagnostics CSVs to `Stats/` using the same naming convention as `run_all_base_samplers.py`, so results from both are directly comparable

---

### `run_all_base_samplers.py`

Runs the hand-coded samplers (RWMH and MALA) across every combination of truth model, inference model, and prior type for the single-season base model.

**Key Components:**
- **27-combination outer loop**: 3 truth models x 3 inference models x 3 prior types; data is generated once per truth model and reused across all recover model x prior type pairs, enabling fair model comparison on the same dataset
- **MALA warm-start**: MALA's dense preconditioner is estimated from the RWMH posterior covariance (`estimate_dense_precond_from_rwmh`), so MALA benefits from an already-adapted geometry
- **Outputs to `Stats/`**: posterior summary CSVs (`posterior_summary_base_{truth}_{inference}_{prior}_{sampler}.csv`) and per-parameter ESS, ESS/s, and R-hat diagnostics tables; skips existing files so runs can be interrupted and resumed

---

### `run_all_hierarchical_samplers.py`

Runs the custom Python samplers across every combination of truth model, prior type, and parameterization for the multi-season hierarchical model.

**Key Components:**
- **18-combination outer loop**: 3 truth models x 3 prior types x 2 parameterizations (centered, noncentered); only uses SPL inference model; tests both parameterizations to address the funnel geometry problem under centered coordinates
- **Same MALA preconditioning strategy** as the base runner
- **Outputs to `Stats/`**: posterior summary and diagnostics CSVs with `_hierarchical_` prefix, in the same format as the base runner

---

### `bridge_sampling.py`

Runs a bridge sampling prodcedure as outlined in Gronau et al. 2017 to estimate the marginal log-likelihood (log Z) for Bayes Factor calculations.  

**Key Components:**
- `estimate_log_ml`: fits a multivariate Normal proposal distribution to posterior samples via second moment matching, draws `n_proposal` samples from it, then runs the iterative bridge equation until convergence; returns `(log_Z, log_Z_se)` where the standard error is estimated by bootstrapping over posterior samples
- `_bridge_iterate`: each iteration of the loop updates log Z using the ratio of importance-weighted expectations over the posterior and proposal distributions (log-sum-exp trick used to prevent underflow)
- `_fit_mvn_proposal`: fits the MVN proposal with a small ridge term to keep the covariance non-singular

Called by `bayes_factor_base.py` and `bayes_factor_hierarchical.py`, and also directly in `Test_comprehensive_sim_study.ipynb` for the full model-selection simulation study.

---

## Dependencies

```
pip install -r requirements.txt
```

Core dependencies: `numpy`, `scipy`, `matplotlib`.

---

## Running the core pipeline

```bash
# Verify the posterior and gradient at true parameters
python posteriors.py

# Verify the simulator
python simulator.py

# Run all base model test cases
python run_all_base_samplers.py

# Run all hierarchical model test cases
python run_all_hierarchical_samplers.py
```
