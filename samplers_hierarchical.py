"""
MCMC Samplers
Pls check correctness
Here we implement RWMH, MALA. And provide trace plots, Rhat, ESS, Autocorrelation
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional
import matplotlib.pyplot as plt


@dataclass
class MCMCResult:
    """container for MCMC output"""
    samples: np.ndarray
    log_posteriors: np.ndarray
    acceptance_rate: float
    wall_time: float
    n_accepted: int
    n_total: int
    param_names: list[str]
    sampler_name: str


# RWMH

def run_rwmh(
    log_posterior_fn: Callable,
    theta_init: np.ndarray,
    n_iterations: int = 20000,
    n_burnin: int = 5000,
    proposal_cov: Optional[np.ndarray] = None,
    adapt_proposal: bool = True,
    adapt_interval: int = 200,
    adapt_until: int = 5000,
    target_accept: float = 0.234,
    param_names: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> MCMCResult:
    """
    RWMH, as well as optional adaptive proposals
    The proposal is theta' = theta + epsilon, epsilon = N(0, proposal_cov)
    During adaptation phase (which is first 'adapt_until' iters), the proposal cov
        is periodically updated to,

        proposal_cov = (2.38^2 / d) * empirical_cov(chain so far) + eps*I

    We took this from Haario et al., 2001
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(theta_init)
    if proposal_cov is None:
        proposal_cov = 0.01 * np.eye(d)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    samples = np.zeros((n_iterations, d))
    log_posts = np.zeros(n_iterations)

    # init
    theta = theta_init.copy()
    lp = log_posterior_fn(theta)
    if not np.isfinite(lp):
        raise ValueError(f"Initial log posterior is not finite")

    n_accepted = 0
    # for proposal cov
    scale_factor = 1.0

    start_time = time.time()

    for i in range(n_iterations):
        # propose
        epsilon = rng.multivariate_normal(np.zeros(d), scale_factor * proposal_cov)
        theta_proposed = theta + epsilon

        # evaluate
        lp_proposed = log_posterior_fn(theta_proposed)

        # accept/reject
        log_alpha = lp_proposed - lp
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_proposed
            lp = lp_proposed
            n_accepted += 1

        samples[i] = theta
        log_posts[i] = lp

        # adapt proposal
        if adapt_proposal and i > 0 and i < adapt_until and i % adapt_interval == 0:
            # update covariance from chain history
            chain_so_far = samples[:i+1]
            if i >= 2 * d:
                emp_cov = np.cov(chain_so_far.T)
                proposal_cov = (2.38**2 / d) * emp_cov + 1e-6 * np.eye(d)

            # tune scale factor based on the recent acceptance rate
            recent_start = max(0, i - adapt_interval)
            recent_accepted = np.sum(np.any(samples[recent_start+1:i+1] != samples[recent_start:i], axis=1))
            recent_rate = recent_accepted / adapt_interval

            if recent_rate < target_accept - 0.05:
                scale_factor *= 0.8
            elif recent_rate > target_accept + 0.05:
                scale_factor *= 1.2

        # progress
        if verbose and (i + 1) % 5000 == 0:
            current_rate = n_accepted / (i + 1)
            elapsed = time.time() - start_time
            print(f"Iteration {i+1}/{n_iterations}: "
                  f"accept rate = {current_rate:.3f}, "
                  f"scale = {scale_factor:.3f}, "
                  f"elapsed = {elapsed:.1f}s")

    wall_time = time.time() - start_time

    post_burnin = samples[n_burnin:]
    post_burnin_lp = log_posts[n_burnin:]

    acceptance_rate = n_accepted / n_iterations

    if verbose:
        print(f"Done. Acceptance rate {acceptance_rate:.3f}, "
              f"Wall time {wall_time:.1f}s")

    return MCMCResult(
        samples=post_burnin,
        log_posteriors=post_burnin_lp,
        acceptance_rate=acceptance_rate,
        wall_time=wall_time,
        n_accepted=n_accepted,
        n_total=n_iterations,
        param_names=param_names,
        sampler_name="RWMH",
    )


# MALA
def run_mala(
    log_posterior_fn: Callable,
    grad_log_posterior_fn: Callable,
    theta_init: np.ndarray,
    n_iterations: int = 20000,
    n_burnin: int = 5000,
    step_size: float = 0.001,
    adapt_step: bool = True,
    adapt_interval: int = 200,
    adapt_until: int = 5000,
    target_accept: float = 0.60,
    param_names: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
    precond: Optional[np.ndarray] = None,
    adapt_precond: bool = False,
    precond_type: str = "diag",          # "diag" for a diagonal preconditioning matrix or "dense"
    precond_start: Optional[int] = None, # when to begin covariance adaptation
    precond_shrinkage: float = 0.1,      # only used for dense
    precond_ridge: float = 1e-6,
    normalize_precond: bool = True,
) -> MCMCResult:
    """
    Uses a preconditioning matrix M to improve the geometry of MALA gradients

    Adaptation strategy:
    - M can be diagonal ("diag" mode) or a full matrix ("dense")
    - eps is adapted using recent acceptance rate (analagous to approach in RWMH function)
    - M can be optionally adapted from empirical covariance of chain history (adapt_precond option)
      during warmup only, then frozen after adapt_until
    - Alternatively, can specify a preconditioning matrix and forego adaptive tuning (precond option)
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(theta_init)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    if precond_start is None:
        precond_start = max(2 * d, adapt_interval)

    samples = np.zeros((n_iterations, d))
    log_posts = np.zeros(n_iterations)

    theta = theta_init.copy()
    lp = log_posterior_fn(theta)
    grad = grad_log_posterior_fn(theta)

    if not np.isfinite(lp):
        raise ValueError("Initial log posterior is not finite")

    # Initialize preconditioner
    if precond is None:
        if precond_type == "diag":
            precond = np.ones(d)
        elif precond_type == "dense":
            precond = np.eye(d)
        else:
            raise ValueError("precond_type must be 'diag' or 'dense'")
    else:
        precond = np.asarray(precond, dtype=float)
        if precond_type == "diag":
            if precond.shape != (d,):
                raise ValueError(f"diag precond must have shape ({d},), got {precond.shape}")
            if np.any(precond <= 0):
                raise ValueError("All entries of diag precond must be positive")
        elif precond_type == "dense":
            if precond.shape != (d, d):
                raise ValueError(f"dense precond must have shape ({d}, {d}), got {precond.shape}")
            precond = 0.5 * (precond + precond.T)
            try:
                np.linalg.cholesky(precond)
            except np.linalg.LinAlgError:
                raise ValueError("dense precond must be positive definite")
        else:
            raise ValueError("precond_type must be 'diag' or 'dense'")

    def _normalize_diag(v: np.ndarray) -> np.ndarray:
        if not normalize_precond:
            return v
        return v / np.exp(np.mean(np.log(v)))

    def _normalize_dense(M: np.ndarray) -> np.ndarray:
        if not normalize_precond:
            return M
        sign, logdet = np.linalg.slogdet(M)
        if sign <= 0:
            raise ValueError("Cannot normalize dense preconditioner with non-positive determinant")
        return M / np.exp(logdet / d)

    if precond_type == "diag":
        precond = _normalize_diag(np.asarray(precond, dtype=float))
        chol_precond = np.sqrt(precond)
    else:
        precond = _normalize_dense(np.asarray(precond, dtype=float))
        chol_precond = np.linalg.cholesky(precond)

    n_accepted = 0
    eps = step_size

    start_time = time.time()

    for i in range(n_iterations):
        # propose
        z = rng.standard_normal(d)

        if precond_type == "diag":
            mean_fwd = theta + 0.5 * eps * (precond * grad)
            theta_proposed = mean_fwd + np.sqrt(eps) * (chol_precond * z)
        else:
            mean_fwd = theta + 0.5 * eps * (precond @ grad)
            theta_proposed = mean_fwd + np.sqrt(eps) * (chol_precond @ z)

        # evaluate at proposal
        lp_proposed = log_posterior_fn(theta_proposed)

        if np.isfinite(lp_proposed):
            grad_proposed = grad_log_posterior_fn(theta_proposed)

            if precond_type == "diag":
                diff_fwd = theta_proposed - mean_fwd
                log_q_fwd = -0.5 * np.sum((diff_fwd ** 2) / (eps * precond))

                mean_rev = theta_proposed + 0.5 * eps * (precond * grad_proposed)
                diff_rev = theta - mean_rev
                log_q_rev = -0.5 * np.sum((diff_rev ** 2) / (eps * precond))
            else:
                diff_fwd = theta_proposed - mean_fwd
                solve_fwd = np.linalg.solve(chol_precond, diff_fwd)
                log_q_fwd = -0.5 / eps * np.dot(solve_fwd, solve_fwd)

                mean_rev = theta_proposed + 0.5 * eps * (precond @ grad_proposed)
                diff_rev = theta - mean_rev
                solve_rev = np.linalg.solve(chol_precond, diff_rev)
                log_q_rev = -0.5 / eps * np.dot(solve_rev, solve_rev)

            log_alpha = (lp_proposed - lp) + (log_q_rev - log_q_fwd)

            if np.log(rng.uniform()) < log_alpha:
                theta = theta_proposed
                lp = lp_proposed
                grad = grad_proposed
                n_accepted += 1

        samples[i] = theta
        log_posts[i] = lp

        # adapt step size
        if adapt_step and i > 0 and i < adapt_until and i % adapt_interval == 0:
            recent_start = max(0, i - adapt_interval)
            recent_accepted = np.sum(
                np.any(samples[recent_start+1:i+1] != samples[recent_start:i], axis=1)
            )
            recent_rate = recent_accepted / adapt_interval

            if recent_rate < target_accept - 0.05:
                eps *= 0.8
            elif recent_rate > target_accept + 0.05:
                eps *= 1.2

        # adapt preconditioner
        if adapt_precond and i > 0 and i < adapt_until and i % adapt_interval == 0:
            if i >= precond_start:
                chain_so_far = samples[:i+1]
                emp_cov = np.cov(chain_so_far.T, ddof=1)
                emp_cov = 0.5 * (emp_cov + emp_cov.T)

                if precond_type == "diag":
                    new_precond = np.diag(emp_cov)
                    new_precond = np.clip(new_precond, precond_ridge, np.inf)
                    new_precond = new_precond + precond_ridge
                    new_precond = _normalize_diag(new_precond)

                    precond = new_precond
                    chol_precond = np.sqrt(precond)

                else:
                    diag_cov = np.diag(np.diag(emp_cov))
                    new_precond = (1.0 - precond_shrinkage) * emp_cov + precond_shrinkage * diag_cov
                    new_precond = new_precond + precond_ridge * np.eye(d)
                    new_precond = 0.5 * (new_precond + new_precond.T)

                    try:
                        new_precond = _normalize_dense(new_precond)
                        new_chol = np.linalg.cholesky(new_precond)
                        precond = new_precond
                        chol_precond = new_chol
                    except np.linalg.LinAlgError:
                        # Keep old preconditioner if update is not PD
                        pass

        # progress
        if verbose and (i + 1) % 5000 == 0:
            current_rate = n_accepted / (i + 1)
            elapsed = time.time() - start_time
            print(
                f"Iteration {i+1}/{n_iterations}: "
                f"accept rate = {current_rate:.3f}, "
                f"step_size = {eps:.6g}, "
                f"elapsed = {elapsed:.1f}s"
            )

    wall_time = time.time() - start_time

    post_burnin = samples[n_burnin:]
    post_burnin_lp = log_posts[n_burnin:]
    acceptance_rate = n_accepted / n_iterations

    if verbose:
        print(
            f"Done. Acceptance rate, {acceptance_rate:.3f}, "
            f"Final step size, {eps:.6g}, "
            f"Wall time, {wall_time:.1f}s"
        )

    return MCMCResult(
        samples=post_burnin,
        log_posteriors=post_burnin_lp,
        acceptance_rate=acceptance_rate,
        wall_time=wall_time,
        n_accepted=n_accepted,
        n_total=n_iterations,
        param_names=param_names,
        sampler_name="MALA",
    )


# estimates preconditioning matrix for MALA from RWMH empirical covariance
def estimate_dense_precond_from_rwmh(results: list[MCMCResult], ridge: float = 1e-6,use_combined: bool = True,) -> np.ndarray:
    """
    fidge: small diagonal ridge for numerical stability
    use_combined: if true, pools all draws across chains; else, averages covariances across chains
    """
    d = results[0].samples.shape[1]

    if use_combined:
        combined = np.concatenate([r.samples for r in results], axis=0)
        cov = np.cov(combined.T, ddof=1)
    else:
        covs = [np.cov(r.samples.T, ddof=1) for r in results]
        cov = np.mean(covs, axis=0)

    cov = 0.5 * (cov + cov.T)  # force symmetry
    cov = cov + ridge * np.eye(d)

    return cov
    
# wrapper for running multiple parallel chains from dispersed starting conditions
def run_multiple_chains(sampler_fn, theta_init: np.ndarray, n_chains: int = 4, init_strategy: str = "jitter", init_scale: float = 0.5, rng: Optional[np.random.Generator] = None, **sampler_kwargs) -> list[MCMCResult]:
    if rng is None:
        rng = np.random.default_rng()

    results = []

    for chain_id in range(n_chains):
        if init_strategy == "same":
            theta0 = theta_init.copy()
        elif init_strategy == "jitter":
            theta0 = theta_init + rng.normal(0.0, init_scale, size=len(theta_init))
        else:
            raise ValueError(f"Unknown init_strategy: {init_strategy}")

        # independent RNG per chain
        chain_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        result = sampler_fn(
            theta_init=theta0,
            rng=chain_rng,
            verbose=True,
            **sampler_kwargs,
        )
        results.append(result)

    return results

# NEW MULTI-CHAIN DIAGNOSTICS

def _stack_chains(results: list[MCMCResult]) -> np.ndarray:
    """
    Stack samples from multiple chains into shape (n_chains, n_samples, d)
    """
    if len(results) == 0:
        raise ValueError("results must be non-empty")

    lengths = [r.samples.shape[0] for r in results]
    m = min(lengths)
    d = results[0].samples.shape[1]

    stacked = np.zeros((len(results), m, d))
    for c, r in enumerate(results):
        if r.samples.shape[1] != d:
            raise ValueError("All chains must have the same number of parameters")
        stacked[c] = r.samples[:m]

    return stacked


def _combined_samples(results: list[MCMCResult]) -> np.ndarray:
    """
    Combine multiple chains into one array of shape (n_chains * n_samples, d).
    """
    stacked = _stack_chains(results)
    n_chains, n_samples, d = stacked.shape
    return stacked.reshape(n_chains * n_samples, d)

# computes R-hat across several parallel chains
def split_rhat(chains: np.ndarray) -> float:
    n_chains, n_samples = chains.shape
    if n_chains < 2 or n_samples < 4:
        return float("nan")

    half = n_samples // 2
    if half < 2:
        return float("nan")

    # split each chain in half
    split = np.concatenate([chains[:, :half], chains[:, half:2*half]], axis=0)
    C, m = split.shape

    chain_means = np.mean(split, axis=1)
    chain_vars = np.var(split, axis=1, ddof=1)

    W = np.mean(chain_vars)
    if W < 1e-30:
        return float("nan")

    B = m * np.var(chain_means, ddof=1)
    var_hat = (m - 1) / m * W + B / m

    return float(np.sqrt(var_hat / W))

def autocorrelation(chain: np.ndarray, max_lag: int = 1000) -> np.ndarray:
    n = len(chain)
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30:
        return np.zeros(min(max_lag + 1, n))

    max_lag = min(max_lag, n - 1)
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.mean(x[:n-k] * x[k:]) / var
    return acf


def effective_sample_size_multi(chains: np.ndarray, max_lag: Optional[int] = None) -> float:
    n_chains, n_samples = chains.shape
    if n_samples < 10:
        return float(n_chains * n_samples)

    if max_lag is None:
        max_lag = min(1000, n_samples - 1)

    # average autocorrelation across chains
    acfs = np.array([autocorrelation(chains[c], max_lag=max_lag) for c in range(n_chains)])
    mean_acf = np.mean(acfs, axis=0)

    # initial positive sequence
    tau = 1.0
    for k in range(1, len(mean_acf)):
        if mean_acf[k] < 0:
            break
        tau += 2.0 * mean_acf[k]

    tau = max(tau, 1.0)
    return float(n_chains * n_samples / tau)

def compute_ess_per_second_multi(results: list[MCMCResult]) -> dict[str, dict[str, float]]:
    stacked = _stack_chains(results)
    total_time = sum(r.wall_time for r in results)
    param_names = results[0].param_names

    ess_dict = {}
    for j, name in enumerate(param_names):
        ess = effective_sample_size_multi(stacked[:, :, j])
        ess_dict[name] = {
            "ESS": ess,
            "ESS_per_sec": ess / total_time,
        }
    return ess_dict

# function to transform MCMC parameters into interpretable scales (for diagnostics + plotting)
def transform_hierarchical_samples(stacked: np.ndarray, parameterization: str, K: int, latent_display: str = "log_phi") -> tuple[np.ndarray, list[str]]:
    """
    parameterization: "centered" or "noncentered"
    K: number of seasons
    latent_display: how to display phi parameters
        - "raw": show raw sampled coords
            centered -> log_phi_k
            noncentered -> z_k
        - "log_phi": always show log_phi_k
        - "phi": always show phi_k
    """
    n_chains, n_samples, d = stacked.shape
    if d not in (K + 5, K + 6):
        raise ValueError(f"Unexpected hierarchical dimension d={d} for K={K}")
    has_prompt = d == K + 6
    out = stacked.copy()

    if parameterization == "centered":
        # theta = [log_phi_1,...,log_phi_K,gamma,log_eta,delta,mu_phi,log_sigma_phi]
        log_phi = stacked[:, :, :K]

        if latent_display == "raw" or latent_display == "log_phi":
            out[:, :, :K] = log_phi
            latent_names = [f"log_phi_{k+1}" for k in range(K)]
        elif latent_display == "phi":
            out[:, :, :K] = np.exp(log_phi)
            latent_names = [f"phi_{k+1}" for k in range(K)]
        else:
            raise ValueError(f"Unknown latent_display: {latent_display}")

    elif parameterization == "noncentered":
        # theta = [z_1,...,z_K,gamma,log_eta,delta,mu_phi,log_sigma_phi]
        z = stacked[:, :, :K]
        mu_idx = K + 4 if has_prompt else K + 3
        lsig_idx = K + 5 if has_prompt else K + 4
        mu_phi = stacked[:, :, mu_idx]
        log_sigma_phi = stacked[:, :, lsig_idx]
        sigma_phi = np.exp(log_sigma_phi)

        if latent_display == "raw":
            out[:, :, :K] = z
            latent_names = [f"z_{k+1}" for k in range(K)]
        elif latent_display == "log_phi":
            log_phi = mu_phi[:, :, None] + sigma_phi[:, :, None] * z
            out[:, :, :K] = log_phi
            latent_names = [f"log_phi_{k+1}" for k in range(K)]
        elif latent_display == "phi":
            log_phi = mu_phi[:, :, None] + sigma_phi[:, :, None] * z
            out[:, :, :K] = np.exp(log_phi)
            latent_names = [f"phi_{k+1}" for k in range(K)]
        else:
            raise ValueError(f"Unknown latent_display: {latent_display}")

    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    # shared parameters
    shared_names = ["gamma", "log_eta", "delta"]
    if has_prompt:
        shared_names.append("log_eta_prompt")
    shared_names.extend(["mu_phi", "log_sigma_phi"])
    names = latent_names + shared_names

    return out, names

# Outputs summary statistics for posterior estimates
def posterior_summary_multi(results: list[MCMCResult], ci: float = 0.90, true_values: Optional[dict[str, float]] = None) -> dict[str, dict[str, float]]:
    combined = _combined_samples(results)
    param_names = results[0].param_names

    alpha = 1.0 - ci
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)

    out = {}
    for j, name in enumerate(param_names):
        x = combined[:, j]
    
        # transform to natural scale
        if name == "log_phi":
            x = np.exp(x)
            display_name = "phi"
        elif name == "log_eta":
            x = np.exp(x)
            display_name = "eta"
        else:
            display_name = name
    
        out[name] = {
            "display_name": display_name,
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)),
            "median": float(np.median(x)),
            "ci_lower": float(np.percentile(x, lo)),
            "ci_upper": float(np.percentile(x, hi)),
            "true": None if true_values is None else true_values.get(display_name, None),
        }
    return out

# prints output tables
def print_diagnostics_multi_hierarchical(results_by_sampler: dict[str, list[MCMCResult]], parameterization: str, K: int, latent_display: str = "log_phi", true_values: dict[str, float] | None = None, ci: float = 0.90) -> None:
    """
    latent_display: how to display phi parameters
        - "raw": show raw sampled coords
            centered -> log_phi_k
            noncentered -> z_k
        - "log_phi": always show log_phi_k
        - "phi": always show phi_k
    """
    first_key = next(iter(results_by_sampler))
    first_results = results_by_sampler[first_key]

    stacked0 = _stack_chains(first_results)
    transformed0, param_names = transform_hierarchical_samples(
        stacked0, parameterization=parameterization, K=K, latent_display=latent_display
    )
    d = transformed0.shape[2]

    # ESS table
    print(f"\n{'Sampler':<12} {'Chains':>6} {'Accept%':>8} {'Time(s)':>8}", end="")
    for name in param_names:
        print(f"ESS({name})", end="")
    print()
    print("-" * (36 + 12 * d))

    for sampler_name, results in results_by_sampler.items():
        stacked = _stack_chains(results)
        transformed, _ = transform_hierarchical_samples(
            stacked, parameterization=parameterization, K=K, latent_display=latent_display
        )

        mean_accept = np.mean([r.acceptance_rate for r in results])
        total_time = np.sum([r.wall_time for r in results])

        print(f"{sampler_name:<12} {len(results):>6} {mean_accept:>8.3f} {total_time:>8.1f}", end="")
        for j in range(d):
            ess = effective_sample_size_multi(transformed[:, :, j])
            print(f"  {ess:>9.0f}", end="")
        print()

    # Rhat table
    print(f"\n{'Sampler':<12} {'Chains':>6} {'Accept%':>8} {'Time(s)':>8}", end="")
    for name in param_names:
        print(f"Rhat({name})", end="")
    print()
    print("-" * (36 + 12 * d))

    for sampler_name, results in results_by_sampler.items():
        stacked = _stack_chains(results)
        transformed, _ = transform_hierarchical_samples(
            stacked, parameterization=parameterization, K=K, latent_display=latent_display
        )

        mean_accept = np.mean([r.acceptance_rate for r in results])
        total_time = np.sum([r.wall_time for r in results])

        print(f"{sampler_name:<12} {len(results):>6} {mean_accept:>8.3f} {total_time:>8.1f}", end="")
        for j in range(d):
            rhat = split_rhat(transformed[:, :, j])
            print(f"  {rhat:>9.3f}", end="")
        print()

    # posterior summaries
    alpha = 1.0 - ci
    lo = 100.0 * (alpha / 2.0)
    hi = 100.0 * (1.0 - alpha / 2.0)

    for sampler_name, results in results_by_sampler.items():
        stacked = _stack_chains(results)
        transformed, names = transform_hierarchical_samples(
            stacked, parameterization=parameterization, K=K, latent_display=latent_display
        )

        combined = transformed.reshape(-1, d)

        print(f"\nPosterior summary for {sampler_name} ({int(ci*100)}% CI)")
        print(f"{'Param':<16} {'True':>12} {'Mean':>12} {'SD':>12} {'Median':>12} {'CI low':>12} {'CI high':>12}")
        print("-" * 92)

        for j, name in enumerate(names):
            x = combined[:, j]
            true_val = None if true_values is None else true_values.get(name, None)
            true_str = "NA" if true_val is None else f"{true_val:.4f}"

            print(
                f"{name:<16} "
                f"{true_str:>12} "
                f"{np.mean(x):>12.4f} "
                f"{np.std(x, ddof=1):>12.4f} "
                f"{np.median(x):>12.4f} "
                f"{np.percentile(x, lo):>12.4f} "
                f"{np.percentile(x, hi):>12.4f}"
            )

# saves traceplots for each sampler, showing traces for each chain for each parameter
def save_traceplots_multi_hierarchical(results: list[MCMCResult],filename: str,parameterization: str, K: int, latent_display: str = "log_phi") -> None:
    """
    latent_display: how to display phi parameters
        - "raw": show raw sampled coords
            centered -> log_phi_k
            noncentered -> z_k
        - "log_phi": always show log_phi_k
        - "phi": always show phi_k
    """
    stacked = _stack_chains(results)
    transformed, param_names = transform_hierarchical_samples(
        stacked, parameterization=parameterization, K=K, latent_display=latent_display
    )

    n_chains, n_samples, d = transformed.shape
    sampler_name = results[0].sampler_name

    fig, axes = plt.subplots(d, 1, figsize=(10, 2.4 * d), sharex=True)
    if d == 1:
        axes = [axes]

    x = np.arange(n_samples)

    for j, ax in enumerate(axes):
        for c in range(n_chains):
            ax.plot(x, transformed[c, :, j], linewidth=0.6, alpha=0.8)
        ax.set_ylabel(param_names[j])
        ax.set_title(f"{sampler_name} trace: {param_names[j]}")

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

# test
if __name__ == "__main__":
    from simulator import generate_hierarchical
    from posteriors import log_posterior_hierarchical, grad_log_posterior_hierarchical
    from load_config import load_config

    config = load_config()
    test_cfg = config["testing"]
    hier_cfg = config["hierarchical"]
    atmo_cfg = config["atmospheric_realism"]
    det_cfg = config["detector_response"]
    phys_prior_cfg = config["priors"]["physics"]
    rwmh_cfg = config["samplers_hierarchical"]["RWMH"]
    mala_cfg = config["samplers_hierarchical"]["MALA"]
    parameterization = test_cfg["hierarchical_parameterization"]

    rng = np.random.default_rng(int(test_cfg["seed_hierarchical_samplers"]))

    print("Generating data")
    ds = generate_hierarchical(
        K=int(hier_cfg["K"]),
        mu_phi=float(hier_cfg["mu_phi"]),
        sigma_phi=float(hier_cfg["sigma_phi"]),
        gamma=float(hier_cfg["gamma"]),
        eta=float(hier_cfg["eta"]),
        delta=float(hier_cfg["delta"]),
        truth_model=hier_cfg["truth_model"],
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(config["background"].get("eta_prompt", float(hier_cfg["eta"]) * float(atmo_cfg["prompt_fraction"]))),
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
    print(f"K = {K}, True means = {ds.phi_k}")

    if parameterization == "centered":
        print("CENTERED PARAMETERIZATION")
        param_names = [f"log_phi_{i+1}" for i in range(K)]
        theta_true = np.log(ds.phi_k).tolist()
        latent_truth = {f"log_phi_{k+1}": np.log(ds.phi_k[k]) for k in range(K)}
    elif parameterization == "noncentered":
        print("NONCENTERED PARAMETERIZATION")
        param_names = [f"z_{i+1}" for i in range(K)]
        z_true = (np.log(ds.phi_k) - ds.true_params["mu_phi"]) / ds.true_params["sigma_phi"]
        theta_true = z_true.tolist()
        latent_truth = {f"z_{k+1}": z_true[k] for k in range(K)}
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    infer_prompt_eta = bool(ds.seasons[0].model_options.get("infer_prompt_eta", False))
    param_names.extend(["gamma", "log_eta", "delta"])
    if infer_prompt_eta:
        param_names.append("log_eta_prompt")
    param_names.extend(["mu_phi", "log_sigma_phi"])
    theta_true.extend([
        ds.true_params["gamma"],
        np.log(ds.true_params["eta"]),
        ds.true_params["delta"],
    ])
    if infer_prompt_eta:
        theta_true.append(np.log(ds.seasons[0].true_params["eta_prompt"]))
    theta_true.extend([
        ds.true_params["mu_phi"],
        np.log(ds.true_params["sigma_phi"]),
    ])
    theta_true = np.array(theta_true)
    print(f"True params, (transformed) are {theta_true}")

    true_values = {
        **latent_truth,
        **{f"log_phi_{k+1}": np.log(ds.phi_k[k]) for k in range(K)},
        "gamma": ds.true_params["gamma"],
        "log_eta": np.log(ds.true_params["eta"]),
        "delta": ds.true_params["delta"],
        **({"log_eta_prompt": np.log(ds.seasons[0].true_params["eta_prompt"])} if infer_prompt_eta else {}),
        "mu_phi": ds.true_params["mu_phi"],
        "log_sigma_phi": np.log(ds.true_params["sigma_phi"]),
    }

    prior_type = config["priors"]["default"]

    def log_post(theta):
        return log_posterior_hierarchical(theta, ds.seasons, parameterization, prior_type)

    def grad_log_post(theta):
        return grad_log_posterior_hierarchical(theta, ds.seasons, parameterization, prior_type)

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

    rwmh_cov = estimate_dense_precond_from_rwmh(
        rwmh_results,
        ridge=float(test_cfg["preconditioner_ridge"]),
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

    print_diagnostics_multi_hierarchical(
        {"RWMH": rwmh_results, "MALA": mala_results},
        parameterization=parameterization,
        K=K,
        latent_display=test_cfg["latent_display_diagnostics"],
        true_values=true_values,
    )

    traceplot_rwmh = test_cfg["traceplot_hierarchical_rwmh"]
    traceplot_mala = test_cfg["traceplot_hierarchical_mala"]
    if parameterization == "noncentered":
        traceplot_rwmh = traceplot_rwmh.replace("centered", "noncentered")
        traceplot_mala = traceplot_mala.replace("centered", "noncentered")

    save_traceplots_multi_hierarchical(
        rwmh_results,
        traceplot_rwmh,
        parameterization=parameterization,
        K=K,
        latent_display=test_cfg["latent_display_traceplots"],
    )

    save_traceplots_multi_hierarchical(
        mala_results,
        traceplot_mala,
        parameterization=parameterization,
        K=K,
        latent_display=test_cfg["latent_display_traceplots"],
    )
