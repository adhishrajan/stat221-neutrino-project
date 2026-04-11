"""
MCMC Samplers
Pls check correctness
Here we implement RWMH, MALA. And provide trace plots, Rhat, ESS, Autocorrelation
"""

import numpy as np
import time
from dataclasses import dataclass
from typing import Callable, Optional


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
    target_accept: float = 0.574,
    param_names: Optional[list[str]] = None,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> MCMCResult:
    """
    MALA proposal is
    theta' = theta + (eps/2) * grad_log_p(theta) + sqrt(eps) * z,   z = N(0, I)
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(theta_init)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    samples = np.zeros((n_iterations, d))
    log_posts = np.zeros(n_iterations)

    theta = theta_init.copy()
    lp = log_posterior_fn(theta)
    grad = grad_log_posterior_fn(theta)

    if not np.isfinite(lp):
        raise ValueError(f"Initial log posterior is not finite")

    n_accepted = 0
    eps = step_size

    start_time = time.time()

    for i in range(n_iterations):
        # propose
        z = rng.standard_normal(d)
        theta_proposed = theta + 0.5 * eps * grad + np.sqrt(eps) * z

        # evaluate at proposal
        lp_proposed = log_posterior_fn(theta_proposed)

        if np.isfinite(lp_proposed):
            grad_proposed = grad_log_posterior_fn(theta_proposed)

            # log proposal densities (for asymmetric correction)
            # q(theta' | theta)= N(theta'; theta + (eps/2)*grad, eps*I)
            mean_fwd = theta + 0.5 * eps * grad
            log_q_fwd = -0.5 / eps * np.sum((theta_proposed - mean_fwd)**2)

            # q(theta | theta')= N(theta; theta' + (eps/2)*grad', eps*I)
            mean_rev = theta_proposed + 0.5 * eps * grad_proposed
            log_q_rev = -0.5 / eps * np.sum((theta - mean_rev)**2)

            # accept/reject
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
            recent_accepted = np.sum(np.any(samples[recent_start+1:i+1] != samples[recent_start:i], axis=1))
            recent_rate = recent_accepted / adapt_interval
            # simple tuning
            if recent_rate < target_accept - 0.05:
                eps *= 0.8
            elif recent_rate > target_accept + 0.05:
                eps *= 1.2

        # progress
        if verbose and (i + 1) % 5000 == 0:
            current_rate = n_accepted / (i + 1)
            elapsed = time.time() - start_time
            print(f"Iteration {i+1}/{n_iterations}: "
                  f"accept rate = {current_rate:.3f}, "
                  f"step_size = {eps:.6f}, "
                  f"elapsed = {elapsed:.1f}s")

    wall_time = time.time() - start_time

    post_burnin = samples[n_burnin:]
    post_burnin_lp = log_posts[n_burnin:]
    acceptance_rate = n_accepted / n_iterations

    if verbose:
        print(f"Done. Acceptance rate, {acceptance_rate:.3f},"
              f"Final step size, {eps:.6f}, "
              f"Wall time, {wall_time:.1f}s")

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


# DIAGNOSTICS

def effective_sample_size(chain) -> float:
    n = len(chain)
    if n < 10:
        return float(n)

    # center chain
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30:
        return 1.0

    # autocorrelation
    fft_x = np.fft.fft(x, n=2*n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (var * n)

    T = 1
    for k in range(1, n):
        if acf_full[k] < 0:
            break
        T += 1

    tau_int = 1.0 + 2.0 * np.sum(acf_full[1:T])
    tau_int = max(tau_int, 1.0)

    return n / tau_int


def compute_ess_per_second(result) -> dict:
    _, d = result.samples.shape
    ess_dict = {}
    for j in range(d):
        name = result.param_names[j]
        ess = effective_sample_size(result.samples[:, j])
        ess_dict[name] = {
            "ESS": ess,
            "ESS_per_sec": ess / result.wall_time,
        }
    return ess_dict


def split_rhat(chains) -> float:
    # split each chain in half
    split_chains = []
    for chain in chains:
        mid = len(chain) // 2
        split_chains.append(chain[:mid])
        split_chains.append(chain[mid:])

    C = len(split_chains)
    m = min(len(c) for c in split_chains)

    # trim all to same length
    split_chains = [c[:m] for c in split_chains]

    # within chain var
    chain_means = np.array([np.mean(c) for c in split_chains])
    chain_vars = np.array([np.var(c, ddof=1) for c in split_chains])
    W = np.mean(chain_vars)

    # between chain var
    grand_mean = np.mean(chain_means)
    B = m * np.var(chain_means, ddof=1)

    # pooled var estimate
    V_hat = (m - 1) / m * W + B / m

    if W < 1e-30:
        return float('nan')

    return np.sqrt(V_hat / W)


def autocorrelation(chain, max_lag = 100) -> np.ndarray:
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


def print_diagnostics(results):
    # we print a comp table across the sampler results
    print(f"\n{'Sampler':<12} {'Accept%':>8} {'Time(s)':>8}", end="")
    d = results[0].samples.shape[1]
    for j in range(d):
        name = results[0].param_names[j]
        print(f"ESS({name})", end="")
    print()
    print("-" * (30 + 12 * d))

    for result in results:
        print(f"{result.sampler_name:<12} {result.acceptance_rate:>7.3f} "
              f"{result.wall_time:>8.1f}", end="")
        for j in range(d):
            ess = effective_sample_size(result.samples[:, j])
            print(f"  {ess:>9.0f}", end="")
        print()


    print(f"\n{'Sampler':<12} {'Accept%':>8} {'Time(s)':>8}", end="")
    for j in range(d):
        name = results[0].param_names[j]
        print(f"ESS/s({name})", end="")
    print()
    print("-" * (30 + 14 * d))

    for result in results:
        print(f"{result.sampler_name:<12} {result.acceptance_rate:>7.3f}"
              f"{result.wall_time:>8.1f}", end="")
        for j in range(d):
            ess = effective_sample_size(result.samples[:, j])
            ess_per_s = ess / result.wall_time
            print(f"{ess_per_s:>11.1f}", end="")
        print()


# test
if __name__ == "__main__":
    from simulator import generate_single_season
    from posteriors import log_posterior_base, grad_log_posterior_base

    rng = np.random.default_rng(42)

    print("Generating data")
    ds = generate_single_season(rng=rng)
    print(f"Total counts: {ds.counts.sum()}, "
          f"Signal: {ds.mu_signal.sum():.0f}, "
          f"Background: {ds.mu_background.sum():.0f}")

    param_names = ["log_phi", "gamma", "log_eta", "delta"]

    theta_true = np.array([
        np.log(ds.true_params["phi"]),
        ds.true_params["gamma"],      
        np.log(ds.true_params["eta"]),
        ds.true_params["delta"],      
    ])
    print(f"True params, (transformed) are {theta_true}")

    def log_post(theta):
        return log_posterior_base(theta, ds.counts, ds.E_centers, ds.E_widths, "weakly")

    def grad_log_post(theta):
        return grad_log_posterior_base(theta, ds.counts, ds.E_centers, ds.E_widths, "weakly")

    # start near true values
    theta_init = theta_true + rng.normal(0, 0.1, size=4)


    print("Running RWMH!")

    result_rwmh = run_rwmh(
        log_posterior_fn=log_post,
        theta_init=theta_init,
        n_iterations=20000,
        n_burnin=5000,
        adapt_proposal=True,
        param_names=param_names,
        rng=rng,
    )


    print("Running MALA!")

    result_mala = run_mala(
        log_posterior_fn=log_post,
        grad_log_posterior_fn=grad_log_post,
        theta_init=theta_init,
        n_iterations=20000,
        n_burnin=5000,
        step_size=0.001,
        adapt_step=True,
        param_names=param_names,
        rng=rng,
    )

    print("SAMPLER COMPARISON")
    print_diagnostics([result_rwmh, result_mala])

    print("POSTERIOR SUMMARY vs true values")

    for j, name in enumerate(param_names):
        chain_rwmh = result_rwmh.samples[:, j]
        chain_mala = result_mala.samples[:, j]

        if name == "log_phi":
            display_name = "phi"
            true_val = ds.true_params["phi"]
            chain_rwmh_display = np.exp(chain_rwmh)
            chain_mala_display = np.exp(chain_mala)
        elif name == "log_eta":
            display_name = "eta"
            true_val = ds.true_params["eta"]
            chain_rwmh_display = np.exp(chain_rwmh)
            chain_mala_display = np.exp(chain_mala)
        else:
            display_name = name
            true_val = ds.true_params[name]
            chain_rwmh_display = chain_rwmh
            chain_mala_display = chain_mala

        print(f"\n{display_name}:")
        print(f"True: {true_val:.4f}")
        print(f"RWMH: {np.mean(chain_rwmh_display):.4f} "
              f"[{np.percentile(chain_rwmh_display, 5):.4f}, "
              f"{np.percentile(chain_rwmh_display, 95):.4f}]")
        print(f"MALA: {np.mean(chain_mala_display):.4f} "
              f"[{np.percentile(chain_mala_display, 5):.4f}, "
              f"{np.percentile(chain_mala_display, 95):.4f}]")