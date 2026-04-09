"""
Phase 2: MCMC Samplers
=======================
Hand-implemented samplers for the neutrino Poisson mixture model:
  - Random-walk Metropolis-Hastings with adaptive proposals
  - Metropolis-adjusted Langevin algorithm (MALA)
  - Diagnostics: trace plots, R-hat, ESS, autocorrelation

All samplers work in transformed (unconstrained) parameter space.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


# ══════════════════════════════════════════════════════════════════════════
# SAMPLER OUTPUT
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MCMCResult:
    """Container for MCMC output."""
    samples: np.ndarray          # (n_samples, d) posterior samples
    log_posteriors: np.ndarray   # (n_samples,) log-posterior values
    acceptance_rate: float       # overall acceptance rate
    wall_time: float             # total wall-clock time in seconds
    n_accepted: int              # number of accepted proposals
    n_total: int                 # total number of proposals
    param_names: list[str]       # names for each parameter dimension
    sampler_name: str            # identifier for this sampler


# ══════════════════════════════════════════════════════════════════════════
# RANDOM-WALK METROPOLIS-HASTINGS
# ══════════════════════════════════════════════════════════════════════════

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
    Random-walk Metropolis-Hastings with optional adaptive proposals.

    The proposal is:
        theta' = theta + epsilon,  epsilon ~ N(0, proposal_cov)

    During the adaptation phase (first `adapt_until` iterations), the
    proposal covariance is periodically updated to:
        proposal_cov = (2.38^2 / d) * empirical_cov(chain so far) + eps*I

    This is the standard adaptive MH recipe (Haario et al., 2001).

    Parameters
    ----------
    log_posterior_fn : callable
        Function theta -> log p(theta | data). Must return -inf for
        invalid parameters.
    theta_init : (d,) array
        Initial parameter values (in transformed space).
    n_iterations : int
        Total number of MCMC iterations (including burn-in).
    n_burnin : int
        Number of initial samples to discard.
    proposal_cov : (d, d) array, optional
        Initial proposal covariance. If None, uses 0.01*I.
    adapt_proposal : bool
        Whether to adapt the proposal covariance.
    adapt_interval : int
        How often (in iterations) to update the proposal.
    adapt_until : int
        Stop adapting after this many iterations.
    target_accept : float
        Target acceptance rate (used for scale tuning).
    param_names : list of str, optional
        Names for each parameter dimension.
    rng : numpy random Generator
    verbose : bool
        Print progress updates.

    Returns
    -------
    MCMCResult
    """
    if rng is None:
        rng = np.random.default_rng()

    d = len(theta_init)
    if proposal_cov is None:
        proposal_cov = 0.01 * np.eye(d)
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    # Storage
    samples = np.zeros((n_iterations, d))
    log_posts = np.zeros(n_iterations)

    # Initialize
    theta = theta_init.copy()
    lp = log_posterior_fn(theta)
    if not np.isfinite(lp):
        raise ValueError(f"Initial log-posterior is not finite: {lp}. "
                         f"Check theta_init = {theta_init}")

    n_accepted = 0
    scale_factor = 1.0  # multiplicative scale for proposal_cov

    start_time = time.time()

    for i in range(n_iterations):
        # ── Propose ──
        epsilon = rng.multivariate_normal(np.zeros(d), scale_factor * proposal_cov)
        theta_proposed = theta + epsilon

        # ── Evaluate ──
        lp_proposed = log_posterior_fn(theta_proposed)

        # ── Accept / Reject ──
        log_alpha = lp_proposed - lp
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_proposed
            lp = lp_proposed
            n_accepted += 1

        samples[i] = theta
        log_posts[i] = lp

        # ── Adapt proposal ──
        if adapt_proposal and i > 0 and i < adapt_until and i % adapt_interval == 0:
            # Update covariance from chain history
            chain_so_far = samples[:i+1]
            if i >= 2 * d:  # need enough samples for a reasonable covariance
                emp_cov = np.cov(chain_so_far.T)
                proposal_cov = (2.38**2 / d) * emp_cov + 1e-6 * np.eye(d)

            # Tune scale factor based on recent acceptance rate
            recent_start = max(0, i - adapt_interval)
            recent_accepted = np.sum(
                np.any(samples[recent_start+1:i+1] != samples[recent_start:i], axis=1)
            )
            recent_rate = recent_accepted / adapt_interval

            if recent_rate < target_accept - 0.05:
                scale_factor *= 0.8  # shrink proposals
            elif recent_rate > target_accept + 0.05:
                scale_factor *= 1.2  # grow proposals

        # ── Progress ──
        if verbose and (i + 1) % 5000 == 0:
            current_rate = n_accepted / (i + 1)
            elapsed = time.time() - start_time
            print(f"  Iteration {i+1}/{n_iterations}: "
                  f"accept rate = {current_rate:.3f}, "
                  f"scale = {scale_factor:.3f}, "
                  f"elapsed = {elapsed:.1f}s")

    wall_time = time.time() - start_time

    # Discard burn-in
    post_burnin = samples[n_burnin:]
    post_burnin_lp = log_posts[n_burnin:]

    acceptance_rate = n_accepted / n_iterations

    if verbose:
        print(f"  Done. Acceptance rate: {acceptance_rate:.3f}, "
              f"Wall time: {wall_time:.1f}s")

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


# ══════════════════════════════════════════════════════════════════════════
# METROPOLIS-ADJUSTED LANGEVIN ALGORITHM (MALA)
# ══════════════════════════════════════════════════════════════════════════

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
    Metropolis-adjusted Langevin algorithm (MALA).

    The proposal is:
        theta' = theta + (eps/2) * grad_log_p(theta) + sqrt(eps) * z
        z ~ N(0, I)

    The acceptance ratio accounts for the asymmetric proposal:
        log alpha = log p(theta') - log p(theta)
                  + log q(theta | theta') - log q(theta' | theta)

    where q(theta' | theta) = N(theta'; theta + (eps/2)*grad, eps*I).

    Parameters
    ----------
    log_posterior_fn : callable
        Function theta -> log p(theta | data).
    grad_log_posterior_fn : callable
        Function theta -> gradient of log p(theta | data).
    theta_init : (d,) array
        Initial parameter values.
    step_size : float
        MALA step size epsilon.
    adapt_step : bool
        Whether to adapt the step size.
    target_accept : float
        Target acceptance rate for MALA (~0.574 is optimal in theory).

    Returns
    -------
    MCMCResult
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
        raise ValueError(f"Initial log-posterior is not finite: {lp}")

    n_accepted = 0
    eps = step_size

    start_time = time.time()

    for i in range(n_iterations):
        # ── Propose: theta' = theta + (eps/2)*grad + sqrt(eps)*z ──
        z = rng.standard_normal(d)
        theta_proposed = theta + 0.5 * eps * grad + np.sqrt(eps) * z

        # ── Evaluate at proposal ──
        lp_proposed = log_posterior_fn(theta_proposed)

        if np.isfinite(lp_proposed):
            grad_proposed = grad_log_posterior_fn(theta_proposed)

            # ── Log proposal densities (for asymmetric correction) ──
            # q(theta' | theta): N(theta'; theta + (eps/2)*grad, eps*I)
            mean_fwd = theta + 0.5 * eps * grad
            log_q_fwd = -0.5 / eps * np.sum((theta_proposed - mean_fwd)**2)

            # q(theta | theta'): N(theta; theta' + (eps/2)*grad', eps*I)
            mean_rev = theta_proposed + 0.5 * eps * grad_proposed
            log_q_rev = -0.5 / eps * np.sum((theta - mean_rev)**2)

            # ── Accept / Reject ──
            log_alpha = (lp_proposed - lp) + (log_q_rev - log_q_fwd)

            if np.log(rng.uniform()) < log_alpha:
                theta = theta_proposed
                lp = lp_proposed
                grad = grad_proposed
                n_accepted += 1
        # else: proposal gave -inf log-posterior, reject automatically

        samples[i] = theta
        log_posts[i] = lp

        # ── Adapt step size ──
        if adapt_step and i > 0 and i < adapt_until and i % adapt_interval == 0:
            recent_start = max(0, i - adapt_interval)
            recent_accepted = np.sum(
                np.any(samples[recent_start+1:i+1] != samples[recent_start:i], axis=1)
            )
            recent_rate = recent_accepted / adapt_interval

            # Simple multiplicative tuning
            if recent_rate < target_accept - 0.05:
                eps *= 0.8
            elif recent_rate > target_accept + 0.05:
                eps *= 1.2

        # ── Progress ──
        if verbose and (i + 1) % 5000 == 0:
            current_rate = n_accepted / (i + 1)
            elapsed = time.time() - start_time
            print(f"  Iteration {i+1}/{n_iterations}: "
                  f"accept rate = {current_rate:.3f}, "
                  f"step_size = {eps:.6f}, "
                  f"elapsed = {elapsed:.1f}s")

    wall_time = time.time() - start_time

    post_burnin = samples[n_burnin:]
    post_burnin_lp = log_posts[n_burnin:]
    acceptance_rate = n_accepted / n_iterations

    if verbose:
        print(f"  Done. Acceptance rate: {acceptance_rate:.3f}, "
              f"Final step size: {eps:.6f}, "
              f"Wall time: {wall_time:.1f}s")

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


# ══════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS (Phase 2E)
# ══════════════════════════════════════════════════════════════════════════

def effective_sample_size(chain: np.ndarray) -> float:
    """
    Estimate ESS for a 1D chain using the autocorrelation method.
    Truncates at the first negative autocorrelation.

    Parameters
    ----------
    chain : (n,) array of scalar samples

    Returns
    -------
    float : estimated ESS
    """
    n = len(chain)
    if n < 10:
        return float(n)

    # Center the chain
    x = chain - np.mean(chain)
    var = np.var(chain)
    if var < 1e-30:
        return 1.0

    # Compute autocorrelation via FFT (fast)
    fft_x = np.fft.fft(x, n=2*n)
    acf_full = np.fft.ifft(fft_x * np.conj(fft_x)).real[:n] / (var * n)

    # Truncate at first negative autocorrelation
    T = 1
    for k in range(1, n):
        if acf_full[k] < 0:
            break
        T += 1

    tau_int = 1.0 + 2.0 * np.sum(acf_full[1:T])
    tau_int = max(tau_int, 1.0)  # floor at 1

    return n / tau_int


def compute_ess_per_second(result: MCMCResult) -> dict:
    """Compute ESS and ESS/second for each parameter."""
    n_samples, d = result.samples.shape
    ess_dict = {}
    for j in range(d):
        name = result.param_names[j]
        ess = effective_sample_size(result.samples[:, j])
        ess_dict[name] = {
            "ESS": ess,
            "ESS_per_sec": ess / result.wall_time,
        }
    return ess_dict


def split_rhat(chains: list[np.ndarray]) -> float:
    """
    Compute split-R-hat for a scalar parameter across multiple chains.

    Parameters
    ----------
    chains : list of (n,) arrays, one per chain

    Returns
    -------
    float : split-R-hat statistic
    """
    # Split each chain in half
    split_chains = []
    for chain in chains:
        mid = len(chain) // 2
        split_chains.append(chain[:mid])
        split_chains.append(chain[mid:])

    C = len(split_chains)
    m = min(len(c) for c in split_chains)

    # Trim all to same length
    split_chains = [c[:m] for c in split_chains]

    # Within-chain variance
    chain_means = np.array([np.mean(c) for c in split_chains])
    chain_vars = np.array([np.var(c, ddof=1) for c in split_chains])
    W = np.mean(chain_vars)

    # Between-chain variance
    grand_mean = np.mean(chain_means)
    B = m * np.var(chain_means, ddof=1)

    # Pooled variance estimate
    V_hat = (m - 1) / m * W + B / m

    if W < 1e-30:
        return float('nan')

    return np.sqrt(V_hat / W)


def autocorrelation(chain: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Compute autocorrelation function for a 1D chain.

    Returns
    -------
    (max_lag+1,) array of autocorrelations at lags 0, 1, ..., max_lag
    """
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


def print_diagnostics(results: list[MCMCResult], param_index: int = 0):
    """
    Print a comparison table of diagnostics across multiple sampler results.
    """
    print(f"\n{'Sampler':<12} {'Accept%':>8} {'Time(s)':>8}", end="")
    d = results[0].samples.shape[1]
    for j in range(d):
        name = results[0].param_names[j]
        print(f"  ESS({name})", end="")
    print()
    print("-" * (30 + 12 * d))

    for result in results:
        print(f"{result.sampler_name:<12} {result.acceptance_rate:>7.3f} "
              f"{result.wall_time:>8.1f}", end="")
        for j in range(d):
            ess = effective_sample_size(result.samples[:, j])
            print(f"  {ess:>9.0f}", end="")
        print()

    # ESS per second
    print(f"\n{'Sampler':<12} {'Accept%':>8} {'Time(s)':>8}", end="")
    for j in range(d):
        name = results[0].param_names[j]
        print(f"  ESS/s({name})", end="")
    print()
    print("-" * (30 + 14 * d))

    for result in results:
        print(f"{result.sampler_name:<12} {result.acceptance_rate:>7.3f} "
              f"{result.wall_time:>8.1f}", end="")
        for j in range(d):
            ess = effective_sample_size(result.samples[:, j])
            ess_per_s = ess / result.wall_time
            print(f"  {ess_per_s:>11.1f}", end="")
        print()


# ══════════════════════════════════════════════════════════════════════════
# TEST: Run both samplers on the base model
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from simulator import generate_single_season
    from posteriors import log_posterior_base, grad_log_posterior_base
    import functools

    rng = np.random.default_rng(42)

    # Generate data
    print("Generating data...")
    ds = generate_single_season(rng=rng)
    print(f"  Total counts: {ds.counts.sum()}, "
          f"Signal: {ds.mu_signal.sum():.0f}, "
          f"Background: {ds.mu_background.sum():.0f}")

    # Parameter names
    param_names = ["log_phi", "gamma", "log_eta", "delta"]

    # True parameters in transformed space
    theta_true = np.array([
        np.log(ds.true_params["phi"]),      # log_phi
        ds.true_params["gamma"],            # gamma
        np.log(ds.true_params["eta"]),      # log_eta
        ds.true_params["delta"],            # delta
    ])
    print(f"  True params (transformed): {theta_true}")

    # Create log-posterior and gradient functions with data baked in
    def log_post(theta):
        return log_posterior_base(theta, ds.counts, ds.E_centers,
                                  ds.E_widths, "weakly")

    def grad_log_post(theta):
        return grad_log_posterior_base(theta, ds.counts, ds.E_centers,
                                       ds.E_widths, "weakly")

    # Start near (but not at) the true values
    theta_init = theta_true + rng.normal(0, 0.1, size=4)

    # ── Run RWMH ──
    print("\n" + "=" * 60)
    print("Running Random-Walk Metropolis-Hastings...")
    print("=" * 60)

    result_rwmh = run_rwmh(
        log_posterior_fn=log_post,
        theta_init=theta_init,
        n_iterations=20000,
        n_burnin=5000,
        adapt_proposal=True,
        param_names=param_names,
        rng=rng,
    )

    # ── Run MALA ──
    print("\n" + "=" * 60)
    print("Running MALA...")
    print("=" * 60)

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

    # ── Compare ──
    print("\n" + "=" * 60)
    print("SAMPLER COMPARISON")
    print("=" * 60)
    print_diagnostics([result_rwmh, result_mala])

    # ── Posterior summary ──
    print("\n" + "=" * 60)
    print("POSTERIOR SUMMARY (vs true values)")
    print("=" * 60)

    for j, name in enumerate(param_names):
        chain_rwmh = result_rwmh.samples[:, j]
        chain_mala = result_mala.samples[:, j]

        # Convert back to natural scale for phi and eta
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

        print(f"\n  {display_name}:")
        print(f"    True:  {true_val:.4f}")
        print(f"    RWMH:  {np.mean(chain_rwmh_display):.4f} "
              f"[{np.percentile(chain_rwmh_display, 5):.4f}, "
              f"{np.percentile(chain_rwmh_display, 95):.4f}]")
        print(f"    MALA:  {np.mean(chain_mala_display):.4f} "
              f"[{np.percentile(chain_mala_display, 5):.4f}, "
              f"{np.percentile(chain_mala_display, 95):.4f}]")