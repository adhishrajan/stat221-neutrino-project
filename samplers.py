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

    print(f"RWMN Proposal Covariance = {proposal_cov}")
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


# wrapper function for running multiple chains
def run_multiple_chains(sampler_fn, theta_init, n_chains=4, init_strategy="jitter", init_scale=0.5,rng=None, **sampler_kwargs):
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

        # Give each chain its own RNG stream
        chain_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))

        result = sampler_fn(
            theta_init=theta0,
            rng=chain_rng,
            verbose=True,
            **sampler_kwargs,
        )

        results.append(result)

    return results


# NEW DIAGNOSTICS


# OLD DIAGNOSTICS

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

    print(f"\n{'Sampler':<12} {'Accept%':>8} {'Time(s)':>8}", end="")
    for j in range(d):
        name = results[0].param_names[j]
        print(f"Rhat({name})", end="")
    print()
    print("-" * (30 + 12 * d))

    for result in results:
        print(f"{result.sampler_name:<12} {result.acceptance_rate:>7.3f} "
              f"{result.wall_time:>8.1f}", end="")
        for j in range(d):
            chain = result.samples[:, j]
            mid = len(chain) // 2

            if mid < 2:
                rhat = float("nan")
            else:
                # split a single chain into two halves and treat them as two chains
                rhat = split_rhat([chain[:mid], chain[mid:]])

            print(f"  {rhat:>9.3f}", end="")
        print()

# saves traceplots for each sampler, showing traces for each chain for each parameter
def save_traceplots(result: MCMCResult, filename: str) -> None:
    n_samples, d = result.samples.shape

    fig, axes = plt.subplots(d, 1, figsize=(10, 2.5 * d), sharex=True)

    if d == 1:
        axes = [axes]

    x = np.arange(n_samples)

    for j, ax in enumerate(axes):
        ax.plot(x, result.samples[:, j], linewidth=0.7, alpha = 0.8)
        ax.set_ylabel(result.param_names[j])
        ax.set_title(f"{result.sampler_name} trace: {result.param_names[j]}")

    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

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


    print("Running MALA using preconditioning with proposal covariance from RWMH!")

    rwmh_cov = np.array([[ 6.03197198e-01,  8.96567747e-02, -2.91649216e-02,  9.60971215e-03],
                         [ 8.96567747e-02,  1.42153684e-01, -7.10077053e-03,  1.23259937e-03],
                         [-2.91649216e-02, -7.10077053e-03,  2.42239884e-03, -8.50658921e-04],
                         [ 9.60971215e-03,  1.23259937e-03, -8.50658921e-04,  3.38134000e-04]])
    
    result_mala = run_mala(
        log_posterior_fn=log_post,
        grad_log_posterior_fn=grad_log_post,
        theta_init=theta_init,
        n_iterations=30000,
        n_burnin=10000,
        step_size=1e-4,
        adapt_step=True,
        adapt_until=10000,
        target_accept=0.57,
        param_names=param_names,
        rng=rng,
        precond=rwmh_cov + 1e-6 * np.eye(4),
        adapt_precond=False,
        precond_type="dense",
        normalize_precond=True,
    )

    print("SAMPLER COMPARISON")
    print_diagnostics([result_rwmh, result_mala])

    # saving traces
    save_traceplots(result_rwmh, "traceplots_rwmh.png")
    save_traceplots(result_mala, "traceplots_mala.png")
    print("\nSaved traceplots to:")
    print("  traceplots_rwmh.png")
    print("  traceplots_mala.png")
    
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