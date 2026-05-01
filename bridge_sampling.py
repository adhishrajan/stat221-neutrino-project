"""
Implements the optimal bridge sampling procedure described in Gronau et al. (2017) to estimate the log marginal likelihood, used for computing Bayes Factors in bayes_factor_base.py.
"""

import numpy as np
from scipy.special import logsumexp
import scipy.stats

# Fits a multivariate Normal distribution to posterior samples (used as proposal distribution in bridge sampling)
    ## Simple method of moments estimation: just matches mean + covariance (with some regulatization)
def _fit_mvn_proposal(samples: np.ndarray, ridge: float = 1e-6):
    mu = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov += ridge * np.eye(len(mu))
    return scipy.stats.multivariate_normal(mean=mu, cov=cov, allow_singular=False)


def _bridge_iterate(log_p_post: np.ndarray, log_g_post: np.ndarray, log_p_prop: np.ndarray, log_g_prop: np.ndarray, n_iter: int, tol: float,) -> float:
    """
    Runs one iteration of bridge sampling
    ----------
    log_p_post: (N,)  log p*(theta) for posterior samples
    log_g_post: (N,)  log g(theta) for posterior samples
    log_p_prop: (M,)  log p*(theta) for proposal samples
    log_g_prop: (M,)  log g(theta) for proposal samples
    """
    N = len(log_p_post)
    M = len(log_p_prop)
    log_N = np.log(N)
    log_M = np.log(M)

    # Initial guess of marginal likelihood using importance sampling 
    finite_mask = np.isfinite(log_p_post) & np.isfinite(log_g_post)
    if finite_mask.sum() == 0:
        return float("nan")
    log_Z = logsumexp(log_p_post[finite_mask] - log_g_post[finite_mask]) - np.log(finite_mask.sum())

    for _ in range(n_iter):
        log_denom_post = np.logaddexp(log_N + log_p_post, log_M + log_Z + log_g_post)
        log_denom_prop = np.logaddexp(log_N + log_p_prop, log_M + log_Z + log_g_prop)

        # numerator: importance sampling expectation over p (posterior)
        log_num = logsumexp(log_p_post - log_denom_post) - log_N
        # denominator: importance sampling expectation over g (proposal)
        log_den = logsumexp(log_g_prop - log_denom_prop) - log_M

        log_Z_new = log_num - log_den

        if abs(log_Z_new - log_Z) < tol:
            break
        log_Z = log_Z_new

    return float(log_Z_new)

# Runs bridge sampling algorithm to estimate marginal likelihood
def estimate_log_ml(log_posterior_fn, posterior_samples: np.ndarray, log_posterior_vals: np.ndarray | None = None, n_proposal: int = 5000, n_iter: int = 1000,tol: float = 1e-6,rng=None,n_bootstrap: int = 100,ridge: float = 1e-6,) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng()

    N, d = posterior_samples.shape
    M = n_proposal

    # Approximate multivariate Normal proposal
    proposal = _fit_mvn_proposal(posterior_samples, ridge=ridge)

    # log-posterior density of posterior samples
    if log_posterior_vals is not None:
        log_p_post = np.asarray(log_posterior_vals, dtype=float)
    else:
        log_p_post = np.array([log_posterior_fn(posterior_samples[i]) for i in range(N)])

    # log-proposal density of posterior samples
    log_g_post = proposal.logpdf(posterior_samples)

    # Draw M samples from proposal distribution
    prop_samples = proposal.rvs(size=M, random_state=rng)

    # Calculate densities of proposals
    log_g_prop = proposal.logpdf(prop_samples)
    log_p_prop = np.array([log_posterior_fn(prop_samples[j]) for j in range(M)])

    log_Z = _bridge_iterate(log_p_post, log_g_post, log_p_prop, log_g_prop, n_iter, tol)

    if n_bootstrap == 0:
        return log_Z, float("nan")

    # Bootstrap SE: resample posterior samples, keep proposal fixed
    bs_vals = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, N, size=N)
        lz = _bridge_iterate(
            log_p_post[idx], log_g_post[idx], log_p_prop, log_g_prop, n_iter, tol
        )
        bs_vals.append(lz)
    log_Z_se = float(np.std(bs_vals))

    return log_Z, log_Z_se
