"""
Phase 1B & 1C: Log-posterior and gradient functions
====================================================
Provides log-posterior evaluation and analytic gradients for:
  - Base model: parameters (phi, gamma, eta, delta)
  - Hierarchical model: + (phi_k, mu_phi, sigma_phi)

All functions work in TRANSFORMED parameter space for unconstrained MCMC:
  log_phi = log(phi)        [phi > 0]
  gamma   = gamma           [unconstrained, or bounded by prior]
  log_eta = log(eta)        [eta > 0]
  delta   = delta           [unconstrained, prior keeps it near 3.7]

For the hierarchical model:
  log_phi_k = log(phi_k)    [phi_k > 0]
  mu_phi    = mu_phi         [unconstrained]
  log_sigma_phi = log(sigma_phi)  [sigma_phi > 0]
"""

import numpy as np
from scipy.special import gammaln
from simulator import (
    signal_power_law, background_atmospheric, E_REF,
    Dataset, HierarchicalDataset
)


# ══════════════════════════════════════════════════════════════════════════
# BASE MODEL
# ══════════════════════════════════════════════════════════════════════════

def expected_counts_base(E: np.ndarray, E_widths: np.ndarray,
                         phi: float, gamma: float,
                         eta: float, delta: float) -> np.ndarray:
    """Compute expected counts mu_i = signal + background for each bin."""
    mu_sig = signal_power_law(E, phi, gamma, E_widths)
    mu_bg = background_atmospheric(E, eta, delta, E_widths)
    return mu_sig + mu_bg


def poisson_log_likelihood(counts: np.ndarray, mu: np.ndarray) -> float:
    """
    Poisson log-likelihood: sum_i [n_i * log(mu_i) - mu_i - log(n_i!)]
    Drops the constant log(n_i!) term.
    """
    # guard against log(0). if mu_i = 0 and n_i = 0, contribution is 0
    # if mu_i = 0 and n_i > 0, log likelihood is -inf
    with np.errstate(divide='ignore', invalid='ignore'):
        log_mu = np.where(mu > 0, np.log(mu), -np.inf)
    ll = np.sum(counts * log_mu - mu)
    return ll


# ── Priors for base model ─────────────────────────────────────────────────

def log_prior_base(phi: float, gamma: float, eta: float, delta: float,
                   prior_type: str = "flat") -> float:
    """
    Log-prior for (phi, gamma, eta, delta).

    Supports:
      'flat'     : improper flat priors on all (with positivity for phi, eta)
      'weakly'   : log-normal on phi/eta, normal on gamma, normal on delta
      'jeffreys' : Jeffreys-style priors (approximate)

    All versions include:
      delta ~ N(3.7, 0.1^2)  [tight prior centered on atmospheric value]
    """
    # Positivity constraints
    if phi <= 0 or eta <= 0:
        return -np.inf

    lp = 0.0

    # Delta prior is always N(3.7, 0.1^2)
    lp += -0.5 * ((delta - 3.7) / 0.1)**2

    if prior_type == "flat":
        # Flat (improper) on phi, gamma, eta; only delta is informative
        pass

    elif prior_type == "weakly":
        # phi ~ LogNormal(log(30), 1^2)  [broad]
        lp += -0.5 * ((np.log(phi) - np.log(30.0)) / 1.0)**2 - np.log(phi)
        # gamma ~ N(2.5, 0.5^2)
        lp += -0.5 * ((gamma - 2.5) / 0.5)**2
        # eta ~ LogNormal(log(500), 1^2)  [broad]
        lp += -0.5 * ((np.log(eta) - np.log(500.0)) / 1.0)**2 - np.log(eta)

    elif prior_type == "lognormal_gamma":
        # gamma ~ LogNormal(log(2.5), 0.2^2)
        if gamma <= 0:
            return -np.inf
        lp += -0.5 * ((np.log(gamma) - np.log(2.5)) / 0.2)**2 - np.log(gamma)
        # phi, eta flat

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    return lp


# ── Log-posterior in TRANSFORMED space ────────────────────────────────────

def unpack_base_params(theta_transformed: np.ndarray) -> tuple:
    """
    Unpack transformed parameters to natural scale.
    theta_transformed = [log_phi, gamma, log_eta, delta]
    Returns: (phi, gamma, eta, delta)
    """
    log_phi, gamma, log_eta, delta = theta_transformed
    return np.exp(log_phi), gamma, np.exp(log_eta), delta


def log_posterior_base(theta_transformed: np.ndarray,
                      counts: np.ndarray,
                      E: np.ndarray,
                      E_widths: np.ndarray,
                      prior_type: str = "weakly") -> float:
    """
    Log-posterior for the base model in transformed parameter space.

    Parameters
    ----------
    theta_transformed : (4,) array
        [log_phi, gamma, log_eta, delta]
    counts : (B,) array of observed bin counts
    E : (B,) array of bin center energies
    E_widths : (B,) array of bin widths
    prior_type : str
        Prior specification to use.

    Returns
    -------
    float : log p(theta | data) up to a constant
    """
    phi, gamma, eta, delta = unpack_base_params(theta_transformed)

    # Log-prior (on natural scale)
    lp = log_prior_base(phi, gamma, eta, delta, prior_type)
    if not np.isfinite(lp):
        return -np.inf

    # Jacobian for log-transform: always add log(phi) + log(eta)
    # This accounts for the change of variables from (phi, eta) to (log_phi, log_eta)
    lp += theta_transformed[0] + theta_transformed[2]  # = log(phi) + log(eta)

    # Log-likelihood
    mu = expected_counts_base(E, E_widths, phi, gamma, eta, delta)
    if np.any(mu <= 0):
        return -np.inf
    lp += poisson_log_likelihood(counts, mu)

    return lp


# ── Analytic gradient in TRANSFORMED space ────────────────────────────────

def grad_log_posterior_base(theta_transformed: np.ndarray,
                            counts: np.ndarray,
                            E: np.ndarray,
                            E_widths: np.ndarray,
                            prior_type: str = "weakly") -> np.ndarray:
    """
    Gradient of log-posterior w.r.t. transformed parameters.

    Returns
    -------
    (4,) array : [d/d(log_phi), d/d(gamma), d/d(log_eta), d/d(delta)]
    """
    log_phi, gamma_val, log_eta, delta_val = theta_transformed
    phi = np.exp(log_phi)
    eta = np.exp(log_eta)

    E_norm = E / E_REF
    dE_norm = E_widths / E_REF

    # Expected counts and their components
    mu_sig = phi * E_norm**(-gamma_val) * dE_norm
    mu_bg = eta * E_norm**(-delta_val) * dE_norm
    mu = mu_sig + mu_bg

    if np.any(mu <= 0):
        return np.zeros(4)

    # Ratio: (n_i / mu_i) - 1
    ratio = counts / mu - 1.0  # (B,)

    # ── Gradient of log-likelihood ──

    # d/d(log_phi) = d/d(phi) * phi  [chain rule for log-transform]
    # d(log L)/d(phi) = sum_i ratio_i * E_norm_i^{-gamma} * dE_norm_i
    dL_dphi = np.sum(ratio * E_norm**(-gamma_val) * dE_norm)
    dL_dlog_phi = dL_dphi * phi

    # d/d(gamma)
    # d(mu_sig)/d(gamma) = phi * E_norm^{-gamma} * (-log E_norm) * dE_norm
    #                    = mu_sig * (-log E_norm)
    dmu_dgamma = -mu_sig * np.log(E_norm)
    dL_dgamma = np.sum(ratio * dmu_dgamma)

    # d/d(log_eta) = d/d(eta) * eta
    dL_deta = np.sum(ratio * E_norm**(-delta_val) * dE_norm)
    dL_dlog_eta = dL_deta * eta

    # d/d(delta)
    dmu_ddelta = -mu_bg * np.log(E_norm)
    dL_ddelta = np.sum(ratio * dmu_ddelta)

    grad = np.array([dL_dlog_phi, dL_dgamma, dL_dlog_eta, dL_ddelta])

    # ── Gradient of log-prior ──

    # Delta prior: N(3.7, 0.1^2)
    grad[3] += -(delta_val - 3.7) / 0.1**2

    if prior_type == "weakly":
        # phi ~ LogNormal(log(30), 1): d/d(log_phi) of log-prior
        # log p(phi) = -0.5*((log(phi)-log(30))/1)^2 - log(phi)
        # d/d(log_phi) = -(log_phi - log(30))/1 - 1
        grad[0] += -(log_phi - np.log(30.0)) / 1.0**2 - 1.0
        # gamma ~ N(2.5, 0.5^2)
        grad[1] += -(gamma_val - 2.5) / 0.5**2
        # eta ~ LogNormal(log(500), 1)
        grad[2] += -(log_eta - np.log(500.0)) / 1.0**2 - 1.0

    # ── Jacobian gradient ──
    # The log-posterior always includes log(phi) + log(eta) as Jacobian terms.
    # d/d(log_phi) of log(phi) = 1, same for log(eta).
    grad[0] += 1.0
    grad[2] += 1.0

    return grad


# ── Gradient verification utility ────────────────────────────────────────

def verify_gradient(theta: np.ndarray, counts: np.ndarray,
                    E: np.ndarray, E_widths: np.ndarray,
                    prior_type: str = "weakly",
                    eps: float = 1e-5) -> dict:
    """
    Verify analytic gradient against finite differences.

    Returns dict with 'analytic', 'numerical', 'max_abs_diff'.
    """
    analytic = grad_log_posterior_base(theta, counts, E, E_widths, prior_type)

    numerical = np.zeros_like(theta)
    for j in range(len(theta)):
        e_j = np.zeros_like(theta)
        e_j[j] = eps
        f_plus = log_posterior_base(theta + e_j, counts, E, E_widths, prior_type)
        f_minus = log_posterior_base(theta - e_j, counts, E, E_widths, prior_type)
        numerical[j] = (f_plus - f_minus) / (2 * eps)

    return {
        "analytic": analytic,
        "numerical": numerical,
        "abs_diff": np.abs(analytic - numerical),
        "max_abs_diff": np.max(np.abs(analytic - numerical)),
        "rel_diff": np.abs(analytic - numerical) / (np.abs(numerical) + 1e-30),
    }


# ══════════════════════════════════════════════════════════════════════════
# HIERARCHICAL MODEL
# ══════════════════════════════════════════════════════════════════════════

def unpack_hierarchical_centered(theta: np.ndarray, K: int) -> dict:
    """
    Unpack transformed parameters for the CENTERED hierarchical model.

    theta layout: [log_phi_1, ..., log_phi_K, gamma, log_eta, delta,
                   mu_phi, log_sigma_phi]
    Total dimension: K + 5
    """
    log_phi_k = theta[:K]
    gamma_val = theta[K]
    log_eta = theta[K + 1]
    delta_val = theta[K + 2]
    mu_phi = theta[K + 3]
    log_sigma_phi = theta[K + 4]

    return {
        "phi_k": np.exp(log_phi_k),
        "log_phi_k": log_phi_k,
        "gamma": gamma_val,
        "eta": np.exp(log_eta),
        "log_eta": log_eta,
        "delta": delta_val,
        "mu_phi": mu_phi,
        "sigma_phi": np.exp(log_sigma_phi),
        "log_sigma_phi": log_sigma_phi,
    }


def unpack_hierarchical_noncentered(theta: np.ndarray, K: int) -> dict:
    """
    Unpack transformed parameters for the NON-CENTERED hierarchical model.

    theta layout: [z_1, ..., z_K, gamma, log_eta, delta,
                   mu_phi, log_sigma_phi]

    where log(phi_k) = mu_phi + sigma_phi * z_k
    """
    z_k = theta[:K]
    gamma_val = theta[K]
    log_eta = theta[K + 1]
    delta_val = theta[K + 2]
    mu_phi = theta[K + 3]
    log_sigma_phi = theta[K + 4]
    sigma_phi = np.exp(log_sigma_phi)

    log_phi_k = mu_phi + sigma_phi * z_k

    return {
        "phi_k": np.exp(log_phi_k),
        "log_phi_k": log_phi_k,
        "z_k": z_k,
        "gamma": gamma_val,
        "eta": np.exp(log_eta),
        "log_eta": log_eta,
        "delta": delta_val,
        "mu_phi": mu_phi,
        "sigma_phi": sigma_phi,
        "log_sigma_phi": log_sigma_phi,
    }


def log_posterior_hierarchical(theta: np.ndarray,
                               seasons: list[Dataset],
                               parameterization: str = "centered",
                               prior_type: str = "weakly") -> float:
    """
    Log-posterior for the hierarchical multi-season model.

    Parameters
    ----------
    theta : parameter vector (length K + 5)
    seasons : list of K Dataset objects
    parameterization : 'centered' or 'noncentered'
    prior_type : prior specification

    Returns
    -------
    float : log p(theta | data)
    """
    K = len(seasons)

    if parameterization == "centered":
        p = unpack_hierarchical_centered(theta, K)
    elif parameterization == "noncentered":
        p = unpack_hierarchical_noncentered(theta, K)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    phi_k = p["phi_k"]
    gamma_val = p["gamma"]
    eta = p["eta"]
    delta_val = p["delta"]
    mu_phi = p["mu_phi"]
    sigma_phi = p["sigma_phi"]

    # Positivity checks
    if eta <= 0 or sigma_phi <= 0 or np.any(phi_k <= 0):
        return -np.inf

    lp = 0.0

    # ── Log-likelihood: sum over seasons and bins ──
    for k in range(K):
        ds = seasons[k]
        mu = expected_counts_base(ds.E_centers, ds.E_widths,
                                  phi_k[k], gamma_val, eta, delta_val)
        if np.any(mu <= 0):
            return -np.inf
        lp += poisson_log_likelihood(ds.counts, mu)

    # ── Population prior on log_phi_k: N(mu_phi, sigma_phi^2) ──
    log_phi_k = np.log(phi_k)
    lp += -0.5 * K * np.log(2 * np.pi) - K * np.log(sigma_phi)
    lp += -0.5 * np.sum(((log_phi_k - mu_phi) / sigma_phi)**2)

    # ── Hyperpriors ──
    # mu_phi ~ N(0, 10)  [on log-flux scale]
    lp += -0.5 * (mu_phi / np.sqrt(10.0))**2

    # sigma_phi ~ Half-Cauchy(0, 1): p(sigma) = 2 / (pi * (1 + sigma^2))
    lp += -np.log(1 + sigma_phi**2)  # + const

    # ── Shared parameter priors ──
    # delta ~ N(3.7, 0.1^2)
    lp += -0.5 * ((delta_val - 3.7) / 0.1)**2

    if prior_type == "weakly":
        # gamma ~ N(2.5, 0.5^2)
        lp += -0.5 * ((gamma_val - 2.5) / 0.5)**2
        # eta ~ LogNormal(log(500), 1)
        lp += -0.5 * ((np.log(eta) - np.log(500.0)) / 1.0)**2 - np.log(eta)

    # ── Jacobians for log-transforms ──
    if parameterization == "centered":
        # Jacobian for log_phi_k -> phi_k: adds sum(log_phi_k)
        lp += np.sum(p["log_phi_k"])
    # For noncentered: z_k are unconstrained, no Jacobian needed for them.
    # But phi_k appears in the likelihood, and the population prior is on
    # log_phi_k = mu_phi + sigma_phi * z_k, which is handled above.

    # Jacobian for log_eta -> eta
    lp += p["log_eta"]
    # Jacobian for log_sigma_phi -> sigma_phi
    lp += p["log_sigma_phi"]

    return lp


# ══════════════════════════════════════════════════════════════════════════
# TESTS
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from simulator import generate_single_season, generate_hierarchical

    rng = np.random.default_rng(42)

    # ── Test base model ──
    print("=" * 60)
    print("BASE MODEL: Log-posterior and gradient verification")
    print("=" * 60)

    ds = generate_single_season(rng=rng)

    # True params in transformed space
    theta_true = np.array([
        np.log(ds.true_params["phi"]),
        ds.true_params["gamma"],
        np.log(ds.true_params["eta"]),
        ds.true_params["delta"],
    ])

    lp = log_posterior_base(theta_true, ds.counts, ds.E_centers,
                            ds.E_widths, "weakly")
    print(f"\nLog-posterior at true params: {lp:.2f}")

    # Verify gradient
    result = verify_gradient(theta_true, ds.counts, ds.E_centers,
                              ds.E_widths, "weakly")
    print(f"\nGradient verification:")
    print(f"  Analytic:  {result['analytic']}")
    print(f"  Numerical: {result['numerical']}")
    print(f"  Abs diff:  {result['abs_diff']}")
    print(f"  Max abs diff: {result['max_abs_diff']:.2e}")

    # Test at a perturbed point too
    theta_perturbed = theta_true + rng.normal(0, 0.1, size=4)
    result2 = verify_gradient(theta_perturbed, ds.counts, ds.E_centers,
                               ds.E_widths, "weakly")
    print(f"\nGradient verification at perturbed point:")
    print(f"  Max abs diff: {result2['max_abs_diff']:.2e}")

    # ── Test hierarchical model ──
    print("\n" + "=" * 60)
    print("HIERARCHICAL MODEL: Log-posterior evaluation")
    print("=" * 60)

    hds = generate_hierarchical(K=10, rng=rng)
    K = len(hds.seasons)

    # Centered parameterization: theta = [log_phi_1,...,log_phi_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
    theta_hier = np.concatenate([
        np.log(hds.phi_k),
        [hds.true_params["gamma"],
         np.log(hds.true_params["eta"]),
         hds.true_params["delta"],
         hds.true_params["mu_phi"],
         np.log(hds.true_params["sigma_phi"])],
    ])

    lp_c = log_posterior_hierarchical(theta_hier, hds.seasons,
                                      "centered", "weakly")
    print(f"\nLog-posterior (centered) at true params: {lp_c:.2f}")

    # Non-centered: convert to z_k
    sigma_phi = hds.true_params["sigma_phi"]
    mu_phi = hds.true_params["mu_phi"]
    z_k = (np.log(hds.phi_k) - mu_phi) / sigma_phi

    theta_nc = np.concatenate([
        z_k,
        [hds.true_params["gamma"],
         np.log(hds.true_params["eta"]),
         hds.true_params["delta"],
         mu_phi,
         np.log(sigma_phi)],
    ])

    lp_nc = log_posterior_hierarchical(theta_nc, hds.seasons,
                                       "noncentered", "weakly")
    print(f"Log-posterior (non-centered) at true params: {lp_nc:.2f}")

    # They won't be exactly equal due to different Jacobians, but both
    # should be finite and reasonable
    print(f"\nBoth finite: {np.isfinite(lp_c) and np.isfinite(lp_nc)}")