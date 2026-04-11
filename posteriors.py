"""
Log posterior and gradient functions.
We give log posterior evaluation and analytic gradients for the base model with params (phi, gamma, eta, delta)
and the hierarchical model with params (phi_k, mu_phi, sigma_phi, others)
We transform the param space for unconstrained MCMC,

log_phi = log(phi),
gamma=gamma,
log_eta = log(eta),
delta=delta

and for the hierarchical model,
log_phi_k = log(phi_k),
mu_phi=mu_phi,
log_sigma_phi = log(sigma_phi)
"""

import numpy as np
from simulator import (signal_power_law, background_atmospheric, E_REF, Dataset, HierarchicalDataset)


# BASE MODEL

def expected_counts_base(E, E_widths, phi, gamma, eta, delta) -> np.ndarray:
    # here we compute expected counts mu_i = signal + background for each bin
    mu_sig = signal_power_law(E, phi, gamma, E_widths)
    mu_bg = background_atmospheric(E, eta, delta, E_widths)
    return mu_sig + mu_bg


def poisson_log_likelihood(counts, mu) -> float:
    # sum_i [n_i * log(mu_i) - mu_i - log(n_i!)], dropping the last term as a constant
    with np.errstate(divide='ignore', invalid='ignore'):
        log_mu = np.where(mu > 0, np.log(mu), -np.inf)
    ll = np.sum(counts * log_mu - mu)
    return ll


# Priors for base model
def log_prior_base(phi, gamma, eta, delta, prior_type = "flat") -> float:
    """
    log prior for (phi, gamma, eta, delta).
    prior_type can be 'flat', 'weakly', 'jeffreys', 'lognormal_gamma'
    'flat' gives improper flat priors on all,
    'weakly' is log normal on phi/eta, normal on gamma, normal on delta
    all versions have delta = N(3.7, 0.1^2)
    """
    # Positivity constraints
    if phi <= 0 or eta <= 0:
        return -np.inf

    lp = 0.0

    # delta prior is always N(3.7, 0.1^2)
    lp += -0.5 * ((delta - 3.7) / 0.1)**2

    if prior_type == "flat":
        pass

    elif prior_type == "weakly":
        # phi
        lp += -0.5 * ((np.log(phi) - np.log(30.0)) / 1.0)**2 - np.log(phi)
        # gamma = normal
        lp += -0.5 * ((gamma - 2.5) / 0.5)**2
        # eta
        lp += -0.5 * ((np.log(eta) - np.log(500.0)) / 1.0)**2 - np.log(eta)

    elif prior_type == "lognormal_gamma":
        # gamma = LogNormal(log(2.5), 0.2^2)
        if gamma <= 0:
            return -np.inf
        lp += -0.5 * ((np.log(gamma) - np.log(2.5)) / 0.2)**2 - np.log(gamma)
        #others flat
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    return lp


# Log posterior in TRANSFORMED space

def unpack_base_params(theta_transformed) -> tuple:
    """
    here we just unpack the transformed parameters to natural scale
    theta_transformed = [log_phi, gamma, log_eta, delta]
    returns (phi, gamma, eta, delta)
    """
    log_phi, gamma, log_eta, delta = theta_transformed
    return np.exp(log_phi), gamma, np.exp(log_eta), delta


def log_posterior_base(theta_transformed, counts, E, E_widths, prior_type = "weakly") -> float:
    """
    log posterior for base model in transformed parameter space, returns log posterior.
    """
    phi, gamma, eta, delta = unpack_base_params(theta_transformed)

    # log prior
    lp = log_prior_base(phi, gamma, eta, delta, prior_type)
    if not np.isfinite(lp):
        return -np.inf

    # jacobian for log transform, always add log(phi) + log(eta)
    lp += theta_transformed[0] + theta_transformed[2]

    # ll
    mu = expected_counts_base(E, E_widths, phi, gamma, eta, delta)
    if np.any(mu <= 0):
        return -np.inf
    lp += poisson_log_likelihood(counts, mu)

    return lp


# Analytic gradient in TRANSFORMED space

def grad_log_posterior_base(theta_transformed, counts, E, E_widths, prior_type = "weakly") -> np.ndarray:
    """
    grad of log posterior wrt transformed parameters.
    returns (4,) array of [d/d(log_phi), d/d(gamma), d/d(log_eta), d/d(delta)]
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

    # ratio is (n_i / mu_i) - 1
    ratio = counts / mu - 1.0

    # grad of log likelihood

    # d/d(log_phi) = d/d(phi) * phi
    # d(log L)/d(phi) = sum_i ratio_i * E_norm_i^{-gamma} * dE_norm_i
    dL_dphi = np.sum(ratio * E_norm**(-gamma_val) * dE_norm)
    dL_dlog_phi = dL_dphi * phi

    # d/d(gamma)
    # d(mu_sig)/d(gamma) = phi * E_norm^{-gamma} * (-log E_norm) * dE_norm
    # = mu_sig * (-log E_norm)
    dmu_dgamma = -mu_sig * np.log(E_norm)
    dL_dgamma = np.sum(ratio * dmu_dgamma)

    # d/d(log_eta) = d/d(eta) * eta
    dL_deta = np.sum(ratio * E_norm**(-delta_val) * dE_norm)
    dL_dlog_eta = dL_deta * eta

    # d/d(delta)
    dmu_ddelta = -mu_bg * np.log(E_norm)
    dL_ddelta = np.sum(ratio * dmu_ddelta)

    grad = np.array([dL_dlog_phi, dL_dgamma, dL_dlog_eta, dL_ddelta])

    # grad of logprior

    # delta prior
    grad[3] += -(delta_val - 3.7) / 0.1**2

    if prior_type == "weakly":
        # phi = LogNormal(log(30), 1), d/d(log_phi) of logprior
        # log p(phi) = -0.5*((log(phi)-log(30))/1)^2 - log(phi)
        # d/d(log_phi) = -(log_phi - log(30))/1 - 1
        grad[0] += -(log_phi - np.log(30.0)) / 1.0**2 - 1.0
        # gamma = N(2.5, 0.5^2)
        grad[1] += -(gamma_val - 2.5) / 0.5**2
        # eta = LogNormal(log(500), 1)
        grad[2] += -(log_eta - np.log(500.0)) / 1.0**2 - 1.0

    # jacobian gradient
    # d/d(log_phi) of log(phi) = 1, same for log(eta)
    grad[0] += 1.0
    grad[2] += 1.0

    return grad


# sanity just to verify the gradient

def verify_gradient(theta, counts, E, E_widths, prior_type = "weakly", eps = 1e-5) -> dict:

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



# HIERARCHICAL MODEL

def unpack_hierarchical_centered(theta, K) -> dict:
    """
    unpack transformed parameters for the centered hierarchical model.
    theta layout= [log_phi_1, ..., log_phi_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
    dim = K + 5
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


def unpack_hierarchical_noncentered(theta, K) -> dict:
    """
    unpack transformed parameters for the non centered hierarchical model
    theta layout= [z_1, ..., z_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
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


def log_posterior_hierarchical(theta, seasons, parameterization = "centered", prior_type = "weakly") -> float:
    """
    Log posterior for the hierarchical multi season model
    params are:
    theta = parameter vector (len K + 5)
    seasons = list of K Dataset objects
    parameterization = 'centered'/'noncentered'
    prior_type
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


    if eta <= 0 or sigma_phi <= 0 or np.any(phi_k <= 0):
        return -np.inf

    lp = 0.0

    # log likelihood is sum over seasons and bins
    for k in range(K):
        ds = seasons[k]
        mu = expected_counts_base(ds.E_centers, ds.E_widths,
                                  phi_k[k], gamma_val, eta, delta_val)
        if np.any(mu <= 0):
            return -np.inf
        lp += poisson_log_likelihood(ds.counts, mu)

    # population prior on log_phi_k is N(mu_phi, sigma_phi^2)
    log_phi_k = np.log(phi_k)
    lp += -0.5 * K * np.log(2 * np.pi) - K * np.log(sigma_phi)
    lp += -0.5 * np.sum(((log_phi_k - mu_phi) / sigma_phi)**2)

    # hyperpriors
    # mu_phi = N(0, 10)
    lp += -0.5 * (mu_phi / np.sqrt(10.0))**2

    # sigma_phi = Half-Cauchy(0, 1)
    lp += -np.log(1 + sigma_phi**2)

    # shared parame priors
    # delta =  N(3.7, 0.1^2)
    lp += -0.5 * ((delta_val - 3.7) / 0.1)**2

    if prior_type == "weakly":
        # gamma = N(2.5, 0.5^2)
        lp += -0.5 * ((gamma_val - 2.5) / 0.5)**2
        # eta = LogNormal(log(500), 1)
        lp += -0.5 * ((np.log(eta) - np.log(500.0)) / 1.0)**2 - np.log(eta)

    # jacobians
    if parameterization == "centered":
        lp += np.sum(p["log_phi_k"])

    # log_eta = eta
    lp += p["log_eta"]
    # log_sigma_phi = sigma_phi
    lp += p["log_sigma_phi"]

    return lp


# test
if __name__ == "__main__":
    from simulator import generate_single_season, generate_hierarchical

    rng = np.random.default_rng(42)

    print("BASE MODEL Log posterior and gradient check")

    ds = generate_single_season(rng=rng)

    # true params in the transformed space
    theta_true = np.array([
        np.log(ds.true_params["phi"]),
        ds.true_params["gamma"],
        np.log(ds.true_params["eta"]),
        ds.true_params["delta"],
    ])

    lp = log_posterior_base(theta_true, ds.counts, ds.E_centers, ds.E_widths, "weakly")
    print(f"\nLog posterior at true params, {lp:.2f}")

    result = verify_gradient(theta_true, ds.counts, ds.E_centers, ds.E_widths, "weakly")
    print(f"\nGradient verification.")
    print(f"Analytic: {result['analytic']}")
    print(f"Numerical: {result['numerical']}")
    print(f"Abs diff: {result['abs_diff']}")
    print(f"Max abs diff: {result['max_abs_diff']:.2e}")

    theta_perturbed = theta_true + rng.normal(0, 0.1, size=4)
    result2 = verify_gradient(theta_perturbed, ds.counts, ds.E_centers, ds.E_widths, "weakly")
    print(f"\nGradient check at perturbed point.")
    print(f"Max abs diff: {result2['max_abs_diff']:.2e}")

    # test hierarchical model
    print("HIERARCHICAL MODEL, log posterior eval")

    hds = generate_hierarchical(K=10, rng=rng)
    K = len(hds.seasons)

    # centered parameterization theta = [log_phi_1,...,log_phi_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
    theta_hier = np.concatenate([
        np.log(hds.phi_k),
        [hds.true_params["gamma"],
         np.log(hds.true_params["eta"]),
         hds.true_params["delta"],
         hds.true_params["mu_phi"],
         np.log(hds.true_params["sigma_phi"])],
    ])

    lp_c = log_posterior_hierarchical(theta_hier, hds.seasons, "centered", "weakly")
    print(f"\nLog posterior (centered) at true params: {lp_c:.2f}")

    # non centered, we convert to z_k
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

    lp_nc = log_posterior_hierarchical(theta_nc, hds.seasons, "noncentered", "weakly")
    print(f"Log posterior (non centered) at true params: {lp_nc:.2f}")
    print(f"\nBoth finite? {np.isfinite(lp_c) and np.isfinite(lp_nc)}")