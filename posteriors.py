"""
Log posterior and gradient functions.
We give log posterior evaluation and analytic gradients for the base model with params (phi, gamma, eta, delta)
also can handle different fit models (power law, broken power law, cutoff)
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

from __future__ import annotations

import numpy as np
from simulator import (
    signal_power_law,
    signal_counts_for_model,
    background_atmospheric,
    atmospheric_shape,
    atmospheric_shape_ddelta,
    E_REF,
    SECONDS_PER_YEAR,
    Dataset,
    HierarchicalDataset,
)


# BASE MODEL

def _apply_response(mu_true, model_options) -> np.ndarray:
    if model_options is None:
        return mu_true
    acc = model_options.get("detector_acceptance", None)
    M = model_options.get("detector_smearing", None)
    livetime = model_options.get("livetime_years", 1.0)
    physical_flux_units = bool(model_options.get("physical_flux_units", False))
    if acc is None:
        acc = np.ones_like(mu_true)
    if M is None:
        M = np.eye(len(mu_true))
    exposure = livetime * (SECONDS_PER_YEAR if physical_flux_units else 1.0)
    return M @ (exposure * acc * mu_true)


def _uses_free_prompt(model_options) -> bool:
    if model_options is None:
        return False
    return (
        model_options.get("atmo_model", "power_law") == "conv_plus_prompt"
        and bool(model_options.get("infer_prompt_eta", True))
    )


def _component_terms(E, E_widths, phi, gamma, eta, delta, eta_prompt=None, model_options=None):
    atmo_model = "power_law" if model_options is None else model_options.get("atmo_model", "power_law")
    prompt_fraction = 0.0 if model_options is None else model_options.get("prompt_fraction", 0.0)
    delta_prompt = 2.7 if model_options is None else model_options.get("delta_prompt", 2.7)
    E_knee = 3e5 if model_options is None else model_options.get("E_knee", 3e5)
    knee_sharpness = 4.0 if model_options is None else model_options.get("knee_sharpness", 4.0)

    infer_prompt_eta = _uses_free_prompt(model_options)
    E_norm = E / E_REF

    mu_sig_true = signal_power_law(E, phi, gamma, E_widths)

    if atmo_model == "conv_plus_prompt":
        conv_shape = E_norm**(-delta) / (1.0 + (E / E_knee)**knee_sharpness)
        prompt_shape = E_norm**(-delta_prompt)
        if infer_prompt_eta:
            if eta_prompt is None:
                raise ValueError("eta_prompt must be provided when infer_prompt_eta is enabled")
            mu_bg_true = (eta * conv_shape + eta_prompt * prompt_shape) * E_widths
            dmu_deta_true = conv_shape * E_widths
            dmu_deta_prompt_true = prompt_shape * E_widths
        else:
            mu_bg_true = eta * (conv_shape + prompt_fraction * prompt_shape) * E_widths
            dmu_deta_true = (conv_shape + prompt_fraction * prompt_shape) * E_widths
            dmu_deta_prompt_true = None
        dshape_ddelta = -conv_shape * np.log(E_norm)
    else:
        bg_shape = atmospheric_shape(
            E=E,
            delta=delta,
            model=atmo_model,
            prompt_fraction=prompt_fraction,
            delta_prompt=delta_prompt,
            E_knee=E_knee,
            knee_sharpness=knee_sharpness,
        )
        mu_bg_true = eta * bg_shape * E_widths
        dmu_deta_true = bg_shape * E_widths
        dmu_deta_prompt_true = None
        dshape_ddelta = atmospheric_shape_ddelta(
            E=E,
            delta=delta,
            model=atmo_model,
            prompt_fraction=prompt_fraction,
            E_knee=E_knee,
            knee_sharpness=knee_sharpness,
        )

    mu_sig = _apply_response(mu_sig_true, model_options)
    mu_bg = _apply_response(mu_bg_true, model_options)

    dmu_dphi_true = E_norm**(-gamma) * E_widths
    dmu_dgamma_true = -mu_sig_true * np.log(E_norm)
    dmu_ddelta_true = eta * dshape_ddelta * E_widths

    dmu_dphi = _apply_response(dmu_dphi_true, model_options)
    dmu_dgamma = _apply_response(dmu_dgamma_true, model_options)
    dmu_deta = _apply_response(dmu_deta_true, model_options)
    dmu_ddelta = _apply_response(dmu_ddelta_true, model_options)
    dmu_deta_prompt = None if dmu_deta_prompt_true is None else _apply_response(dmu_deta_prompt_true, model_options)

    return {
        "mu_sig": mu_sig,
        "mu_bg": mu_bg,
        "dmu_dphi": dmu_dphi,
        "dmu_dgamma": dmu_dgamma,
        "dmu_deta": dmu_deta,
        "dmu_deta_prompt": dmu_deta_prompt,
        "dmu_ddelta": dmu_ddelta,
    }


def expected_counts_base(E, E_widths, phi, gamma, eta, delta, eta_prompt=None, model_options=None) -> np.ndarray:
    # here we compute expected counts mu_i = signal + background for each bin
    comp = _component_terms(E, E_widths, phi, gamma, eta, delta, eta_prompt=eta_prompt, model_options=model_options)
    return comp["mu_sig"] + comp["mu_bg"]


def get_spectral_param_layout(recover_model: str, infer_prompt_eta: bool) -> list[str]:
    if recover_model == "power_law":
        base = ["log_phi", "gamma"]
    elif recover_model == "broken_power_law":
        # continuous BPL with single normalization at pivot
        base = ["log_phi", "gamma1", "gamma2", "log_E_break"]
    elif recover_model == "cutoff":
        base = ["log_phi", "gamma", "log_E_cut"]
    else:
        raise ValueError(f"Unknown recover_model: {recover_model}")

    shared = ["log_eta", "delta"]
    if infer_prompt_eta:
        shared.append("log_eta_prompt")
    return base + shared


def unpack_spectral_params(theta_transformed, recover_model: str, model_options=None) -> dict:
    infer_prompt_eta = _uses_free_prompt(model_options)
    names = get_spectral_param_layout(recover_model, infer_prompt_eta)
    if len(theta_transformed) != len(names):
        raise ValueError(
            f"Expected {len(names)} params for {recover_model} with infer_prompt_eta={infer_prompt_eta}, "
            f"got {len(theta_transformed)}"
        )
    out = {k: float(v) for k, v in zip(names, theta_transformed)}
    for k in list(out.keys()):
        if k.startswith("log_"):
            out[k.replace("log_", "")] = float(np.exp(out[k]))
    return out


def expected_counts_spectral(E, E_widths, recover_model: str, p: dict, model_options=None) -> np.ndarray:
    mu_sig_true = signal_counts_for_model(E, E_widths, recover_model, p)

    atmo_model = "power_law" if model_options is None else model_options.get("atmo_model", "power_law")
    prompt_fraction = 0.0 if model_options is None else model_options.get("prompt_fraction", 0.0)
    delta_prompt = 2.7 if model_options is None else model_options.get("delta_prompt", 2.7)
    E_knee = 3e5 if model_options is None else model_options.get("E_knee", 3e5)
    knee_sharpness = 4.0 if model_options is None else model_options.get("knee_sharpness", 4.0)

    mu_bg_true = background_atmospheric(
        E=E,
        eta=p["eta"],
        delta=p["delta"],
        E_widths=E_widths,
        model=atmo_model,
        prompt_fraction=prompt_fraction,
        eta_prompt=p.get("eta_prompt", None),
        delta_prompt=delta_prompt,
        E_knee=E_knee,
        knee_sharpness=knee_sharpness,
    )

    mu_sig = _apply_response(mu_sig_true, model_options)
    mu_bg = _apply_response(mu_bg_true, model_options)
    return mu_sig + mu_bg


def log_prior_spectral(p: dict, recover_model: str, prior_type="weakly", model_options=None) -> float:
    phi = p["phi"]
    eta = p["eta"]
    delta = p["delta"]
    eta_prompt = p.get("eta_prompt", None)

    if phi <= 0 or eta <= 0:
        return -np.inf
    if eta_prompt is not None and eta_prompt <= 0:
        return -np.inf
    if recover_model == "broken_power_law" and p["E_break"] <= 0:
        return -np.inf
    if recover_model == "cutoff" and p["E_cut"] <= 0:
        return -np.inf

    lp = 0.0

    delta_mean = 3.7 if model_options is None else model_options.get("prior_delta_mean", 3.7)
    delta_sd = 0.15 if model_options is None else model_options.get("prior_delta_sd", 0.15)
    lp += -0.5 * ((delta - delta_mean) / delta_sd)**2

    phi_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_phi_log_mean", np.log(1e-5))
    phi_log_sd = 2.0 if model_options is None else model_options.get("prior_phi_log_sd", 2.0)
    gamma_mean = 2.5 if model_options is None else model_options.get("prior_gamma_mean", 2.5)
    gamma_sd = 0.4 if model_options is None else model_options.get("prior_gamma_sd", 0.4)
    eta_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_eta_log_mean", np.log(1e-5))
    eta_log_sd = 1.0 if model_options is None else model_options.get("prior_eta_log_sd", 1.0)

    if prior_type == "flat":
        pass
    elif prior_type == "weakly":
        lp += -0.5 * ((np.log(phi) - phi_log_mean) / phi_log_sd)**2 - np.log(phi)
        lp += -0.5 * ((np.log(eta) - eta_log_mean) / eta_log_sd)**2 - np.log(eta)

        if recover_model in {"power_law", "cutoff"}:
            lp += -0.5 * ((p["gamma"] - gamma_mean) / gamma_sd)**2
        else:
            lp += -0.5 * ((p["gamma1"] - gamma_mean) / gamma_sd)**2
            lp += -0.5 * ((p["gamma2"] - gamma_mean) / gamma_sd)**2

        if recover_model == "broken_power_law":
            e0 = 3e4 if model_options is None else model_options.get("E_break_ref", 3e4)
            lp += -0.5 * ((np.log(p["E_break"]) - np.log(e0)) / 1.0)**2 - np.log(p["E_break"])
        if recover_model == "cutoff":
            e0 = 1e6 if model_options is None else model_options.get("E_cut_ref", 1e6)
            lp += -0.5 * ((np.log(p["E_cut"]) - np.log(e0)) / 1.0)**2 - np.log(p["E_cut"])

        if eta_prompt is not None:
            log_mean = np.log(1e-6) if model_options is None else model_options.get("prompt_prior_log_mean", np.log(1e-6))
            log_sd = 1.0 if model_options is None else model_options.get("prompt_prior_log_sd", 1.0)
            lp += -0.5 * ((np.log(eta_prompt) - log_mean) / log_sd)**2 - np.log(eta_prompt)
    elif prior_type == "lognormal_gamma":
        if recover_model in {"power_law", "cutoff"}:
            gamma_val = p["gamma"]
            if gamma_val <= 0:
                return -np.inf
            lp += -0.5 * ((np.log(gamma_val) - np.log(2.5)) / 0.2)**2 - np.log(gamma_val)
        else:
            for gk in ("gamma1", "gamma2"):
                gv = p[gk]
                if gv <= 0:
                    return -np.inf
                lp += -0.5 * ((np.log(gv) - np.log(2.5)) / 0.2)**2 - np.log(gv)
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    return float(lp)


def log_posterior_spectral(theta_transformed, counts, E, E_widths, recover_model: str, prior_type="weakly", model_options=None) -> float:
    p = unpack_spectral_params(theta_transformed, recover_model=recover_model, model_options=model_options)
    lp = log_prior_spectral(p, recover_model=recover_model, prior_type=prior_type, model_options=model_options)
    if not np.isfinite(lp):
        return -np.inf

    # transformed-space Jacobian terms
    lp += p["log_phi"] + p["log_eta"]
    if "log_eta_prompt" in p:
        lp += p["log_eta_prompt"]
    if recover_model == "broken_power_law":
        lp += p["log_E_break"]
    if recover_model == "cutoff":
        lp += p["log_E_cut"]

    mu = expected_counts_spectral(E, E_widths, recover_model, p, model_options=model_options)
    if np.any((mu <= 0) & (counts > 0)):
        return -np.inf
    lp += poisson_log_likelihood(counts, mu)
    return float(lp)


def grad_log_posterior_spectral_numerical(
    theta_transformed,
    counts,
    E,
    E_widths,
    recover_model: str,
    prior_type="weakly",
    model_options=None,
    eps=1e-5,
) -> np.ndarray:
    theta = np.asarray(theta_transformed, dtype=float)
    g = np.zeros_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = eps
        f_plus = log_posterior_spectral(
            theta + e, counts, E, E_widths, recover_model=recover_model, prior_type=prior_type, model_options=model_options
        )
        f_minus = log_posterior_spectral(
            theta - e, counts, E, E_widths, recover_model=recover_model, prior_type=prior_type, model_options=model_options
        )
        if np.isfinite(f_plus) and np.isfinite(f_minus):
            g[j] = (f_plus - f_minus) / (2.0 * eps)
    return g


def poisson_log_likelihood(counts, mu) -> float:
    # sum_i [n_i * log(mu_i) - mu_i - log(n_i!)], dropping the last term as a constant
    # Handle bins with mu=0 robustly:
    # - if n_i=0, contribution is 0
    # - if n_i>0, likelihood is zero => log-likelihood = -inf
    if np.any(mu < 0):
        return -np.inf

    counts_pos = counts > 0
    if np.any(mu[counts_pos] <= 0):
        return -np.inf

    ll = -np.sum(mu)
    if np.any(counts_pos):
        ll += np.sum(counts[counts_pos] * np.log(mu[counts_pos]))
    return float(ll)


# Priors for base model
def log_prior_base(phi, gamma, eta, delta, prior_type = "flat", eta_prompt=None, model_options=None) -> float:
    """
    log prior for (phi, gamma, eta, delta).
    prior_type can be 'flat', 'weakly', 'lognormal_gamma'
    'flat' gives improper flat priors on all,
    'weakly' is log normal on phi/eta, normal on gamma, normal on delta
    all versions have delta = N(3.7, 0.1^2)
    """
    # Positivity constraints
    if phi <= 0 or eta <= 0:
        return -np.inf
    if eta_prompt is not None and eta_prompt <= 0:
        return -np.inf

    lp = 0.0

    delta_mean = 3.7 if model_options is None else model_options.get("prior_delta_mean", 3.7)
    delta_sd = 0.15 if model_options is None else model_options.get("prior_delta_sd", 0.15)
    # Conventional atmospheric index prior.
    lp += -0.5 * ((delta - delta_mean) / delta_sd)**2

    if prior_type == "flat":
        pass

    elif prior_type == "weakly":
        phi_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_phi_log_mean", np.log(1e-5))
        phi_log_sd = 2.0 if model_options is None else model_options.get("prior_phi_log_sd", 2.0)
        gamma_mean = 2.5 if model_options is None else model_options.get("prior_gamma_mean", 2.5)
        gamma_sd = 0.4 if model_options is None else model_options.get("prior_gamma_sd", 0.4)
        eta_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_eta_log_mean", np.log(1e-5))
        eta_log_sd = 1.0 if model_options is None else model_options.get("prior_eta_log_sd", 1.0)

        # Astrophysical and atmospheric normalization priors.
        lp += -0.5 * ((np.log(phi) - phi_log_mean) / phi_log_sd)**2 - np.log(phi)
        lp += -0.5 * ((gamma - gamma_mean) / gamma_sd)**2
        lp += -0.5 * ((np.log(eta) - eta_log_mean) / eta_log_sd)**2 - np.log(eta)

    elif prior_type == "lognormal_gamma":
        # gamma = LogNormal(log(2.5), 0.2^2)
        if gamma <= 0:
            return -np.inf
        lp += -0.5 * ((np.log(gamma) - np.log(2.5)) / 0.2)**2 - np.log(gamma)
        #others flat        
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    # Optional weak prior for free prompt normalization nuisance.
    if eta_prompt is not None and prior_type == "weakly":
        log_mean = np.log(1e-6) if model_options is None else model_options.get("prompt_prior_log_mean", np.log(1e-6))
        log_sd = 1.0 if model_options is None else model_options.get("prompt_prior_log_sd", 1.0)
        lp += -0.5 * ((np.log(eta_prompt) - log_mean) / log_sd)**2 - np.log(eta_prompt)

    return lp


# Log posterior in TRANSFORMED space

def unpack_base_params(theta_transformed, model_options=None) -> dict:
    """
    here we just unpack the transformed parameters to natural scale
    theta_transformed = [log_phi, gamma, log_eta, delta] or
    [log_phi, gamma, log_eta, delta, log_eta_prompt]
    """
    infer_prompt_eta = _uses_free_prompt(model_options)
    if infer_prompt_eta:
        if len(theta_transformed) != 5:
            raise ValueError("Expected 5 parameters [log_phi, gamma, log_eta, delta, log_eta_prompt]")
        log_phi, gamma, log_eta, delta, log_eta_prompt = theta_transformed
        return {
            "phi": np.exp(log_phi),
            "gamma": gamma,
            "eta": np.exp(log_eta),
            "delta": delta,
            "eta_prompt": np.exp(log_eta_prompt),
            "log_phi": log_phi,
            "log_eta": log_eta,
            "log_eta_prompt": log_eta_prompt,
        }

    if len(theta_transformed) != 4:
        raise ValueError("Expected 4 parameters [log_phi, gamma, log_eta, delta]")
    log_phi, gamma, log_eta, delta = theta_transformed
    return {
        "phi": np.exp(log_phi),
        "gamma": gamma,
        "eta": np.exp(log_eta),
        "delta": delta,
        "eta_prompt": None,
        "log_phi": log_phi,
        "log_eta": log_eta,
        "log_eta_prompt": None,
    }


def log_posterior_base(theta_transformed, counts, E, E_widths, prior_type = "weakly", model_options=None) -> float:
    """
    log posterior for base model in transformed parameter space, returns log posterior.
    """
    p = unpack_base_params(theta_transformed, model_options=model_options)
    phi = p["phi"]
    gamma = p["gamma"]
    eta = p["eta"]
    delta = p["delta"]
    eta_prompt = p["eta_prompt"]

    # log prior
    lp = log_prior_base(phi, gamma, eta, delta, prior_type, eta_prompt=eta_prompt, model_options=model_options)
    if not np.isfinite(lp):
        return -np.inf

    # jacobian for log transform, always add log(phi) + log(eta)
    lp += theta_transformed[0] + theta_transformed[2]
    if eta_prompt is not None:
        lp += p["log_eta_prompt"]

    # ll
    mu = expected_counts_base(
        E, E_widths, phi, gamma, eta, delta, eta_prompt=eta_prompt, model_options=model_options
    )
    if np.any((mu <= 0) & (counts > 0)):
        return -np.inf
    lp += poisson_log_likelihood(counts, mu)

    return lp


# Analytic gradient in TRANSFORMED space
# TO ADD: all the different prior_types!
    
def grad_log_posterior_base(theta_transformed, counts, E, E_widths, prior_type = "weakly", model_options=None) -> np.ndarray:
    """
    grad of log posterior wrt transformed parameters.
    returns transformed-space gradient.
    """
    p = unpack_base_params(theta_transformed, model_options=model_options)
    log_phi = p["log_phi"]
    gamma_val = p["gamma"]
    log_eta = p["log_eta"]
    delta_val = p["delta"]
    phi = p["phi"]
    eta = p["eta"]
    eta_prompt = p["eta_prompt"]
    log_eta_prompt = p["log_eta_prompt"]
    infer_prompt_eta = eta_prompt is not None

    comp = _component_terms(
        E=E,
        E_widths=E_widths,
        phi=phi,
        gamma=gamma_val,
        eta=eta,
        delta=delta_val,
        eta_prompt=eta_prompt,
        model_options=model_options,
    )
    mu_sig = comp["mu_sig"]
    mu_bg = comp["mu_bg"]
    mu = mu_sig + mu_bg
    if np.any((mu <= 0) & (counts > 0)):
        return np.zeros(len(theta_transformed))

    # ratio is (n_i / mu_i) - 1; for bins with mu=0 and n=0, limiting value is -1.
    ratio = np.full_like(mu, -1.0, dtype=float)
    positive_mu = mu > 0
    ratio[positive_mu] = counts[positive_mu] / mu[positive_mu] - 1.0

    # grad of log likelihood

    # d/d(log_phi) = d/d(phi) * phi
    dL_dphi = np.sum(ratio * comp["dmu_dphi"])
    dL_dlog_phi = dL_dphi * phi

    # d/d(gamma)
    dL_dgamma = np.sum(ratio * comp["dmu_dgamma"])

    # d/d(log_eta) = d/d(eta) * eta
    dL_deta = np.sum(ratio * comp["dmu_deta"])
    dL_dlog_eta = dL_deta * eta

    # d/d(delta)
    dL_ddelta = np.sum(ratio * comp["dmu_ddelta"])

    if infer_prompt_eta:
        dL_deta_prompt = np.sum(ratio * comp["dmu_deta_prompt"])
        dL_dlog_eta_prompt = dL_deta_prompt * eta_prompt
        grad = np.array([dL_dlog_phi, dL_dgamma, dL_dlog_eta, dL_ddelta, dL_dlog_eta_prompt])
    else:
        grad = np.array([dL_dlog_phi, dL_dgamma, dL_dlog_eta, dL_ddelta])

    # grad of logprior

    delta_mean = 3.7 if model_options is None else model_options.get("prior_delta_mean", 3.7)
    delta_sd = 0.15 if model_options is None else model_options.get("prior_delta_sd", 0.15)
    # delta prior
    grad[3] += -(delta_val - delta_mean) / (delta_sd**2)


    if prior_type == "flat":
        pass
        
    elif prior_type == "weakly":
        phi_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_phi_log_mean", np.log(1e-5))
        phi_log_sd = 2.0 if model_options is None else model_options.get("prior_phi_log_sd", 2.0)
        gamma_mean = 2.5 if model_options is None else model_options.get("prior_gamma_mean", 2.5)
        gamma_sd = 0.4 if model_options is None else model_options.get("prior_gamma_sd", 0.4)
        eta_log_mean = np.log(1e-5) if model_options is None else model_options.get("prior_eta_log_mean", np.log(1e-5))
        eta_log_sd = 1.0 if model_options is None else model_options.get("prior_eta_log_sd", 1.0)

        grad[0] += -(log_phi - phi_log_mean) / (phi_log_sd**2) - 1.0
        grad[1] += -(gamma_val - gamma_mean) / (gamma_sd**2)
        grad[2] += -(log_eta - eta_log_mean) / (eta_log_sd**2) - 1.0
        if infer_prompt_eta:
            log_mean = np.log(1e-6) if model_options is None else model_options.get("prompt_prior_log_mean", np.log(1e-6))
            log_sd = 1.0 if model_options is None else model_options.get("prompt_prior_log_sd", 1.0)
            grad[4] += -(log_eta_prompt - log_mean) / (log_sd**2) - 1.0

    elif prior_type == "lognormal_gamma":
        # log gamma = N(log 2.5, 0.2^2)
        if gamma_val <= 0: # guards against gamma < 0 case
            return np.zeros(len(theta_transformed))
        grad[1] += -((np.log(gamma_val) - np.log(2.5))/(0.2**2))*(1/gamma_val) - 1/gamma_val
    
    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    # jacobian gradient
    # d/d(log_phi) of log(phi) = 1, same for log(eta)
    grad[0] += 1.0
    grad[2] += 1.0
    if infer_prompt_eta:
        grad[4] += 1.0

    return grad

# sanity just to verify the gradient

def verify_gradient(theta, counts, E, E_widths, prior_type = "weakly", eps = 1e-5, model_options=None) -> dict:

    analytic = grad_log_posterior_base(theta, counts, E, E_widths, prior_type, model_options=model_options)
    numerical = np.zeros_like(theta)
    for j in range(len(theta)):
        e_j = np.zeros_like(theta)
        e_j[j] = eps
        f_plus = log_posterior_base(theta + e_j, counts, E, E_widths, prior_type, model_options=model_options)
        f_minus = log_posterior_base(theta - e_j, counts, E, E_widths, prior_type, model_options=model_options)
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
    theta layout:
      no prompt: [log_phi_1, ..., log_phi_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
      with prompt: [log_phi_1, ..., log_phi_K, gamma, log_eta, delta, log_eta_prompt, mu_phi, log_sigma_phi]
    """
    use_prompt = len(theta) == K + 6
    if len(theta) not in (K + 5, K + 6):
        raise ValueError(f"Unexpected hierarchical theta length {len(theta)} for K={K}")

    log_phi_k = theta[:K]
    gamma_val = theta[K]
    log_eta = theta[K + 1]
    delta_val = theta[K + 2]
    if use_prompt:
        log_eta_prompt = theta[K + 3]
        mu_phi = theta[K + 4]
        log_sigma_phi = theta[K + 5]
    else:
        log_eta_prompt = None
        mu_phi = theta[K + 3]
        log_sigma_phi = theta[K + 4]

    return {
        "phi_k": np.exp(log_phi_k),
        "log_phi_k": log_phi_k,
        "gamma": gamma_val,
        "eta": np.exp(log_eta),
        "log_eta": log_eta,
        "delta": delta_val,
        "eta_prompt": None if log_eta_prompt is None else np.exp(log_eta_prompt),
        "log_eta_prompt": log_eta_prompt,
        "mu_phi": mu_phi,
        "sigma_phi": np.exp(log_sigma_phi),
        "log_sigma_phi": log_sigma_phi,
    }


def unpack_hierarchical_noncentered(theta, K) -> dict:
    """
    unpack transformed parameters for the non centered hierarchical model
    theta layout:
      no prompt: [z_1, ..., z_K, gamma, log_eta, delta, mu_phi, log_sigma_phi]
      with prompt: [z_1, ..., z_K, gamma, log_eta, delta, log_eta_prompt, mu_phi, log_sigma_phi]
    where log(phi_k) = mu_phi + sigma_phi * z_k
    """
    use_prompt = len(theta) == K + 6
    if len(theta) not in (K + 5, K + 6):
        raise ValueError(f"Unexpected hierarchical theta length {len(theta)} for K={K}")

    z_k = theta[:K]
    gamma_val = theta[K]
    log_eta = theta[K + 1]
    delta_val = theta[K + 2]
    if use_prompt:
        log_eta_prompt = theta[K + 3]
        mu_phi = theta[K + 4]
        log_sigma_phi = theta[K + 5]
    else:
        log_eta_prompt = None
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
        "eta_prompt": None if log_eta_prompt is None else np.exp(log_eta_prompt),
        "log_eta_prompt": log_eta_prompt,
        "mu_phi": mu_phi,
        "sigma_phi": sigma_phi,
        "log_sigma_phi": log_sigma_phi,
    }


def log_posterior_hierarchical(theta, seasons, parameterization = "centered", prior_type = "weakly") -> float:
    """
    Log posterior for the hierarchical multi season model
    params are:
    theta = parameter vector (len K + 5 or K + 6 with prompt nuisance)
    seasons = list of K Dataset objects
    parameterization = 'centered'/'noncentered'
    prior_type: can be one of flat, weakly, lognormal_gamma
    """
    K = len(seasons)
    mo0 = seasons[0].model_options if K > 0 else {}
    allow_prompt = _uses_free_prompt(mo0)
    expected_dim = K + 6 if allow_prompt else K + 5
    if len(theta) != expected_dim:
        return -np.inf

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
    eta_prompt = p["eta_prompt"] if allow_prompt else None
    log_eta_prompt = p["log_eta_prompt"] if allow_prompt else None
    mu_phi = p["mu_phi"]
    sigma_phi = p["sigma_phi"]


    if eta <= 0 or sigma_phi <= 0 or np.any(phi_k <= 0):
        return -np.inf
    if eta_prompt is not None and eta_prompt <= 0:
        return -np.inf

    lp = 0.0

    # log likelihood is sum over seasons and bins
    for k in range(K):
        ds = seasons[k]
        mu = expected_counts_base(
            ds.E_centers,
            ds.E_widths,
            phi_k[k],
            gamma_val,
            eta,
            delta_val,
            eta_prompt=eta_prompt,
            model_options=ds.model_options,
        )
        if np.any((mu <= 0) & (ds.counts > 0)):
            return -np.inf
        lp += poisson_log_likelihood(ds.counts, mu)

    if parameterization == "centered":
        # log_phi_k | mu_phi, sigma_phi ~ N(mu_phi, sigma_phi^2)
        log_phi_k = p["log_phi_k"]
        lp += -0.5 * K * np.log(2 * np.pi) - K * np.log(sigma_phi)
        lp += -0.5 * np.sum(((log_phi_k - mu_phi) / sigma_phi)**2)
    
    elif parameterization == "noncentered":
        # z_k ~ N(0, 1)
        z_k = p["z_k"]
        lp += -0.5 * K * np.log(2 * np.pi)
        lp += -0.5 * np.sum(z_k**2)

    # hyperpriors for population center of log-phi
    mu_phi_prior_mean = mo0.get("prior_mu_phi_mean", mo0.get("prior_phi_log_mean", 0.0))
    mu_phi_prior_sd = mo0.get("prior_mu_phi_sd", mo0.get("prior_phi_log_sd", np.sqrt(10.0)))
    lp += -0.5 * ((mu_phi - mu_phi_prior_mean) / mu_phi_prior_sd)**2

    # sigma_phi = Half-Cauchy(0, 1)
    lp += -np.log(1 + sigma_phi**2)

    # shared parame priors
    delta_mean = mo0.get("prior_delta_mean", 3.7)
    delta_sd = mo0.get("prior_delta_sd", 0.15)
    lp += -0.5 * ((delta_val - delta_mean) / delta_sd)**2

    if prior_type == "flat":
        pass
    
    elif prior_type == "weakly":
        gamma_mean = mo0.get("prior_gamma_mean", 2.5)
        gamma_sd = mo0.get("prior_gamma_sd", 0.4)
        eta_log_mean = mo0.get("prior_eta_log_mean", np.log(1e-5))
        eta_log_sd = mo0.get("prior_eta_log_sd", 1.0)
        lp += -0.5 * ((gamma_val - gamma_mean) / gamma_sd)**2
        lp += -0.5 * ((np.log(eta) - eta_log_mean) / eta_log_sd)**2 - np.log(eta)
        if eta_prompt is not None:
            prompt_log_mean = mo0.get("prompt_prior_log_mean", np.log(1e-6))
            prompt_log_sd = mo0.get("prompt_prior_log_sd", 1.0)
            lp += -0.5 * ((np.log(eta_prompt) - prompt_log_mean) / prompt_log_sd)**2 - np.log(eta_prompt)

    elif prior_type == "lognormal_gamma":
        # gamma = LogNormal(log(2.5), 0.2^2)
        if gamma_val <= 0:
            return -np.inf
        lp += -0.5 * ((np.log(gamma_val) - np.log(2.5)) / 0.2)**2 - np.log(gamma_val)

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")


    # log_eta = eta
    lp += p["log_eta"]
    if log_eta_prompt is not None:
        lp += log_eta_prompt
    # log_sigma_phi = sigma_phi
    lp += p["log_sigma_phi"]

    return lp

# computes analytic gradient for hierarchical model in transformed space, as defined for the theta vector
def grad_log_posterior_hierarchical(theta, seasons, parameterization="centered", prior_type="weakly",) -> np.ndarray:
    K = len(seasons)
    mo0 = seasons[0].model_options if K > 0 else {}
    allow_prompt = _uses_free_prompt(mo0)
    expected_dim = K + 6 if allow_prompt else K + 5
    if len(theta) != expected_dim:
        return np.zeros(expected_dim)

    if parameterization == "centered":
        p = unpack_hierarchical_centered(theta, K)
    elif parameterization == "noncentered":
        p = unpack_hierarchical_noncentered(theta, K)
    else:
        raise ValueError(f"Unknown parameterization: {parameterization}")

    phi_k = p["phi_k"]
    log_phi_k = p["log_phi_k"]
    gamma_val = p["gamma"]
    eta = p["eta"]
    log_eta = p["log_eta"]
    delta_val = p["delta"]
    mu_phi = p["mu_phi"]
    sigma_phi = p["sigma_phi"]
    log_sigma_phi = p["log_sigma_phi"]

    eta_prompt = p["eta_prompt"] if allow_prompt else None
    log_eta_prompt = p["log_eta_prompt"] if allow_prompt else None

    has_prompt = allow_prompt
    n_params = K + 6 if has_prompt else K + 5

    if eta <= 0 or sigma_phi <= 0 or np.any(phi_k <= 0):
        return np.zeros(n_params)
    if eta_prompt is not None and eta_prompt <= 0:
        return np.zeros(n_params)

    grad = np.zeros(n_params)

    # Accumulators for shared parameters
    d_gamma = 0.0
    d_log_eta = 0.0
    d_delta = 0.0
    d_log_eta_prompt = 0.0

    # These differ by parameterization, but it's convenient to accumulate them here
    d_mu_phi_like = 0.0
    d_log_sigma_like = 0.0

    for k, ds in enumerate(seasons):
        comp = _component_terms(
            E=ds.E_centers,
            E_widths=ds.E_widths,
            phi=phi_k[k],
            gamma=gamma_val,
            eta=eta,
            delta=delta_val,
            eta_prompt=eta_prompt,
            model_options=ds.model_options,
        )
        mu_sig = comp["mu_sig"]
        mu_bg = comp["mu_bg"]
        mu = mu_sig + mu_bg

        if np.any((mu <= 0) & (ds.counts > 0)):
            return np.zeros(n_params)

        ratio = np.full_like(mu, -1.0, dtype=float)
        positive_mu = mu > 0
        ratio[positive_mu] = ds.counts[positive_mu] / mu[positive_mu] - 1.0

        # Shared parameter gradients from likelihood
        d_gamma += np.sum(ratio * comp["dmu_dgamma"])
        d_log_eta += eta * np.sum(ratio * comp["dmu_deta"])
        d_delta += np.sum(ratio * comp["dmu_ddelta"])
        if has_prompt:
            if comp["dmu_deta_prompt"] is None:
                return np.zeros(n_params)
            d_log_eta_prompt += eta_prompt * np.sum(ratio * comp["dmu_deta_prompt"])

        if parameterization == "centered":
            # theta_k = log_phi_k
            grad[k] = (
                phi_k[k] * np.sum(ratio * comp["dmu_dphi"])
                - (log_phi_k[k] - mu_phi) / sigma_phi**2
            )
        else:
            # theta_k = z_k, where log_phi_k = mu_phi + sigma_phi z_k
            z_k = p["z_k"][k]
            grad[k] = sigma_phi * phi_k[k] * np.sum(ratio * comp["dmu_dphi"]) - z_k

        # Contributions to hyperparameter gradients
        if parameterization == "centered":
            # no likelihood contribution to mu_phi or log_sigma_phi in centered coords
            pass
        else:
            # non-centered: likelihood depends on mu_phi and sigma_phi through phi_k
            z_k = p["z_k"][k]
            d_mu_phi_like += phi_k[k] * np.sum(ratio * comp["dmu_dphi"])
            d_log_sigma_like += sigma_phi * z_k * phi_k[k] * np.sum(ratio * comp["dmu_dphi"])

    # Shared priors / Jacobians for shared params
    grad[K] = d_gamma
    grad[K + 1] = d_log_eta
    grad[K + 2] = d_delta
    prompt_offset = 1 if has_prompt else 0
    if has_prompt:
        grad[K + 3] = d_log_eta_prompt

    delta_mean = mo0.get("prior_delta_mean", 3.7)
    delta_sd = mo0.get("prior_delta_sd", 0.15)
    grad[K + 2] += -(delta_val - delta_mean) / (delta_sd**2)

    if prior_type == "flat":
        pass

    elif prior_type == "weakly":
        gamma_mean = mo0.get("prior_gamma_mean", 2.5)
        gamma_sd = mo0.get("prior_gamma_sd", 0.4)
        eta_log_mean = mo0.get("prior_eta_log_mean", np.log(1e-5))
        eta_log_sd = mo0.get("prior_eta_log_sd", 1.0)

        grad[K] += -(gamma_val - gamma_mean) / (gamma_sd**2)

        # eta ~ LogNormal(eta_log_mean, eta_log_sd)
        # prior derivative wrt log_eta: -(log_eta - eta_log_mean)/eta_log_sd^2 - 1
        # Jacobian wrt log_eta: +1
        # net effect: -(log_eta - eta_log_mean)/eta_log_sd^2
        grad[K + 1] += -(log_eta - eta_log_mean) / (eta_log_sd**2)
        if has_prompt:
            prompt_log_mean = mo0.get("prompt_prior_log_mean", np.log(1e-6))
            prompt_log_sd = mo0.get("prompt_prior_log_sd", 1.0)
            grad[K + 3] += -(log_eta_prompt - prompt_log_mean) / (prompt_log_sd**2)

    elif prior_type == "lognormal_gamma":
        if gamma_val <= 0:
            return np.zeros(n_params)
        grad[K] += -((np.log(gamma_val) - np.log(2.5)) / 0.2**2) * (1.0 / gamma_val) - 1.0 / gamma_val

        # eta is flat in this case, so only Jacobian remains
        grad[K + 1] += 1.0
        if has_prompt:
            grad[K + 3] += 1.0

    else:
        raise ValueError(f"Unknown prior type: {prior_type}")

    if prior_type == "flat":
        # eta flat in natural space, so transformed-space Jacobian contributes +1
        grad[K + 1] += 1.0
        if has_prompt:
            grad[K + 3] += 1.0

    # Hyperparameter gradients
    mu_phi_prior_mean = mo0.get("prior_mu_phi_mean", mo0.get("prior_phi_log_mean", 0.0))
    mu_phi_prior_sd = mo0.get("prior_mu_phi_sd", mo0.get("prior_phi_log_sd", np.sqrt(10.0)))
    if parameterization == "centered":
        grad[K + 3 + prompt_offset] = (
            np.sum((log_phi_k - mu_phi) / sigma_phi**2)
            - (mu_phi - mu_phi_prior_mean) / (mu_phi_prior_sd**2)
        )
        grad[K + 4 + prompt_offset] = (
            -K
            + np.sum((log_phi_k - mu_phi)**2 / sigma_phi**2)
            - 2.0 * sigma_phi**2 / (1.0 + sigma_phi**2)
            + 1.0  # Jacobian for sigma_phi = exp(log_sigma_phi)
        )
    else:
        grad[K + 3 + prompt_offset] = (
            d_mu_phi_like
            - (mu_phi - mu_phi_prior_mean) / (mu_phi_prior_sd**2)
        )
        grad[K + 4 + prompt_offset] = (
            d_log_sigma_like
            - 2.0 * sigma_phi**2 / (1.0 + sigma_phi**2)
            + 1.0
        )

    return grad

# sanity check for gradient of hierarchical model
def verify_gradient_hierarchical(theta, seasons, parameterization = "centered", prior_type = "weakly", eps = 1e-5) -> dict:
    analytic = grad_log_posterior_hierarchical(theta, seasons, parameterization, prior_type)
    numerical = np.zeros_like(theta)
    for j in range(len(theta)):
        e_j = np.zeros_like(theta)
        e_j[j] = eps
        f_plus = log_posterior_hierarchical(theta + e_j, seasons, parameterization, prior_type)
        f_minus = log_posterior_hierarchical(theta - e_j, seasons, parameterization, prior_type)

        if np.isfinite(f_plus) and np.isfinite(f_minus):
            numerical[j] = (f_plus - f_minus) / (2 * eps)

    return {
        "analytic": analytic,
        "numerical": numerical,
        "abs_diff": np.abs(analytic - numerical),
        "max_abs_diff": np.max(np.abs(analytic - numerical)),
        "rel_diff": np.abs(analytic - numerical) / (np.abs(numerical) + 1e-30),
    }

# test
if __name__ == "__main__":
    from simulator import generate_single_season, generate_hierarchical
    from load_config import load_config

    config = load_config()
    test_cfg = config["testing"]
    signal_cfg = config["signal"]
    background_cfg = config["background"]
    bins_cfg = config["energy_bins"]
    hier_cfg = config["hierarchical"]
    atmo_cfg = config["atmospheric_realism"]
    det_cfg = config["detector_response"]
    phys_prior_cfg = config["priors"]["physics"]

    rng = np.random.default_rng(int(test_cfg["seed_posteriors"]))

    print("BASE MODEL Log posterior and gradient check")

    ds = generate_single_season(
        phi=float(signal_cfg["phi"]),
        gamma=float(signal_cfg["gamma"]),
        eta=float(background_cfg["eta"]),
        delta=float(background_cfg["delta"]),
        truth_model=test_cfg["truth_model"],
        n_bins=int(bins_cfg["n_bins"]),
        E_min=float(bins_cfg["E_min"]),
        E_max=float(bins_cfg["E_max"]),
        gamma2=float(signal_cfg["gamma2"]),
        E_break=float(signal_cfg["E_break"]),
        E_cut=float(signal_cfg["E_cut"]),
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(background_cfg.get("eta_prompt", float(background_cfg["eta"]) * float(atmo_cfg["prompt_fraction"]))),
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

    # true params in the transformed space
    if bool(ds.model_options.get("infer_prompt_eta", False)):
        theta_true = np.array([
            np.log(ds.true_params["phi"]),
            ds.true_params["gamma"],
            np.log(ds.true_params["eta"]),
            ds.true_params["delta"],
            np.log(ds.true_params["eta_prompt"]),
        ])
    else:
        theta_true = np.array([
            np.log(ds.true_params["phi"]),
            ds.true_params["gamma"],
            np.log(ds.true_params["eta"]),
            ds.true_params["delta"],
        ])

    for prior_type in test_cfg["base_prior_types"]:
        print(f"{prior_type.upper()} PRIOR CASE")
        lp = log_posterior_base(theta_true, ds.counts, ds.E_centers, ds.E_widths, prior_type, model_options=ds.model_options)
        print(f"Log posterior at true params, {lp:.2f}")

        result = verify_gradient(theta_true, ds.counts, ds.E_centers, ds.E_widths, prior_type, model_options=ds.model_options)
        print(f"\nGradient verification at true params.")
        print(f"Analytic: {result['analytic']}")
        print(f"Numerical: {result['numerical']}")
        print(f"Abs diff: {result['abs_diff']}")
        print(f"Max abs diff: {result['max_abs_diff']:.2e}")

        theta_perturbed = theta_true + rng.normal(
            0,
            float(test_cfg["theta_perturb_scale"]),
            size=theta_true.size,
        )
        result2 = verify_gradient(theta_perturbed, ds.counts, ds.E_centers, ds.E_widths, prior_type, model_options=ds.model_options)
        print(f"\nGradient check at perturbed point.")
        print(f"Analytic: {result2['analytic']}")
        print(f"Numerical: {result2['numerical']}")
        print(f"Abs diff: {result2['abs_diff']}")
        print(f"Max abs diff: {result2['max_abs_diff']:.2e}\n\n")

    # test hierarchical model
    print("HIERARCHICAL MODEL")

    hds = generate_hierarchical(
        K=int(hier_cfg["K"]),
        mu_phi=float(hier_cfg["mu_phi"]),
        sigma_phi=float(hier_cfg["sigma_phi"]),
        gamma=float(hier_cfg["gamma"]),
        eta=float(hier_cfg["eta"]),
        delta=float(hier_cfg["delta"]),
        truth_model=hier_cfg["truth_model"],
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(background_cfg.get("eta_prompt", float(background_cfg["eta"]) * float(atmo_cfg["prompt_fraction"]))),
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
    K = len(hds.seasons)
    infer_prompt_eta_hier = bool(hds.seasons[0].model_options.get("infer_prompt_eta", False))
    shared_true = [
        hds.true_params["gamma"],
        np.log(hds.true_params["eta"]),
        hds.true_params["delta"],
    ]
    if infer_prompt_eta_hier:
        shared_true.append(np.log(hds.seasons[0].true_params["eta_prompt"]))
    shared_true.extend([hds.true_params["mu_phi"], np.log(hds.true_params["sigma_phi"])])

    # centered parameterization theta = [log_phi_1,...,shared...]
    theta_hier = np.concatenate([np.log(hds.phi_k), shared_true])
    
    
    sigma_phi = hds.true_params["sigma_phi"]
    mu_phi = hds.true_params["mu_phi"]
    z_k = (np.log(hds.phi_k) - mu_phi) / sigma_phi

    shared_true_nc = [hds.true_params["gamma"], np.log(hds.true_params["eta"]), hds.true_params["delta"]]
    if infer_prompt_eta_hier:
        shared_true_nc.append(np.log(hds.seasons[0].true_params["eta_prompt"]))
    shared_true_nc.extend([mu_phi, np.log(sigma_phi)])
    theta_nc = np.concatenate([z_k, shared_true_nc])

    for prior_type in test_cfg["hierarchical_prior_types"]:
        print(f"{prior_type.upper()} PRIOR CASE")

        lp_c = log_posterior_hierarchical(theta_hier, hds.seasons, "centered", prior_type)
        print(f"Log posterior (centered) at true params: {lp_c:.2f}")
        result_c = verify_gradient_hierarchical(theta_hier, hds.seasons, "centered", prior_type)
        print(f"Gradient check at true params.")
        print(f"Max abs diff: {result_c['max_abs_diff']:.2e}")

        lp_nc = log_posterior_hierarchical(theta_nc, hds.seasons, "noncentered", prior_type)
        print(f"Log posterior (non centered) at true params: {lp_nc:.2f}")
        result_nc = verify_gradient_hierarchical(theta_nc, hds.seasons, "noncentered", prior_type)
        print(f"Gradient check at true params.")
        print(f"Max abs diff: {result_nc['max_abs_diff']:.2e}\n")
