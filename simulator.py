"""
Here we generate synthetic bin counts under multiple truth models
"""

import numpy as np
import os
from dataclasses import dataclass, field
from typing import Optional



def make_energy_bins(n_bins = 20, E_min = 1e4, E_max = 1e7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_edges = np.logspace(np.log10(E_min), np.log10(E_max), n_bins + 1)
    E_centers = np.sqrt(E_edges[:-1] * E_edges[1:])   # geometric mean
    E_widths = E_edges[1:] - E_edges[:-1]
    return E_edges, E_centers, E_widths
# Reference energy
E_REF = 1e5 
SECONDS_PER_YEAR = 365.25 * 24 * 3600.0

def signal_power_law(E, phi, gamma, E_widths) -> np.ndarray:
    """
    Single power law signal, mu_i^signal = phi * (E_i/E_ref)^{-gamma} * dE_i
    We use the normalized energy E/E_ref to keep phi at a good scale.
    """
    return phi * (E / E_REF)**(-gamma) * E_widths


def signal_broken_power_law(E, phi, gamma1, gamma2, E_break, E_widths) -> np.ndarray:
    """
    Broken power law, slope gamma1 below E_break, gamma2 above.
    """
    E_norm = E / E_REF
    E_break_norm = E_break / E_REF
    # phi * E_break_norm^{-gamma1} = A * E_break_norm^{-gamma2} for continuity
    A = phi * E_break_norm**(gamma2 - gamma1)
    mu = np.where(
        E < E_break,
        phi * E_norm**(-gamma1),
        A * E_norm**(-gamma2)
    )
    return mu * E_widths


def signal_cutoff(E, phi, gamma, E_cut, E_widths) -> np.ndarray:
    """
    Power law with exponential cutoff, phi * (E/E_ref)^{-gamma} * exp(-E / E_cut) * dE_i
    """
    return phi * (E / E_REF)**(-gamma) * np.exp(-E / E_cut) * E_widths


# BACKGROUND MODEL

def atmospheric_shape(
    E,
    delta,
    model = "power_law",
    prompt_fraction = 0.0,
    delta_prompt = 2.7,
    E_knee = 3e5,
    knee_sharpness = 4.0,
) -> np.ndarray:
    """
    Atmospheric spectral shape.
    Two variants: single power law or conventional + prompt (conv is power law with cutoff, prompt is pure power law)
    """
    E_norm = E / E_REF

    if model == "power_law":
        return E_norm**(-delta)

    if model == "conv_plus_prompt":
        # Shape helper for conv-plus-prompt; normalization is handled in background_atmospheric.
        conv = E_norm**(-delta) / (1.0 + (E / E_knee)**knee_sharpness)
        prompt = E_norm**(-delta_prompt)
        return conv + prompt

    raise ValueError(f"Unknown atmospheric model: {model}. Use 'power_law' or 'conv_plus_prompt'.")


def atmospheric_shape_ddelta(
    E,
    delta,
    model = "power_law",
    prompt_fraction = 0.0,
    E_knee = 3e5,
    knee_sharpness = 4.0,
) -> np.ndarray:
    """
    Derivative d/d(delta) of atmospheric shape
    """
    E_norm = E / E_REF
    log_term = np.log(E_norm)

    if model == "power_law":
        return -E_norm**(-delta) * log_term

    if model == "conv_plus_prompt":
        conv = E_norm**(-delta) / (1.0 + (E / E_knee)**knee_sharpness)
        # prompt term uses fixed delta_prompt, so no delta derivative contribution
        return -conv * log_term

    raise ValueError(f"Unknown atmospheric model: {model}. Use 'power_law' or 'conv_plus_prompt'.")


def background_atmospheric(
    E,
    eta,
    delta,
    E_widths,
    model = "power_law",
    prompt_fraction = 0.0,
    eta_prompt: Optional[float] = None,
    delta_prompt = 2.7,
    E_knee = 3e5,
    knee_sharpness = 4.0,
) -> np.ndarray:
    """
    Atmospheric background.
    """
    if model == "conv_plus_prompt":
        E_norm = E / E_REF
        conv_shape = E_norm**(-delta) / (1.0 + (E / E_knee)**knee_sharpness)
        prompt_shape = E_norm**(-delta_prompt)
        if eta_prompt is None:
            eta_prompt = eta * prompt_fraction
        return (eta * conv_shape + eta_prompt * prompt_shape) * E_widths

    shape = atmospheric_shape(
        E=E,
        delta=delta,
        model=model,
        prompt_fraction=prompt_fraction,
        delta_prompt=delta_prompt,
        E_knee=E_knee,
        knee_sharpness=knee_sharpness,
    )
    return eta * shape * E_widths


def load_hese75_derived_response(
    E_edges: np.ndarray,
    aeff_allsky_path: str,
    migration_path: str,
    sky_factor_sr: float = 4.0 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load precomputed HESE 7.5-year derived response:
      - sky-averaged Aeff(E_true) table [m^2]
      - migration P(E_reco | E_true)

    Returns acceptance in cm^2*sr and migration matrix (reco x true).
    """
    if not os.path.exists(aeff_allsky_path):
        raise FileNotFoundError(f"HESE all-sky Aeff file not found: {aeff_allsky_path}")
    if not os.path.exists(migration_path):
        raise FileNotFoundError(f"HESE migration file not found: {migration_path}")

    aeff_tbl = np.loadtxt(aeff_allsky_path)
    mig_tbl = np.loadtxt(migration_path)
    if aeff_tbl.ndim == 1:
        aeff_tbl = aeff_tbl[None, :]
    if mig_tbl.ndim == 1:
        mig_tbl = mig_tbl[None, :]
    if aeff_tbl.shape[1] < 4:
        raise ValueError("HESE all-sky Aeff table must have >=4 columns")
    if mig_tbl.shape[1] < 5:
        raise ValueError("HESE migration table must have >=5 columns")

    e_lo = aeff_tbl[:, 0]
    e_hi = aeff_tbl[:, 1]
    # builder stores sky-averaged Aeff [m^2]
    aeff_avg_m2 = aeff_tbl[:, 2]
    file_edges = np.unique(np.concatenate([e_lo, e_hi]))
    file_edges.sort()

    if len(file_edges) != len(E_edges) or not np.allclose(file_edges, E_edges, rtol=1e-6, atol=1e-8):
        raise ValueError(
            "HESE derived response bin edges do not match current analysis bins. "
            "Use matching energy bins (default 20 bins from 1e4 to 1e7 GeV)."
        )

    n = len(E_edges) - 1
    # Convert sky-averaged m^2 to cm^2*sr (diffuse flux integration).
    acceptance = np.maximum(aeff_avg_m2, 0.0) * 1e4 * sky_factor_sr

    t_lo = mig_tbl[:, 0]
    t_hi = mig_tbl[:, 1]
    r_lo = mig_tbl[:, 2]
    r_hi = mig_tbl[:, 3]
    prob = mig_tbl[:, 4]
    M = np.zeros((n, n), dtype=float)
    for tl, th, rl, rh, p in zip(t_lo, t_hi, r_lo, r_hi, prob):
        j = np.where((np.isclose(E_edges[:-1], tl, rtol=1e-6, atol=1e-8)) & (np.isclose(E_edges[1:], th, rtol=1e-6, atol=1e-8)))[0]
        i = np.where((np.isclose(E_edges[:-1], rl, rtol=1e-6, atol=1e-8)) & (np.isclose(E_edges[1:], rh, rtol=1e-6, atol=1e-8)))[0]
        if len(j) == 1 and len(i) == 1:
            M[i[0], j[0]] = max(float(p), 0.0)

    # Normalize columns to ensure probabilities sum to 1 for populated bins.
    col_sum = M.sum(axis=0)
    nz = col_sum > 0
    if np.any(nz):
        M[:, nz] /= col_sum[nz]
    # fallback identity for empty columns
    for j in range(n):
        if not nz[j]:
            M[j, j] = 1.0
    return acceptance, M


def build_energy_smearing_matrix(E_edges, sigma_log10 = 0.0) -> np.ndarray:
    """
    Build a Gaussian migration matrix in log10(E).
    Rows = reconstructed bins, columns = true bins.
    Columns are normalized to sum to 1.
    """
    n_bins = len(E_edges) - 1
    if sigma_log10 <= 0:
        return np.eye(n_bins)

    log_centers = 0.5 * (np.log10(E_edges[:-1]) + np.log10(E_edges[1:]))
    M = np.zeros((n_bins, n_bins), dtype=float)
    for j in range(n_bins):
        diff = (log_centers - log_centers[j]) / sigma_log10
        M[:, j] = np.exp(-0.5 * diff**2)
    col_sum = M.sum(axis=0, keepdims=True)
    col_sum = np.where(col_sum > 0, col_sum, 1.0)
    return M / col_sum


def apply_detector_response(
    mu_true,
    acceptance,
    smearing_matrix,
    livetime_years = 1.0,
    physical_flux_units = False,
) -> np.ndarray:
    """
    Apply acceptance, livetime, and optional migration to expected counts.
    """
    exposure = livetime_years * (SECONDS_PER_YEAR if physical_flux_units else 1.0)
    mu_det_true = exposure * acceptance * mu_true
    return smearing_matrix @ mu_det_true


# DATA CONTAINERS

@dataclass
class Dataset:
    # Container for a single season simulated dataset.
    #(B, ) Reference energy
    counts: np.ndarray
    # Bin centers
    E_centers: np.ndarray
    # (B+1,) bin edges
    E_edges: np.ndarray
    # Bin widths
    E_widths: np.ndarray
    # true expected signal counts
    mu_signal: np.ndarray
    # true expected background counts
    mu_background: np.ndarray
    true_params: dict
    model_options: dict = field(default_factory=dict)


@dataclass
class HierarchicalDataset:
    # Container for a multi season hierarchical dataset
    # list of K single season datasets
    seasons: list[Dataset]
    true_params: dict
    # (K,) season specific true fluxes
    phi_k: np.ndarray
    eta_k: np.ndarray


# Main generation functions

def generate_single_season(
    phi = 1e-5,
    gamma = 2.5,
    eta = 1e-5,
    delta = 3.7,
    truth_model = "power_law",
    n_bins = 20,
    E_min = 1e4,
    E_max = 1e7,
    rng = None,
    # extra params for alternative truth models
    gamma2 = 2.8,
    E_break = 3e4,
    E_cut = 1e6,
    # optional atmospheric realism
    atmo_model = "power_law",
    prompt_fraction = 0.0,
    eta_prompt: Optional[float] = None,
    infer_prompt_eta: Optional[bool] = None,
    delta_prompt = 2.7,
    E_knee = 3e5,
    knee_sharpness = 4.0,
    # prior settings
    prompt_prior_log_mean = np.log(1e-6),
    prompt_prior_log_sd = 1.0,
    prior_phi_log_mean = np.log(1e-5),
    prior_phi_log_sd = 2.0,
    prior_gamma_mean = 2.5,
    prior_gamma_sd = 0.4,
    prior_eta_log_mean = np.log(1e-5),
    prior_eta_log_sd = 1.0,
    prior_delta_mean = 3.7,
    prior_delta_sd = 0.15,
    # optional detector response realism
    detector_mode = "none",
    livetime_years = 1.0,
    sigma_log10 = 0.0,
    physical_flux_units = False,
    hese75_aeff_allsky_path: Optional[str] = None,
    hese75_migration_path: Optional[str] = None,
    hese75_sky_factor_sr: float = 4.0 * np.pi,
    **legacy_kwargs,
) -> Dataset:
    """
    Generates a single season synthetic dataset.
    """
    _ = legacy_kwargs  # backward-compatible

    if rng is None:
        rng = np.random.default_rng()

    if atmo_model == "conv_plus_prompt":
        infer_prompt_eta_effective = True if infer_prompt_eta is None else bool(infer_prompt_eta)
    else:
        infer_prompt_eta_effective = False

    E_edges, E_centers, E_widths = make_energy_bins(n_bins, E_min, E_max)

    if truth_model == "power_law":
        mu_signal = signal_power_law(E_centers, phi, gamma, E_widths)
    elif truth_model == "broken_power_law":
        mu_signal = signal_broken_power_law(
            E_centers, phi, gamma, gamma2, E_break, E_widths
        )
    elif truth_model == "cutoff":
        mu_signal = signal_cutoff(E_centers, phi, gamma, E_cut, E_widths)
    else:
        raise ValueError(f"Unknown truth model: {truth_model}")

    # background at true-energy level
    mu_background_true = background_atmospheric(
        E=E_centers,
        eta=eta,
        delta=delta,
        E_widths=E_widths,
        model=atmo_model,
        prompt_fraction=prompt_fraction,
        eta_prompt=eta_prompt,
        delta_prompt=delta_prompt,
        E_knee=E_knee,
        knee_sharpness=knee_sharpness,
    )

    # detector response 
    if detector_mode == "hese75_derived":
        if hese75_aeff_allsky_path is None or hese75_migration_path is None:
            raise ValueError(
                "detector_mode='hese75_derived' requires hese75_aeff_allsky_path and hese75_migration_path"
            )
        acceptance, smearing_matrix = load_hese75_derived_response(
            E_edges=E_edges,
            aeff_allsky_path=hese75_aeff_allsky_path,
            migration_path=hese75_migration_path,
            sky_factor_sr=hese75_sky_factor_sr,
        )
    elif detector_mode == "none":
        acceptance = np.ones_like(E_centers, dtype=float)
        smearing_matrix = build_energy_smearing_matrix(E_edges, sigma_log10=sigma_log10)
    else:
        raise ValueError(
            f"Unsupported detector_mode='{detector_mode}'. "
            "Use 'none' or 'hese75_derived'."
        )

    # observed-level expected counts
    mu_signal = apply_detector_response(
        mu_true=mu_signal,
        acceptance=acceptance,
        smearing_matrix=smearing_matrix,
        livetime_years=livetime_years,
        physical_flux_units=physical_flux_units,
    )
    mu_background = apply_detector_response(
        mu_true=mu_background_true,
        acceptance=acceptance,
        smearing_matrix=smearing_matrix,
        livetime_years=livetime_years,
        physical_flux_units=physical_flux_units,
    )

    # total expected counts
    mu_total = mu_signal + mu_background
    assert np.all(mu_total >= 0), f"Negative rates: {mu_total}"

    # draw Poisson counts
    counts = rng.poisson(mu_total)

    if eta_prompt is None and atmo_model == "conv_plus_prompt":
        eta_prompt = eta * prompt_fraction

    true_params = {
        "phi": phi, "gamma": gamma, "eta": eta, "delta": delta,
        "truth_model": truth_model,
    }
    if atmo_model == "conv_plus_prompt":
        true_params["eta_prompt"] = eta_prompt
    if truth_model == "broken_power_law":
        true_params.update({"gamma2": gamma2, "E_break": E_break})
    elif truth_model == "cutoff":
        true_params.update({"E_cut": E_cut})
    return Dataset(
        counts=counts,
        E_centers=E_centers,
        E_edges=E_edges,
        E_widths=E_widths,
        mu_signal=mu_signal,
        mu_background=mu_background,
        true_params=true_params,
        model_options={
            "atmo_model": atmo_model,
            "prompt_fraction": prompt_fraction,
            "infer_prompt_eta": infer_prompt_eta_effective,
            "delta_prompt": delta_prompt,
            "E_knee": E_knee,
            "knee_sharpness": knee_sharpness,
            "prompt_prior_log_mean": prompt_prior_log_mean,
            "prompt_prior_log_sd": prompt_prior_log_sd,
            "prior_phi_log_mean": prior_phi_log_mean,
            "prior_phi_log_sd": prior_phi_log_sd,
            "prior_gamma_mean": prior_gamma_mean,
            "prior_gamma_sd": prior_gamma_sd,
            "prior_eta_log_mean": prior_eta_log_mean,
            "prior_eta_log_sd": prior_eta_log_sd,
            "prior_delta_mean": prior_delta_mean,
            "prior_delta_sd": prior_delta_sd,
            "detector_mode": detector_mode,
            "livetime_years": livetime_years,
            "physical_flux_units": physical_flux_units,
            "detector_acceptance": acceptance,
            "detector_smearing": smearing_matrix,
        },
    )


def generate_hierarchical(
    K = 10,
    mu_phi = np.log(1e-5),
    sigma_phi = 0.3,
    mu_eta = -39.14394658089878,
    sigma_eta = 1,
    gamma = 2.5,
    delta = 3.7,
    truth_model = "power_law",
    n_bins = 20,
    E_min = 1e4,
    E_max = 1e7,
    rng = None,
    **truth_model_kwargs,
) -> HierarchicalDataset:
    """
    Generate a hierarchical multi season dataset.
    Each season k has its own flux normalization phi_k drawn from
    log(phi_k) = N(mu_phi, sigma_phi^2) but shares gamma, eta, delta.
    """
    if rng is None:
        rng = np.random.default_rng()

    truth_model_kwargs = dict(truth_model_kwargs)
    # Interpret livetime_years as total exposure for the full hierarchical dataset
    # Split evenly across seasons to avoid multiplying exposure by K
    if "livetime_years" in truth_model_kwargs:
        total_livetime = float(truth_model_kwargs["livetime_years"])
        if total_livetime <= 0:
            raise ValueError("livetime_years must be positive when provided to generate_hierarchical")
        truth_model_kwargs["livetime_years"] = total_livetime / float(K)

    # draw season specific fluxes
    log_phi_k = rng.normal(mu_phi, sigma_phi, size=K)
    phi_k = np.exp(log_phi_k)
    log_eta_k = rng.normal(mu_eta, sigma_eta, size=K)
    eta_k = np.exp(log_eta_k)
    
    seasons = []
    for k in range(K):
        ds = generate_single_season(
            phi=phi_k[k],
            gamma=gamma,
            eta=eta_k[k],
            delta=delta,
            truth_model=truth_model,
            n_bins=n_bins,
            E_min=E_min,
            E_max=E_max,
            rng=rng,
            **truth_model_kwargs,
        )
        seasons.append(ds)

    true_params = {
        "mu_phi": mu_phi,
        "sigma_phi": sigma_phi,
        "mu_eta": mu_eta,
        "sigma_eta": sigma_eta,
        "gamma": gamma,
        "delta": delta,
        "truth_model": truth_model,
        "K": K,
    }

    return HierarchicalDataset(
        seasons=seasons,
        true_params=true_params,
        phi_k=phi_k,
        eta_k=eta_k
    )




def summarize_dataset(ds) -> None:
    # just prints a quick summary of a single season dataset."""
    total_signal = ds.mu_signal.sum()
    total_bg = ds.mu_background.sum()
    total_counts = ds.counts.sum()
    snr = total_signal / total_bg if total_bg > 0 else float('inf')

    print(f"Truth model: {ds.true_params['truth_model']}")
    print(f"Bins: {len(ds.counts)}")
    print(f"Energy range: [{ds.E_edges[0]:.1e}, {ds.E_edges[-1]:.1e}] GeV")
    print(f"Expected signal counts: {total_signal:.1f}")
    print(f"Expected background counts: {total_bg:.1f}")
    print(f"Signal-to-noise ratio: {snr:.3f}")
    print(f"Observed total counts: {total_counts}")
    print(f"True params: {ds.true_params}")


if __name__ == "__main__":
    from load_config import load_config

    config = load_config()
    test_cfg = config["testing"]
    signal_cfg = config["signal"]
    background_cfg = config["background"]
    bins_cfg = config["energy_bins"]
    hier_cfg = config["hierarchical"]
    atmo_cfg = config["atmospheric_realism"]
    det_cfg = config["detector_response"]

    rng = np.random.default_rng(int(test_cfg["seed_simulator"]))
    print("SINGLE-SEASON DATASETS")

    for model in test_cfg["truth_models"]:
        print(f"\n {model} ")
        ds = generate_single_season(
            phi=float(signal_cfg["phi"]),
            gamma=float(signal_cfg["gamma"]),
            eta=float(background_cfg["eta"]),
            delta=float(background_cfg["delta"]),
            truth_model=model,
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
            prior_phi_log_mean=float(config["priors"]["physics"]["phi_log_mean"]),
            prior_phi_log_sd=float(config["priors"]["physics"]["phi_log_sd"]),
            prior_gamma_mean=float(config["priors"]["physics"]["gamma_mean"]),
            prior_gamma_sd=float(config["priors"]["physics"]["gamma_sd"]),
            prior_eta_log_mean=float(config["priors"]["physics"]["eta_log_mean"]),
            prior_eta_log_sd=float(config["priors"]["physics"]["eta_log_sd"]),
            prior_delta_mean=float(config["priors"]["physics"]["delta_mean"]),
            prior_delta_sd=float(config["priors"]["physics"]["delta_sd"]),
            detector_mode=det_cfg["mode"],
            livetime_years=float(det_cfg["livetime_years"]),
            sigma_log10=float(det_cfg["sigma_log10"]),
            physical_flux_units=bool(det_cfg.get("physical_flux_units", False)),
            hese75_aeff_allsky_path=det_cfg.get("hese75_aeff_allsky_path", None),
            hese75_migration_path=det_cfg.get("hese75_migration_path", None),
            hese75_sky_factor_sr=float(det_cfg.get("hese75_sky_factor_sr", 4.0 * np.pi)),
            rng=rng,
        )
        summarize_dataset(ds)

    print("\n" + "=" * 60)
    print(f"HIERARCHICAL DATASET (K={int(hier_cfg['K'])} seasons)")
    print("=" * 60)

    hds = generate_hierarchical(
        K=int(hier_cfg["K"]),
        mu_phi=float(hier_cfg["mu_phi"]),
        sigma_phi=float(hier_cfg["sigma_phi"]),
        mu_eta=float(hier_cfg["mu_eta"]),
        sigma_eta=float(hier_cfg["sigma_eta"]),
        gamma=float(hier_cfg["gamma"]),
        delta=float(hier_cfg["delta"]),
        truth_model=hier_cfg["truth_model"],
        n_bins=int(bins_cfg["n_bins"]),
        E_min=float(bins_cfg["E_min"]),
        E_max=float(bins_cfg["E_max"]),
        atmo_model=atmo_cfg["model"],
        prompt_fraction=float(atmo_cfg["prompt_fraction"]),
        eta_prompt=float(background_cfg.get("eta_prompt", float(background_cfg["eta"]) * float(atmo_cfg["prompt_fraction"]))),
        delta_prompt=float(atmo_cfg["delta_prompt"]),
        E_knee=float(atmo_cfg["E_knee"]),
        knee_sharpness=float(atmo_cfg["knee_sharpness"]),
        prompt_prior_log_mean=float(atmo_cfg.get("prompt_prior_log_mean", np.log(1e-6))),
        prompt_prior_log_sd=float(atmo_cfg.get("prompt_prior_log_sd", 1.0)),
        prior_phi_log_mean=float(config["priors"]["physics"]["phi_log_mean"]),
        prior_phi_log_sd=float(config["priors"]["physics"]["phi_log_sd"]),
        prior_gamma_mean=float(config["priors"]["physics"]["gamma_mean"]),
        prior_gamma_sd=float(config["priors"]["physics"]["gamma_sd"]),
        prior_eta_log_mean=float(config["priors"]["physics"]["eta_log_mean"]),
        prior_eta_log_sd=float(config["priors"]["physics"]["eta_log_sd"]),
        prior_delta_mean=float(config["priors"]["physics"]["delta_mean"]),
        prior_delta_sd=float(config["priors"]["physics"]["delta_sd"]),
        detector_mode=det_cfg["mode"],
        livetime_years=float(det_cfg["livetime_years"]),
        sigma_log10=float(det_cfg["sigma_log10"]),
        physical_flux_units=bool(det_cfg.get("physical_flux_units", False)),
        hese75_aeff_allsky_path=det_cfg.get("hese75_aeff_allsky_path", None),
        hese75_migration_path=det_cfg.get("hese75_migration_path", None),
        hese75_sky_factor_sr=float(det_cfg.get("hese75_sky_factor_sr", 4.0 * np.pi)),
        rng=rng,
    )
    print(f"\nShared params: gamma={hds.true_params['gamma']}, delta={hds.true_params['delta']}")
    print(f"Phi population: mu_phi={hds.true_params['mu_phi']:.2f}, sigma_phi={hds.true_params['sigma_phi']:.2f}")
    print(f"Eta population: mu_eta={hds.true_params['mu_eta']:.2f}, sigma_eta={hds.true_params['sigma_eta']:.2f}")
    print(f"Season fluxes phi_k: {hds.phi_k}")
    print(f"\nPer season total counts:")
    for k, season in enumerate(hds.seasons):
        print(f"  Season {k+1}: {season.counts.sum()} counts "
              f"(phi_k = {hds.phi_k[k]:.3e})")
        
