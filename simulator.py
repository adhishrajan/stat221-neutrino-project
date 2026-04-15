"""
Here we generate synthetic bin counts under multiple truth models
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional



def make_energy_bins(n_bins = 20, E_min = 1e4, E_max = 1e7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E_edges = np.logspace(np.log10(E_min), np.log10(E_max), n_bins + 1)
    E_centers = np.sqrt(E_edges[:-1] * E_edges[1:])   # geometric mean
    E_widths = E_edges[1:] - E_edges[:-1]
    return E_edges, E_centers, E_widths
# Reference energy
E_REF = 1e5 

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

def background_atmospheric(E, eta, delta, E_widths) -> np.ndarray:
    """
    Atmospheric background, eta * (E/E_ref)^{-delta} * dE_i
    """
    return eta * (E / E_REF)**(-delta) * E_widths


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


@dataclass
class HierarchicalDataset:
    # Container for a multi season hierarchical dataset
    # list of K single season datasets
    seasons: list[Dataset]
    true_params: dict
    # (K,) season specific true fluxes
    phi_k: np.ndarray


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
) -> Dataset:
    """
    Generates a single season synthetic dataset.
    """
    if rng is None:
        rng = np.random.default_rng()

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

    # background
    mu_background = background_atmospheric(E_centers, eta, delta, E_widths)

    # total expected counts
    mu_total = mu_signal + mu_background
    assert np.all(mu_total >= 0), f"Negative rates: {mu_total}"

    # draw Poisson counts
    counts = rng.poisson(mu_total)

    true_params = {
        "phi": phi, "gamma": gamma, "eta": eta, "delta": delta,
        "truth_model": truth_model,
    }
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
    )


def generate_hierarchical(
    K = 10,
    mu_phi = np.log(1e-5),
    sigma_phi = 0.3,
    gamma = 2.5,
    eta = 1e-5,
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

    # draw season specific fluxes
    log_phi_k = rng.normal(mu_phi, sigma_phi, size=K)
    phi_k = np.exp(log_phi_k)

    seasons = []
    for k in range(K):
        ds = generate_single_season(
            phi=phi_k[k],
            gamma=gamma,
            eta=eta,
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
        "gamma": gamma,
        "eta": eta,
        "delta": delta,
        "truth_model": truth_model,
        "K": K,
    }

    return HierarchicalDataset(
        seasons=seasons,
        true_params=true_params,
        phi_k=phi_k,
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
        gamma=float(hier_cfg["gamma"]),
        eta=float(hier_cfg["eta"]),
        delta=float(hier_cfg["delta"]),
        truth_model=hier_cfg["truth_model"],
        n_bins=int(bins_cfg["n_bins"]),
        E_min=float(bins_cfg["E_min"]),
        E_max=float(bins_cfg["E_max"]),
        rng=rng,
    )
    print(f"\nShared params: gamma={hds.true_params['gamma']}, "
          f"eta={hds.true_params['eta']:.2e}, delta={hds.true_params['delta']}")
    print(f"Population: mu_phi={hds.true_params['mu_phi']:.2f}, "
          f"sigma_phi={hds.true_params['sigma_phi']:.2f}")
    print(f"Season fluxes phi_k: {hds.phi_k}")
    print(f"\nPer season total counts:")
    for k, season in enumerate(hds.seasons):
        print(f"  Season {k+1}: {season.counts.sum()} counts "
              f"(phi_k = {hds.phi_k[k]:.3e})")
        
