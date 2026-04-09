"""
Phase 1A: Data Simulator for Neutrino Poisson Mixture Models
=============================================================
Generates synthetic bin counts under multiple truth models:
  - Single power law
  - Broken power law
  - Power law with exponential cutoff
  - Two-component astrophysical + prompt atmospheric

Supports both single-season and hierarchical multi-season generation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Energy bin configuration ──────────────────────────────────────────────

def make_energy_bins(n_bins: int = 20,
                     E_min: float = 1e4,    # 10 TeV in GeV
                     E_max: float = 1e7     # 10 PeV in GeV
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create log-spaced energy bins.

    Returns
    -------
    E_edges : (n_bins+1,) bin edges
    E_centers : (n_bins,) geometric center of each bin
    E_widths : (n_bins,) width of each bin (used for rate integration)
    """
    E_edges = np.logspace(np.log10(E_min), np.log10(E_max), n_bins + 1)
    E_centers = np.sqrt(E_edges[:-1] * E_edges[1:])   # geometric mean
    E_widths = E_edges[1:] - E_edges[:-1]
    return E_edges, E_centers, E_widths


# ── Signal models ─────────────────────────────────────────────────────────

E_REF = 1e5  # Reference energy: 100 TeV in GeV (standard IceCube pivot)


def signal_power_law(E: np.ndarray, phi: float, gamma: float,
                     E_widths: np.ndarray) -> np.ndarray:
    """
    Single power-law signal: mu_i^signal = phi * (E_i/E_ref)^{-gamma} * dE_i

    Using normalized energy (E/E_ref) keeps phi at a human-readable scale.
    With E_ref = 100 TeV and IceCube-like parameters, phi ~ O(1-100) gives
    realistic event counts.
    """
    return phi * (E / E_REF)**(-gamma) * (E_widths / E_REF)


def signal_broken_power_law(E: np.ndarray, phi: float,
                            gamma1: float, gamma2: float,
                            E_break: float,
                            E_widths: np.ndarray) -> np.ndarray:
    """
    Broken power law: slope gamma1 below E_break, gamma2 above.
    Continuous at E_break. Uses normalized energy.
    """
    E_norm = E / E_REF
    E_break_norm = E_break / E_REF
    # Continuity: phi * E_break_norm^{-gamma1} = A * E_break_norm^{-gamma2}
    A = phi * E_break_norm**(gamma2 - gamma1)
    mu = np.where(
        E < E_break,
        phi * E_norm**(-gamma1),
        A * E_norm**(-gamma2)
    )
    return mu * (E_widths / E_REF)


def signal_cutoff(E: np.ndarray, phi: float, gamma: float,
                  E_cut: float, E_widths: np.ndarray) -> np.ndarray:
    """
    Power law with exponential cutoff:
      phi * (E/E_ref)^{-gamma} * exp(-E / E_cut) * dE/E_ref
    """
    return phi * (E / E_REF)**(-gamma) * np.exp(-E / E_cut) * (E_widths / E_REF)


def signal_two_component(E: np.ndarray,
                         phi1: float, gamma1: float,
                         phi2: float, gamma2: float,
                         E_widths: np.ndarray) -> np.ndarray:
    """
    Two astrophysical components, each a separate power law.
    """
    E_norm = E / E_REF
    dE_norm = E_widths / E_REF
    return (phi1 * E_norm**(-gamma1) + phi2 * E_norm**(-gamma2)) * dE_norm


# ── Background model ──────────────────────────────────────────────────────

def background_atmospheric(E: np.ndarray, eta: float, delta: float,
                           E_widths: np.ndarray) -> np.ndarray:
    """
    Atmospheric background: eta * (E/E_ref)^{-delta} * dE/E_ref
    """
    return eta * (E / E_REF)**(-delta) * (E_widths / E_REF)


# ── Data containers ───────────────────────────────────────────────────────

@dataclass
class Dataset:
    """Container for a single-season simulated dataset."""
    counts: np.ndarray          # (B,) observed Poisson counts
    E_centers: np.ndarray       # (B,) bin centers
    E_edges: np.ndarray         # (B+1,) bin edges
    E_widths: np.ndarray        # (B,) bin widths
    mu_signal: np.ndarray       # (B,) true expected signal counts
    mu_background: np.ndarray   # (B,) true expected background counts
    true_params: dict           # true parameter values used to generate


@dataclass
class HierarchicalDataset:
    """Container for a multi-season hierarchical dataset."""
    seasons: list[Dataset]      # list of K single-season datasets
    true_params: dict           # shared + population-level true params
    phi_k: np.ndarray           # (K,) season-specific true fluxes


# ── Main generation functions ─────────────────────────────────────────────

def generate_single_season(
    phi: float = 30.0,
    gamma: float = 2.5,
    eta: float = 500.0,
    delta: float = 3.7,
    truth_model: str = "power_law",
    n_bins: int = 20,
    E_min: float = 1e4,
    E_max: float = 1e7,
    rng: Optional[np.random.Generator] = None,
    # Extra params for alternative truth models
    gamma2: float = 3.0,       # broken power law: slope above break
    E_break: float = 5e5,      # broken power law: break energy (GeV)
    E_cut: float = 3e6,        # cutoff: cutoff energy (GeV)
    phi2: float = 15.0,        # two-component: second flux norm
    gamma2_alt: float = 2.0,   # two-component: second spectral index
) -> Dataset:
    """
    Generate a single-season synthetic dataset.

    Parameters
    ----------
    phi : float
        Flux normalization for the astrophysical signal.
    gamma : float
        Spectral index for the astrophysical signal.
    eta : float
        Background normalization.
    delta : float
        Background spectral index (atmospheric).
    truth_model : str
        One of: 'power_law', 'broken_power_law', 'cutoff', 'two_component'
    n_bins : int
        Number of energy bins.
    rng : numpy random Generator, optional
        For reproducibility.

    Returns
    -------
    Dataset
    """
    if rng is None:
        rng = np.random.default_rng()

    E_edges, E_centers, E_widths = make_energy_bins(n_bins, E_min, E_max)

    # Compute signal expected counts under the chosen truth model
    if truth_model == "power_law":
        mu_signal = signal_power_law(E_centers, phi, gamma, E_widths)
    elif truth_model == "broken_power_law":
        mu_signal = signal_broken_power_law(
            E_centers, phi, gamma, gamma2, E_break, E_widths
        )
    elif truth_model == "cutoff":
        mu_signal = signal_cutoff(E_centers, phi, gamma, E_cut, E_widths)
    elif truth_model == "two_component":
        mu_signal = signal_two_component(
            E_centers, phi, gamma, phi2, gamma2_alt, E_widths
        )
    else:
        raise ValueError(f"Unknown truth model: {truth_model}")

    # Background
    mu_background = background_atmospheric(E_centers, eta, delta, E_widths)

    # Total expected counts
    mu_total = mu_signal + mu_background

    # Sanity: all rates should be non-negative
    assert np.all(mu_total >= 0), f"Negative rates detected: {mu_total}"

    # Draw Poisson counts
    counts = rng.poisson(mu_total)

    true_params = {
        "phi": phi, "gamma": gamma, "eta": eta, "delta": delta,
        "truth_model": truth_model,
    }
    if truth_model == "broken_power_law":
        true_params.update({"gamma2": gamma2, "E_break": E_break})
    elif truth_model == "cutoff":
        true_params.update({"E_cut": E_cut})
    elif truth_model == "two_component":
        true_params.update({"phi2": phi2, "gamma2": gamma2_alt})

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
    K: int = 10,
    mu_phi: float = np.log(30.0),
    sigma_phi: float = 0.3,
    gamma: float = 2.5,
    eta: float = 500.0,
    delta: float = 3.7,
    truth_model: str = "power_law",
    n_bins: int = 20,
    E_min: float = 1e4,
    E_max: float = 1e7,
    rng: Optional[np.random.Generator] = None,
    **truth_model_kwargs,
) -> HierarchicalDataset:
    """
    Generate a hierarchical multi-season dataset.

    Each season k has its own flux normalization phi_k drawn from
    log(phi_k) ~ N(mu_phi, sigma_phi^2), but shares gamma, eta, delta.

    Parameters
    ----------
    K : int
        Number of observation seasons.
    mu_phi : float
        Population mean of log(phi_k).
    sigma_phi : float
        Population std dev of log(phi_k).
    gamma, eta, delta : float
        Shared parameters.
    truth_model : str
        Signal model for all seasons.

    Returns
    -------
    HierarchicalDataset
    """
    if rng is None:
        rng = np.random.default_rng()

    # Draw season-specific fluxes
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


# ── Convenience: quick summary ───────────────────────────────────────────

def summarize_dataset(ds: Dataset) -> None:
    """Print a quick summary of a single-season dataset."""
    total_signal = ds.mu_signal.sum()
    total_bg = ds.mu_background.sum()
    total_counts = ds.counts.sum()
    snr = total_signal / total_bg if total_bg > 0 else float('inf')

    print(f"Truth model: {ds.true_params['truth_model']}")
    print(f"  Bins: {len(ds.counts)}")
    print(f"  Energy range: [{ds.E_edges[0]:.1e}, {ds.E_edges[-1]:.1e}] GeV")
    print(f"  Expected signal counts: {total_signal:.1f}")
    print(f"  Expected background counts: {total_bg:.1f}")
    print(f"  Signal-to-noise ratio: {snr:.3f}")
    print(f"  Observed total counts: {total_counts}")
    print(f"  True params: {ds.true_params}")


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    print("=" * 60)
    print("SINGLE-SEASON DATASETS")
    print("=" * 60)

    for model in ["power_law", "broken_power_law", "cutoff", "two_component"]:
        print(f"\n--- {model} ---")
        ds = generate_single_season(truth_model=model, rng=rng)
        summarize_dataset(ds)

    print("\n" + "=" * 60)
    print("HIERARCHICAL DATASET (K=10 seasons)")
    print("=" * 60)

    hds = generate_hierarchical(K=10, rng=rng)
    print(f"\nShared params: gamma={hds.true_params['gamma']}, "
          f"eta={hds.true_params['eta']:.2e}, delta={hds.true_params['delta']}")
    print(f"Population: mu_phi={hds.true_params['mu_phi']:.2f}, "
          f"sigma_phi={hds.true_params['sigma_phi']:.2f}")
    print(f"Season fluxes phi_k: {hds.phi_k}")
    print(f"\nPer-season total counts:")
    for k, season in enumerate(hds.seasons):
        print(f"  Season {k+1}: {season.counts.sum()} counts "
              f"(phi_k = {hds.phi_k[k]:.3e})")