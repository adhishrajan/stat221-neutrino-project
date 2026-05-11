#!/usr/bin/env python3
"""
Build HESE 7.5y migration/energy-resolution products used by this project.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


NU_ABS_PDGS = {12, 14, 16}
DEFAULT_DATA_DIR = Path(
    "data/icecube/hese75/HESE-7-year-data-release-main/HESE-7-year-data-release/resources/data"
)
DEFAULT_OUTDIR = Path("data/icecube/hese75/derived")
E_EDGES = np.logspace(4, 7, 21)


def load_payload(base_dir: Path) -> dict:
    payload = {}
    for name in ("HESE_mc_truth.json", "HESE_mc_observable.json", "HESE_mc_flux.json"):
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing HESE MC file: {path}")
        payload.update(json.loads(path.read_text()))
    return payload


def weighted_mean_var(x: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    sw = np.sum(w)
    if sw <= 0:
        return np.nan, np.nan
    mu = np.sum(w * x) / sw
    var = np.sum(w * (x - mu) ** 2) / sw
    return float(mu), float(var)


def select_mask(payload: dict, morphology: str) -> np.ndarray:
    ptype = np.asarray(payload["primaryType"], dtype=int)
    e_true = np.asarray(payload["primaryEnergy"], dtype=float)
    e_reco = np.asarray(payload["recoDepositedEnergy"], dtype=float)
    w = np.asarray(payload["weightOverFluxOverLivetime"], dtype=float)
    reco_morph = np.asarray(payload["recoMorphology"], dtype=int)

    mask = (
        np.isin(np.abs(ptype), list(NU_ABS_PDGS))
        & np.isfinite(e_true)
        & np.isfinite(e_reco)
        & np.isfinite(w)
        & (e_true > 0)
        & (e_reco > 0)
        & (w > 0)
    )
    if morphology == "track":
        mask &= reco_morph == 1
    elif morphology == "cascade":
        mask &= reco_morph == 0
    elif morphology == "double_cascade":
        mask &= reco_morph == 2
    return mask


def build_products(payload: dict, morphology: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    e_true = np.asarray(payload["primaryEnergy"], dtype=float)
    e_reco = np.asarray(payload["recoDepositedEnergy"], dtype=float)
    w = np.asarray(payload["weightOverFluxOverLivetime"], dtype=float)
    mask = select_mask(payload, morphology)
    e_true, e_reco, w = e_true[mask], e_reco[mask], w[mask]

    n = len(E_EDGES) - 1
    i_t = np.digitize(e_true, E_EDGES) - 1
    i_r = np.digitize(e_reco, E_EDGES) - 1
    valid = (0 <= i_t) & (i_t < n) & (0 <= i_r) & (i_r < n)
    i_t, i_r, w = i_t[valid], i_r[valid], w[valid]
    e_true, e_reco = e_true[valid], e_reco[valid]

    M = np.zeros((n, n), dtype=float)  # reco x true
    np.add.at(M, (i_r, i_t), w)
    col_sum = M.sum(axis=0)
    P = np.eye(n)
    nz = col_sum > 0
    P[:, nz] = M[:, nz] / col_sum[nz]

    ratio_log10 = np.log10(e_reco / e_true)
    e_ctr = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    bias = np.full(n, np.nan)
    sigma = np.full(n, np.nan)
    n_eff = np.zeros(n, dtype=float)
    for j in range(n):
        m = i_t == j
        if not np.any(m):
            continue
        mu, var = weighted_mean_var(ratio_log10[m], w[m])
        bias[j] = mu
        sigma[j] = np.sqrt(max(var, 0.0))
        sw = np.sum(w[m])
        sw2 = np.sum(w[m] ** 2)
        n_eff[j] = (sw * sw / sw2) if sw2 > 0 else 0.0
    return P, e_ctr, bias, sigma, n_eff, len(w)


def write_outputs(
    outdir: Path,
    morphology: str,
    P: np.ndarray,
    e_ctr: np.ndarray,
    bias: np.ndarray,
    sigma: np.ndarray,
    n_eff: np.ndarray,
    n_used: int,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    n = len(E_EDGES) - 1

    rows = []
    for j in range(n):
        for i in range(n):
            rows.append([E_EDGES[j], E_EDGES[j + 1], E_EDGES[i], E_EDGES[i + 1], P[i, j]])
    np.savetxt(
        outdir / f"hese75_migration_{morphology}.txt",
        np.asarray(rows),
        header="Etrue_lo Etrue_hi Ereco_lo Ereco_hi P(Ereco|Etrue)",
        fmt="%.8e",
    )

    prof = np.c_[E_EDGES[:-1], E_EDGES[1:], e_ctr, bias, sigma, n_eff]
    np.savetxt(
        outdir / f"hese75_resolution_profile_{morphology}.txt",
        prof,
        header="Etrue_lo Etrue_hi Etrue_center bias_log10 sigma_log10 n_eff",
        fmt="%.8e",
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(e_ctr, sigma, marker="o", label=morphology)
    ax.set(xscale="log", xlabel="True neutrino energy [GeV]", ylabel=r"$\sigma[\log_{10}(E_{\mathrm{reco}}/E_{\mathrm{true}})]$")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"hese75_resolution_sigma_{morphology}.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    z = np.maximum(P, 1e-8)
    pcm = ax.pcolormesh(E_EDGES, E_EDGES, z, shading="auto", norm=LogNorm())
    ax.set(xscale="log", yscale="log", xlabel="True neutrino energy [GeV]", ylabel="Reco deposited energy [GeV]")
    fig.colorbar(pcm, ax=ax, label=r"$P(E_{\mathrm{reco}}|E_{\mathrm{true}})$")
    fig.tight_layout()
    fig.savefig(outdir / f"hese75_migration_{morphology}.png", dpi=180)
    plt.close(fig)

    valid_sigma = np.isfinite(sigma)
    lines = [
        "HESE 7.5-year Energy Resolution Summary",
        "=====================================",
        f"Morphology: {morphology}",
        f"Bins with finite sigma: {int(valid_sigma.sum())}/{len(sigma)}",
        f"Events used (weighted entries): {n_used}",
    ]
    if np.any(valid_sigma):
        lines.insert(-1, f"Median sigma_log10: {np.nanmedian(sigma):.4f}")
        lines.insert(-1, f"Min/Max sigma_log10: {np.nanmin(sigma):.4f} / {np.nanmax(sigma):.4f}")
    (outdir / f"hese75_resolution_summary_{morphology}.txt").write_text("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hese75-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--morphology", choices=["all", "track", "cascade", "double_cascade"], default="all")
    args = p.parse_args()

    payload = load_payload(args.hese75_data_dir)
    P, e_ctr, bias, sigma, n_eff, n_used = build_products(payload, args.morphology)
    write_outputs(args.outdir, args.morphology, P, e_ctr, bias, sigma, n_eff, n_used)
    print(f"Wrote resolution products to: {args.outdir}")


if __name__ == "__main__":
    main()
