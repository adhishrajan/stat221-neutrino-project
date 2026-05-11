#!/usr/bin/env python3
"""
Build HESE 7.5y effective-area tables used by this project.
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
CZ_EDGES = np.linspace(-1.0, 1.0, 11)


def load_payload(base_dir: Path) -> dict:
    payload = {}
    for name in ("HESE_mc_truth.json", "HESE_mc_observable.json", "HESE_mc_flux.json"):
        path = base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing HESE MC file: {path}")
        payload.update(json.loads(path.read_text()))
    return payload


def compute_tables(payload: dict) -> dict:
    energy = np.asarray(payload["primaryEnergy"], dtype=float)
    cosz = np.cos(np.asarray(payload["primaryZenith"], dtype=float))
    ptype = np.asarray(payload["primaryType"], dtype=int)
    w = np.asarray(payload["weightOverFluxOverLivetime"], dtype=float)

    mask = np.isin(np.abs(ptype), list(NU_ABS_PDGS)) & np.isfinite(w) & (w > 0)
    energy, cosz, w = energy[mask], cosz[mask], w[mask]

    n_e = len(E_EDGES) - 1
    n_cz = len(CZ_EDGES) - 1
    i_e = np.digitize(energy, E_EDGES) - 1
    i_cz = np.digitize(cosz, CZ_EDGES) - 1
    valid = (0 <= i_e) & (i_e < n_e) & (0 <= i_cz) & (i_cz < n_cz)
    i_e, i_cz, w = i_e[valid], i_cz[valid], w[valid]

    sum_w = np.zeros((n_e, n_cz), dtype=float)
    sum_w2 = np.zeros((n_e, n_cz), dtype=float)
    n_mc = np.zeros((n_e, n_cz), dtype=int)
    np.add.at(sum_w, (i_e, i_cz), w)
    np.add.at(sum_w2, (i_e, i_cz), w * w)
    np.add.at(n_mc, (i_e, i_cz), 1)

    dE = np.diff(E_EDGES)[:, None]
    dOmega = 2.0 * np.pi * np.diff(CZ_EDGES)[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        aeff_cm2 = np.where(dE * dOmega > 0, sum_w / (dE * dOmega), 0.0)
        aeff_err_cm2 = np.where(dE * dOmega > 0, np.sqrt(sum_w2) / (dE * dOmega), 0.0)

    aeff_m2 = aeff_cm2 / 1e4
    aeff_err_m2 = aeff_err_cm2 / 1e4

    dE_1d = np.diff(E_EDGES)
    with np.errstate(divide="ignore", invalid="ignore"):
        aeff_1d_m2 = np.where(dE_1d > 0, np.sum(sum_w, axis=1) / (dE_1d * 4.0 * np.pi) / 1e4, 0.0)
        aeff_1d_err_m2 = np.where(
            dE_1d > 0, np.sqrt(np.sum(sum_w2, axis=1)) / (dE_1d * 4.0 * np.pi) / 1e4, 0.0
        )

    return {
        "aeff_m2": aeff_m2,
        "aeff_err_m2": aeff_err_m2,
        "aeff_1d_m2": aeff_1d_m2,
        "aeff_1d_err_m2": aeff_1d_err_m2,
        "n_mc": n_mc,
    }


def write_tables(outdir: Path, t: dict) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    rows_2d = []
    for ie in range(len(E_EDGES) - 1):
        for ic in range(len(CZ_EDGES) - 1):
            rows_2d.append(
                [
                    E_EDGES[ie],
                    E_EDGES[ie + 1],
                    CZ_EDGES[ic],
                    CZ_EDGES[ic + 1],
                    t["aeff_m2"][ie, ic],
                    t["aeff_err_m2"][ie, ic],
                    t["n_mc"][ie, ic],
                ]
            )
    np.savetxt(
        outdir / "hese75_effective_area_trueE_truecosz.txt",
        np.asarray(rows_2d),
        header="E_lo_GeV E_hi_GeV cosz_lo cosz_hi Aeff_m2 Aeff_err_m2 Nmc",
        fmt=["%.8e", "%.8e", "%.6f", "%.6f", "%.8e", "%.8e", "%d"],
    )

    rows_1d = np.c_[E_EDGES[:-1], E_EDGES[1:], t["aeff_1d_m2"], t["aeff_1d_err_m2"]]
    np.savetxt(
        outdir / "hese75_effective_area_trueE_allsky.txt",
        rows_1d,
        header="E_lo_GeV E_hi_GeV Aeff_m2 Aeff_err_m2",
        fmt="%.8e",
    )


def write_plots(outdir: Path, t: dict) -> None:
    e_ctr = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    cz_ctr = 0.5 * (CZ_EDGES[:-1] + CZ_EDGES[1:])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.step(E_EDGES[1:], t["aeff_1d_m2"], where="post", label="HESE 7.5y all-sky")
    ax.errorbar(e_ctr, t["aeff_1d_m2"], yerr=t["aeff_1d_err_m2"], fmt="none", alpha=0.7)
    ax.set(xscale="log", yscale="log", xlabel="True neutrino energy [GeV]", ylabel="Effective area [m$^2$]")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "hese75_aeff_allsky.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.8))
    z = np.maximum(t["aeff_m2"], 1e-12)
    pcm = ax.pcolormesh(E_EDGES, CZ_EDGES, z.T, shading="auto", norm=LogNorm())
    ax.set(xscale="log", xlabel="True neutrino energy [GeV]", ylabel="cos(zenith)")
    fig.colorbar(pcm, ax=ax, label="Aeff [m$^2$]")
    fig.tight_layout()
    fig.savefig(outdir / "hese75_aeff_trueE_truecosz.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for target in (-0.75, -0.25, 0.25, 0.75):
        j = int(np.argmin(np.abs(cz_ctr - target)))
        ax.step(E_EDGES[1:], t["aeff_m2"][:, j], where="post", label=f"cosz~{cz_ctr[j]:.2f}")
    ax.set(xscale="log", yscale="log", xlabel="True neutrino energy [GeV]", ylabel="Effective area [m$^2$]")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "hese75_aeff_zenith_slices.png", dpi=180)
    plt.close(fig)


def write_summary(outdir: Path, t: dict) -> None:
    y = t["aeff_1d_m2"]
    e_ctr = np.sqrt(E_EDGES[:-1] * E_EDGES[1:])
    peak_idx = int(np.argmax(y))
    nonzero = y > 0
    text = "\n".join(
        [
            "HESE 7.5-year Effective Area Sanity Summary",
            "============================================",
            f"Energy bins: {len(y)}",
            f"Nonzero 1D bins: {int(nonzero.sum())}/{len(y)} ({100*np.mean(nonzero):.1f}%)",
            f"Peak Aeff_1D: {y[peak_idx]:.3e} m^2 at E~{e_ctr[peak_idx]:.3e} GeV",
            f"Aeff_1D min/max: {np.min(y):.3e} / {np.max(y):.3e} m^2",
            f"- finite values: {bool(np.all(np.isfinite(y)))}",
            f"- non-negative: {bool(np.all(y >= 0))}",
            f"- has high-energy support (>1e6 GeV): {bool(np.any((e_ctr > 1e6) & (y > 0)))}",
            "",
        ]
    )
    (outdir / "hese75_aeff_summary.txt").write_text(text)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hese75-data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = p.parse_args()

    payload = load_payload(args.hese75_data_dir)
    tables = compute_tables(payload)
    write_tables(args.outdir, tables)
    write_plots(args.outdir, tables)
    write_summary(args.outdir, tables)
    print(f"Wrote HESE 7.5 derived Aeff products to: {args.outdir}")


if __name__ == "__main__":
    main()

