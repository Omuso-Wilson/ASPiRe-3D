"""
ASPiRe-3D : postprocessing/damage_plots.py
===============================================================================
Visualization for coupled formation-damage reactive transport (Phase 4):
  * injectivity decline curve (the headline formation-damage result),
  * permeability and porosity evolution vs pore volumes,
  * formation damage index (FDI) longitudinal map,
  * precipitate / deposited-fines concentration maps,
  * salinity & species axial profiles,
  * reaction-stiffness and stability diagnostics.

All Matplotlib (headless), saved to disk for direct thesis inclusion.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_injectivity_decline(sim, outfile):
    """Injectivity ratio (I/I0) and normalised dP vs pore volumes injected."""
    pv = np.array(sim.history["pv"])
    inj = np.array(sim.history["injectivity_ratio"])
    dP = np.array(sim.history["dP"]); dP0 = dP[0] if len(dP) else 1.0

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(pv, inj, "b-", lw=2, label="Injectivity I/I₀")
    ax1.set_xlabel("Pore volumes injected  [-]")
    ax1.set_ylabel("Injectivity ratio  I/I₀  [-]", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.set_ylim(0, 1.05)
    ax2 = ax1.twinx()
    ax2.plot(pv, dP / dP0, "r--", lw=2, label="Δp/Δp₀")
    ax2.set_ylabel("Normalised Δp  [-]", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title("Injectivity decline (formation damage)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_permeability_porosity_evolution(sim, outfile):
    """Minimum/mean k/k0 and minimum porosity vs pore volumes."""
    pv = np.array(sim.history["pv"])
    kmin = np.array(sim.history["k_ratio_min"])
    kmean = np.array(sim.history["k_ratio_mean"])
    phimin = np.array(sim.history["phi_min"])

    fig, ax1 = plt.subplots(figsize=(7.5, 4.2))
    ax1.plot(pv, kmin, "-", color="darkred", lw=2, label="k/k₀ (min)")
    ax1.plot(pv, kmean, "--", color="salmon", lw=2, label="k/k₀ (mean)")
    ax1.set_xlabel("Pore volumes injected  [-]")
    ax1.set_ylabel("Permeability ratio  k/k₀  [-]")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="upper right")
    ax2 = ax1.twinx()
    ax2.plot(pv, phimin, ":", color="navy", lw=2, label="φ (min)")
    ax2.set_ylabel("Minimum porosity  φ  [-]", color="navy")
    ax2.tick_params(axis="y", labelcolor="navy")
    ax1.set_title("Permeability & porosity evolution")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_damage_index_map(mesh, damage, outfile):
    """Longitudinal (mid-z) map of the Formation Damage Index FDI = 1 - k/k0."""
    FDI = damage.damage_index_grid()
    k_mid = mesh.nz // 2
    field = np.ma.masked_invalid(FDI[:, :, k_mid].T)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(field, origin="lower", aspect="auto",
                   extent=[0, mesh.Lx, 0, mesh.Ly], cmap="inferno",
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title("Formation Damage Index  (FDI = 1 − k/k₀)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("FDI  [-]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_species_map(sim, mesh, species_name, outfile, cmap="viridis",
                     title=None):
    """Longitudinal (mid-z) concentration map of one species."""
    C = sim.concentration_grid(species_name)
    k_mid = mesh.nz // 2
    field = np.ma.masked_invalid(C[:, :, k_mid].T)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(field, origin="lower", aspect="auto",
                   extent=[0, mesh.Lx, 0, mesh.Ly], cmap=cmap)
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title(title or f"{species_name} concentration")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label(f"{species_name}  [-]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_axial_species_profiles(sim, mesh, species_names, outfile):
    """Slice-averaged axial concentration profiles for several species."""
    x = mesh.xc
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for name in species_names:
        C = sim.concentration_grid(name)
        prof = np.array([np.nanmean(C[i, :, :][mesh.active[i, :, :]])
                         for i in range(mesh.nx)])
        ax.plot(x, prof, lw=2, label=name)
    ax.set_xlabel("Axial position x  [m]")
    ax.set_ylabel("Concentration  [-]")
    ax.set_title("Axial species profiles (final state)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_stability_diagnostics(sim, outfile):
    """Courant number and reaction-stiffness (Damkohler) vs pore volumes."""
    pv = np.array(sim.history["pv"])
    courant = np.array(sim.history["courant"])
    Da = np.array(sim.history["stiffness_Da"])
    fig, ax1 = plt.subplots(figsize=(7.5, 4.0))
    ax1.plot(pv, courant, "g-", lw=2, label="Courant number")
    ax1.axhline(1.0, color="g", ls=":", lw=1)
    ax1.set_xlabel("Pore volumes injected  [-]")
    ax1.set_ylabel("Courant number  [-]", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax2 = ax1.twinx()
    ax2.semilogy(pv, np.maximum(Da, 1e-6), "m-", lw=2, label="Stiffness Da")
    ax2.axhline(1.0, color="m", ls=":", lw=1)
    ax2.set_ylabel("Reaction stiffness  Da  [-]", color="m")
    ax2.tick_params(axis="y", labelcolor="m")
    ax1.set_title("Stability diagnostics (Courant & reaction stiffness)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
