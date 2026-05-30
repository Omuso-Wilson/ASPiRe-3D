"""
ASPiRe-3D : postprocessing/adsorption_plots.py
===============================================================================
Visualization for adsorption / retention (Phase 5):
  * adsorbed surfactant / retained polymer concentration maps and profiles,
  * concentration retardation (adsorbing vs conservative breakthrough),
  * pressure-drop buildup during retention,
  * mobility-ratio evolution.
All Matplotlib (headless), saved to disk for thesis inclusion.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_retained_profile(sim, mesh, species_name, outfile, ylabel=None):
    """Slice-averaged axial profile of an immobile retained/adsorbed species."""
    C = sim.concentration_grid(species_name)
    prof = np.array([np.nanmean(C[i, :, :][mesh.active[i, :, :]])
                     for i in range(mesh.nx)])
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(mesh.xc, prof, "-", lw=2, color="darkgreen")
    ax.fill_between(mesh.xc, 0, prof, alpha=0.2, color="darkgreen")
    ax.set_xlabel("Axial position x  [m]")
    ax.set_ylabel(ylabel or f"{species_name}  [kg/m³ bulk]")
    ax.set_title(f"Retention profile: {species_name}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_retardation(pv_cons, out_cons, pv_ads, out_ads, outfile,
                     label_cons="conservative", label_ads="adsorbing"):
    """Overlay conservative vs adsorbing breakthrough to show retardation."""
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(pv_cons, out_cons, "-", lw=2, color="steelblue", label=label_cons)
    ax.plot(pv_ads, out_ads, "-", lw=2, color="firebrick", label=label_ads)
    ax.axvline(1.0, color="k", ls=":", lw=1, label="1 pore volume")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Pore volumes injected  [-]")
    ax.set_ylabel("Outlet C / C_inj  [-]")
    ax.set_title("Concentration retardation from adsorption")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.02, 1.05)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_pressure_buildup(sim, outfile):
    """Differential pressure vs pore volumes (retention/damage buildup)."""
    pv = np.array(sim.history["pv"]); dP = np.array(sim.history["dP"])
    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(pv, dP / 1e3, "-", lw=2, color="crimson")
    ax.set_xlabel("Pore volumes injected  [-]")
    ax.set_ylabel("Differential pressure  Δp  [kPa]")
    ax.set_title("Pressure buildup during retention / damage")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_mobility_evolution(sim, mesh, properties, outfile,
                            polymer_name="polymer", visco_factor=10.0):
    """
    Mobility-ratio surrogate vs pore volumes. Polymer raises aqueous viscosity,
    lowering mobility (k/mu); we report a normalised mobility index
        M_index = mean( (k/k0) / (1 + visco_factor * polymer) )
    falling below 1 indicates improved (lower) mobility -> better sweep, the
    intended polymer effect, convolved here with permeability damage.
    """
    pv = np.array(sim.history["pv"])
    kmean = np.array(sim.history["k_ratio_mean"])
    # Use final polymer field as a representative mobility snapshot trend proxy:
    # build an index time series from k_ratio_mean and outlet polymer.
    if polymer_name in sim.registry.names:
        poly_out = np.array(sim.history["outlet"][polymer_name])
    else:
        poly_out = np.zeros_like(pv)
    M_index = kmean / (1.0 + visco_factor * poly_out)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(pv, M_index, "-", lw=2, color="purple")
    ax.axhline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("Pore volumes injected  [-]")
    ax.set_ylabel("Mobility index  [-]")
    ax.set_title("Mobility evolution (polymer + damage)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_adsorbed_map(sim, mesh, species_name, outfile, cmap="YlOrBr"):
    """Longitudinal (mid-z) map of an adsorbed/retained species."""
    C = sim.concentration_grid(species_name)
    k_mid = mesh.nz // 2
    field = np.ma.masked_invalid(C[:, :, k_mid].T)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(field, origin="lower", aspect="auto",
                   extent=[0, mesh.Lx, 0, mesh.Ly], cmap=cmap)
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"{species_name} (adsorbed/retained)")
    cbar = fig.colorbar(im, ax=ax); cbar.set_label("kg/m³ bulk")
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile
