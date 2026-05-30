"""
ASPiRe-3D : postprocessing/plots.py
===============================================================================
Quantitative (line) plots and analytical validation for Phase 1.

The headline validation for a 1D-equivalent core flood is that the simulated
axial pressure profile is LINEAR and matches the analytical Darcy solution.
This is the baseline a thesis examiner expects before any damage physics.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def axial_pressure_profile(mesh, pressure):
    """
    Cross-sectional-average pressure as a function of axial position x.
    Averaging over each x-slice (active cells only) collapses the 3D field to
    the 1D profile that core-flood theory predicts.
    """
    x = mesh.xc.copy()
    p_avg = np.full(mesh.nx, np.nan)
    for i in range(mesh.nx):
        sl = pressure[i, :, :]
        active_slice = mesh.active[i, :, :]
        if np.any(active_slice):
            p_avg[i] = np.nanmean(sl[active_slice])
    return x, p_avg


def plot_axial_profile_with_analytic(mesh, pressure, outfile,
                                     analytic_dp=None):
    """
    Plot the simulated axial pressure profile and, if a constant-rate /
    constant-dP case provides it, overlay the analytical linear Darcy profile

        P(x) = P_in - dP * (x / L)

    A near-perfect overlay confirms the discretization and solver are correct.
    """
    x, p = axial_pressure_profile(mesh, pressure)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, p, "o-", ms=4, label="ASPiRe-3D (slice-averaged)")

    if analytic_dp is not None:
        p_in = np.nanmax(p)
        p_analytic = p_in - analytic_dp * (x / mesh.Lx)
        ax.plot(x, p_analytic, "k--", lw=1.5, label="Analytical Darcy (linear)")

    ax.set_xlabel("Axial position x  [m]")
    ax.set_ylabel("Pressure  [Pa]")
    ax.set_title("Axial pressure profile along core")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_multispecies_breakthrough(reactive, mesh, properties, outfile):
    """
    Plot outlet breakthrough curves for all MOBILE species vs pore volumes
    injected. For an ASP slug displacing resident brine, salinity should fall
    from 1 toward its injected value while alkali/surfactant/polymer rise from
    0 toward 1 -- the classic complementary displacement signature.
    """
    import numpy as np
    pore_volume = float(np.sum(properties.phi[mesh.active]) * mesh.cell_volume)
    Q_total = float(np.sum(reactive._inlet_flux))
    t = np.array(reactive.history["time"])
    pv = t * Q_total / pore_volume

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for s in reactive.registry:
        if not s.mobile:
            continue
        out = np.array(reactive.history["outlet"][s.name])
        ax.plot(pv, out, lw=2, label=s.name)
    ax.axvline(1.0, color="k", ls=":", lw=1, label="1 pore volume")
    ax.set_xlabel("Pore volumes injected  [-]")
    ax.set_ylabel("Outlet concentration  [-]")
    ax.set_title("Multi-species breakthrough (ASP slug, no chemistry yet)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_breakthrough_curve(tracer, velocity, mesh, properties, outfile):
    """
    Plot the outlet breakthrough curve: normalised outlet concentration vs
    pore volumes injected. For a passive tracer the 0.5 crossing should sit
    near 1 PV; the S-curve steepness reflects dispersion + numerical diffusion.
    This is the headline experimental-comparison plot for a tracer core flood.
    """
    import numpy as np
    pore_volume = float(np.sum(properties.phi[mesh.active]) * mesh.cell_volume)
    Q_total = 0.0
    for j in range(mesh.ny):
        for k in range(mesh.nz):
            if mesh.active[0, j, k]:
                Q_total += velocity.ux[0, j, k] * mesh.area_x

    t = np.array(tracer.history["time"])
    outC = np.array(tracer.history["outlet_C"])
    pv = t * Q_total / pore_volume

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pv, outC, "-", lw=2, label="ASPiRe-3D outlet")
    ax.axvline(1.0, color="k", ls=":", lw=1, label="1 pore volume")
    ax.axhline(0.5, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Pore volumes injected  [-]")
    ax.set_ylabel("Outlet C / C_inj  [-]")
    ax.set_title("Tracer breakthrough curve")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def analytical_darcy_dp(mesh, properties, flow_rate, cross_section_area):
    """
    Analytical differential pressure for steady 1D Darcy flow through the core:

        dP = Q * mu * L / (k * A)

    Used to validate the simulator under CONSTANT_RATE. Uses mean active-cell
    k and mu (homogeneous in Phase 1, so the mean is exact).
    """
    act = mesh.active
    k = properties.k[act].mean()
    mu = properties.mu[act].mean()
    return flow_rate * mu * mesh.Lx / (k * cross_section_area)
