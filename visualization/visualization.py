"""
ASPiRe-3D : postprocessing/visualization.py
===============================================================================
2D slice visualization of 3D fields using Matplotlib (publication-oriented).

We use Matplotlib (not PyVista) in Phase 1 to keep dependencies minimal and
outputs reproducible on any machine. PyVista 3D rendering is an optional later
add-on. All figures are saved to disk so they can drop straight into a thesis.

Convention: the flow axis is x. The most informative views are:
  * a longitudinal slice (x-y plane at mid-z) showing pressure decline along
    the core, and
  * an axial cross-section (y-z plane) showing the circular core footprint.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless backend: safe on servers/CI
import matplotlib.pyplot as plt


def plot_pressure_longitudinal(mesh, pressure, outfile, bc_label=""):
    """
    Longitudinal (x-y) slice of the pressure field at the mid-z plane.
    Inactive cells (outside the cylinder) are masked so the core footprint is
    visually distinct.
    """
    k_mid = mesh.nz // 2
    field = pressure[:, :, k_mid].T          # transpose so x is horizontal
    field = np.ma.masked_invalid(field)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(field, origin="lower", aspect="auto",
                   extent=[0, mesh.Lx, 0, mesh.Ly], cmap="viridis")
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"Pressure field, longitudinal slice (mid-z)  {bc_label}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pressure [Pa]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_pressure_cross_section(mesh, pressure, outfile, i_slice=None):
    """
    Axial (y-z) cross-section of pressure at an x-slice (default: inlet, i=0),
    showing the circular core footprint and any in-plane pressure variation.
    """
    if i_slice is None:
        i_slice = 0
    field = pressure[i_slice, :, :].T
    field = np.ma.masked_invalid(field)

    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    im = ax.imshow(field, origin="lower",
                   extent=[0, mesh.Ly, 0, mesh.Lz], cmap="viridis")
    ax.set_xlabel("y  [m]")
    ax.set_ylabel("z  [m]")
    ax.set_aspect("equal")
    ax.set_title(f"Pressure cross-section (x-slice i={i_slice})")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pressure [Pa]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_concentration_snapshots(mesh, snapshots, outfile):
    """
    Multi-panel longitudinal (x-y, mid-z) snapshots of the tracer field at a
    sequence of pore-volume times, showing the front advancing through the
    core. `snapshots` is a list of (pv_label, concentration_grid) tuples.

    This is the canonical "front propagation" figure for a tracer core flood
    and the visual companion to the breakthrough curve.
    """
    n = len(snapshots)
    k_mid = mesh.nz // 2
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 1.6 * n + 0.6), squeeze=False)
    for ax, (label, C) in zip(axes[:, 0], snapshots):
        field = np.ma.masked_invalid(C[:, :, k_mid].T)
        im = ax.imshow(field, origin="lower", aspect="auto",
                       extent=[0, mesh.Lx, 0, mesh.Ly], cmap="magma",
                       vmin=0.0, vmax=1.0)
        ax.set_ylabel("y [m]")
        ax.set_title(f"{label}", fontsize=9, loc="left")
        ax.tick_params(labelbottom=False)
    axes[-1, 0].tick_params(labelbottom=True)
    axes[-1, 0].set_xlabel("x  [m]  (flow direction)")
    cbar = fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8)
    cbar.set_label("C / C_inj  [-]")
    fig.suptitle("Tracer front propagation (mid-z slices)", y=0.99)
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return outfile


def plot_concentration_longitudinal(mesh, concentration, outfile, time_label=""):
    """
    Longitudinal (x-y) slice of the tracer concentration field at mid-z, with
    inactive cells masked. Shows the advancing tracer front along the core.
    """
    k_mid = mesh.nz // 2
    field = concentration[:, :, k_mid].T
    field = np.ma.masked_invalid(field)

    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(field, origin="lower", aspect="auto",
                   extent=[0, mesh.Lx, 0, mesh.Ly], cmap="magma",
                   vmin=0.0, vmax=1.0)
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title(f"Tracer concentration, longitudinal slice (mid-z) {time_label}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("C / C_inj  [-]")
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile


def plot_velocity_quiver(mesh, ux, uy, outfile, stride=2):
    """
    In-plane Darcy velocity quiver on the mid-z longitudinal slice. For a clean
    axial flood the vectors should point uniformly along +x.
    """
    k_mid = mesh.nz // 2
    X, Y = np.meshgrid(mesh.xc, mesh.yc, indexing="ij")
    U = np.ma.masked_invalid(np.where(mesh.active[:, :, k_mid], ux[:, :, k_mid], np.nan))
    V = np.ma.masked_invalid(np.where(mesh.active[:, :, k_mid], uy[:, :, k_mid], np.nan))

    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.quiver(X[::stride, ::stride], Y[::stride, ::stride],
              U[::stride, ::stride], V[::stride, ::stride],
              scale_units="xy", angles="xy")
    ax.set_xlabel("x  [m]  (flow direction)")
    ax.set_ylabel("y  [m]")
    ax.set_title("Darcy velocity field (mid-z slice)")
    ax.set_xlim(0, mesh.Lx); ax.set_ylim(0, mesh.Ly)
    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    plt.close(fig)
    return outfile
