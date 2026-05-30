"""
ASPiRe-3D : utils/helpers.py
===============================================================================
Small, dependency-light numerical helpers used across the core modules.

These are deliberately kept tiny and pure (no side effects) so they can be
unit-tested in isolation -- important for a defensible thesis codebase.
===============================================================================
"""

import numpy as np
from .constants import PERMEABILITY_FLOOR_M2


def harmonic_mean(a, b, floor=PERMEABILITY_FLOOR_M2):
    """
    Harmonic mean of two non-negative quantities, used for FACE permeability.

    WHY HARMONIC, NOT ARITHMETIC?
    -----------------------------
    Flow across the face between cell P and cell N passes through half of cell
    P's rock and half of cell N's rock *in series*. Series resistances add, so
    the effective face permeability is the harmonic mean:

        k_face = 2 * k_P * k_N / (k_P + k_N)

    Physical consequence: if either cell is nearly impermeable (k -> 0), the
    harmonic mean -> 0 and the face correctly chokes flow. An arithmetic mean
    would let fluid "leak" through a plugged cell, which is the classic bug
    that makes formation-damage models under-predict permeability decline.

    A small floor avoids 0/0 when both cells are fully plugged.
    """
    a = np.maximum(a, floor)
    b = np.maximum(b, floor)
    return 2.0 * a * b / (a + b)


def report_global_mass_balance(divergence_per_cell, source_per_cell, label="",
                               reference_rate=None):
    """
    Report the discrete global mass-balance residual.

    For incompressible steady flow the sum over all cells of (net outflow -
    source) must be ~0 to machine precision IF the discretization is
    conservative and the linear system was solved accurately. This is the
    single most important sanity check in a finite-volume reservoir simulator,
    so we expose it explicitly rather than hiding it.

    `reference_rate` (e.g. the injected volumetric rate, [m^3/s]) is used to
    normalise the residual into a physically meaningful relative error. If not
    supplied we fall back to the summed |source|, which can be misleading when
    interior and boundary fluxes are accounted together -- so passing the true
    throughput is strongly preferred.

    Returns the absolute residual; the caller decides what to do with it.
    """
    residual = float(np.sum(divergence_per_cell - source_per_cell))
    if reference_rate is not None and reference_rate > 0:
        scale = reference_rate
    else:
        scale = float(np.sum(np.abs(source_per_cell))) + 1e-30
    rel = abs(residual) / scale
    print(f"[mass-balance{(' ' + label) if label else ''}] "
          f"absolute residual = {residual:+.3e} m^3/s, "
          f"relative = {rel:.3e}")
    return residual


def advective_timestep(velocity_magnitude, cell_size, porosity, cfl=0.5):
    """
    Estimate a stable advective (CFL-limited) timestep.

    PHASE 1 STATUS: This is SCAFFOLDING. In Phase 1 we solve a steady elliptic
    pressure equation, which has NO timestep stability limit. But the moment we
    add species transport (fines, reactive ions) in later phases, an explicit
    advection update is limited by the Courant-Friedrichs-Lewy condition:

        dt <= CFL * dx / v_interstitial,   v_interstitial = darcy_velocity/phi

    We compute and report it now so the time-stepping framework and its
    diagnostics already exist when transport is switched on.
    """
    v_interstitial = velocity_magnitude / np.maximum(porosity, 1e-6)
    v_max = float(np.max(v_interstitial))
    if v_max <= 0.0:
        return np.inf
    return cfl * cell_size / v_max
