"""
ASPiRe-3D : core/properties.py
===============================================================================
Petrophysical (rock) and fluid property fields.

THE MOST IMPORTANT EXTENSIBILITY HOOK IN PHASE 1
------------------------------------------------
Permeability k, porosity phi, viscosity mu, and density rho are stored as
PER-CELL ARRAYS over the full (nx,ny,nz) grid -- never as scalars baked into
the solver. In Phase 1 they happen to be homogeneous (constant), but the
entire thesis is about these fields EVOLVING:

    * fines migration / deposition  -> local porosity and permeability decline
    * precipitation / scaling       -> pore plugging -> k decline
    * adsorption                    -> effective porosity change
    * polymer in solution           -> spatially variable viscosity

Because the pressure solver reads these arrays, any future module can simply
overwrite, e.g., k[i,j,k] *= damage_factor and the next pressure solve
automatically reflects the damage. No solver changes required. This is the
designed coupling point.

UNITS
-----
All values are SI (m^2 for k, Pa.s for mu, kg/m^3 for rho, dimensionless phi).
Use utils.constants to convert from mD / cP at input time.
===============================================================================
"""

import numpy as np
from utils.constants import MILLIDARCY_TO_M2, CENTIPOISE_TO_PAS


class FluidRockProperties:
    """Container for spatially-resolved rock and fluid properties (SI units)."""

    def __init__(self, mesh,
                 permeability_mD=100.0,
                 porosity=0.20,
                 viscosity_cP=1.0,
                 density=1000.0):
        """
        Parameters
        ----------
        mesh : StructuredMesh
        permeability_mD : float
            Initial homogeneous absolute permeability [milliDarcy]. ~100 mD is
            a representative value for a moderately permeable sandstone.
        porosity : float
            Initial homogeneous porosity [-]. ~0.20 typical for sandstone.
        viscosity_cP : float
            Aqueous-phase dynamic viscosity [centipoise]. Brine ~ 1 cP.
        density : float
            Aqueous-phase density [kg/m^3]. Brine ~ 1000 kg/m^3.
        """
        self.mesh = mesh
        shape = (mesh.nx, mesh.ny, mesh.nz)

        # ---- Rock properties (full-grid arrays) ----------------------------
        # Stored over the whole box for simple (i,j,k) indexing; inactive cells
        # carry placeholder values that the solver never references.
        self.k = np.full(shape, permeability_mD * MILLIDARCY_TO_M2)   # [m^2]
        self.phi = np.full(shape, porosity)                          # [-]

        # ---- Fluid properties ----------------------------------------------
        self.mu = np.full(shape, viscosity_cP * CENTIPOISE_TO_PAS)   # [Pa.s]
        self.rho = np.full(shape, density)                           # [kg/m^3]

        # Record initial homogeneous values for later damage-ratio reporting.
        self.k0 = self.k.copy()
        self.phi0 = self.phi.copy()

    # -----------------------------------------------------------------------
    def mobility(self):
        """
        Phase-mobility-like field lambda = k / mu  [m^2 / (Pa.s)].

        In single-phase Phase 1 this is just absolute permeability over
        viscosity. In later phases relative permeability enters here, which is
        why we expose it as a method rather than hard-coding k/mu in the solver.
        """
        return self.k / self.mu

    # -----------------------------------------------------------------------
    def summary(self):
        m = self.mesh
        act = m.active
        k_mD = self.k[act].mean() / MILLIDARCY_TO_M2
        return (
            "FluidRockProperties (mean over active cells)\n"
            f"  permeability : {k_mD:.3f} mD ({self.k[act].mean():.3e} m^2)\n"
            f"  porosity     : {self.phi[act].mean():.3f}\n"
            f"  viscosity    : {self.mu[act].mean() / CENTIPOISE_TO_PAS:.3f} cP\n"
            f"  density      : {self.rho[act].mean():.1f} kg/m^3\n"
        )
