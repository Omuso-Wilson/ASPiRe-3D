"""
ASPiRe-3D : core/velocity.py
===============================================================================
Darcy velocity field reconstruction from the converged pressure field.

WHY A SEPARATE STEP?
--------------------
In FVM, pressure lives at cell centers but FLUX lives at faces. After solving
for pressure we reconstruct the face fluxes using the SAME transmissibilities
used in assembly -- this guarantees the reconstructed velocity field is
DISCRETELY CONSISTENT with the equation we solved, so the net flux balance per
cell equals its source to machine precision. (Using a different gradient
formula here would silently break conservation -- a subtle but fatal bug for
later transport.)

OUTPUTS
-------
1. Face volumetric fluxes Q_f [m^3/s] across each interior face.
2. Cell-centered Darcy velocity vector u = (ux, uy, uz) [m/s], obtained by
   averaging the two opposing face Darcy fluxes per axis (a standard
   cell-center reconstruction). Darcy (superficial) velocity is flux / area.
3. Per-cell net outflow (divergence) for the global mass-balance audit.

INTERSTITIAL vs DARCY VELOCITY
------------------------------
Darcy velocity u is superficial (flux per total area). The actual pore-fluid
(interstitial) velocity that advects fines and reactants is v = u / phi. We
provide both, because transport (next phases) needs v, while injectivity /
flux reporting uses u.
===============================================================================
"""

import numpy as np
from utils.helpers import harmonic_mean


class VelocityField:
    """Reconstructs Darcy fluxes/velocities from a solved pressure field."""

    def __init__(self, mesh, properties):
        self.mesh = mesh
        self.props = properties
        self.ux = self.uy = self.uz = None      # cell-center Darcy velocity [m/s]
        self.speed = None                       # |u| [m/s]
        self.net_outflow = None                 # per-cell net flux [m^3/s]

    # -----------------------------------------------------------------------
    def _mobility_face(self, A_cell, B_cell):
        """Face mobility (k/mu) with harmonic k and arithmetic mu (matches solver)."""
        k_face = harmonic_mean(self.props.k[A_cell], self.props.k[B_cell])
        mu_face = 0.5 * (self.props.mu[A_cell] + self.props.mu[B_cell])
        return k_face / mu_face

    # -----------------------------------------------------------------------
    def compute(self, pressure, bc=None):
        """
        Reconstruct velocities from the full-grid `pressure` array.

        For each axis we compute the Darcy velocity at the two faces of a cell
        and average them to the center. The face Darcy velocity is

            u_face = -(k/mu)_face * (P_N - P_P) / d_f

        (Darcy's law with central-difference gradient). Volumetric face flux is
        u_face * A_f; net per-cell outflow sums signed face fluxes.
        """
        mesh, props = self.mesh, self.props
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        ux = np.zeros((nx, ny, nz))
        uy = np.zeros((nx, ny, nz))
        uz = np.zeros((nx, ny, nz))
        net_outflow = np.zeros((nx, ny, nz))    # [m^3/s], +ve = net out

        # Helper to test activity safely.
        def is_active(i, j, k):
            return (0 <= i < nx and 0 <= j < ny and 0 <= k < nz
                    and mesh.active[i, j, k])

        for dof in range(mesh.n_active):
            i, j, k = mesh.dof_to_ijk[dof]
            P0 = pressure[i, j, k]

            # ---- x-axis faces (west i-1 / east i+1) ----------------------
            # Both face velocities are expressed as the +x COMPONENT of the
            # Darcy velocity at that face, using a consistent forward gradient:
            #     u_x = -(k/mu) dP/dx
            # West face (between i-1 and i): dP/dx ~ (P0 - P[i-1])/dx.
            # East face (between i and i+1): dP/dx ~ (P[i+1] - P0)/dx.
            # Averaging two consistently-signed components gives the true
            # cell-centred velocity (previously a sign clash cancelled them).
            ux_w = ux_e = None
            if is_active(i - 1, j, k):
                mob = self._mobility_face((i, j, k), (i - 1, j, k))
                ux_w = -mob * (P0 - pressure[i - 1, j, k]) / mesh.dx
                # net outflow across the west face (outward normal = -x):
                net_outflow[i, j, k] += (-ux_w) * mesh.area_x
            if is_active(i + 1, j, k):
                mob = self._mobility_face((i, j, k), (i + 1, j, k))
                ux_e = -mob * (pressure[i + 1, j, k] - P0) / mesh.dx
                # net outflow across the east face (outward normal = +x):
                net_outflow[i, j, k] += (ux_e) * mesh.area_x
            # Cell-centre value = average of available consistently-signed faces.
            if ux_w is not None and ux_e is not None:
                ux[i, j, k] = 0.5 * (ux_w + ux_e)
            elif ux_e is not None:
                ux[i, j, k] = ux_e
            elif ux_w is not None:
                ux[i, j, k] = ux_w

            # ---- y-axis faces (consistent +y component) ------------------
            uy_s = uy_n = None
            if is_active(i, j - 1, k):
                mob = self._mobility_face((i, j, k), (i, j - 1, k))
                uy_s = -mob * (P0 - pressure[i, j - 1, k]) / mesh.dy
                net_outflow[i, j, k] += (-uy_s) * mesh.area_y
            if is_active(i, j + 1, k):
                mob = self._mobility_face((i, j, k), (i, j + 1, k))
                uy_n = -mob * (pressure[i, j + 1, k] - P0) / mesh.dy
                net_outflow[i, j, k] += (uy_n) * mesh.area_y
            if uy_s is not None and uy_n is not None:
                uy[i, j, k] = 0.5 * (uy_s + uy_n)
            elif uy_n is not None:
                uy[i, j, k] = uy_n
            elif uy_s is not None:
                uy[i, j, k] = uy_s

            # ---- z-axis faces (consistent +z component) ------------------
            uz_b = uz_t = None
            if is_active(i, j, k - 1):
                mob = self._mobility_face((i, j, k), (i, j, k - 1))
                uz_b = -mob * (P0 - pressure[i, j, k - 1]) / mesh.dz
                net_outflow[i, j, k] += (-uz_b) * mesh.area_z
            if is_active(i, j, k + 1):
                mob = self._mobility_face((i, j, k), (i, j, k + 1))
                uz_t = -mob * (pressure[i, j, k + 1] - P0) / mesh.dz
                net_outflow[i, j, k] += (uz_t) * mesh.area_z
            if uz_b is not None and uz_t is not None:
                uz[i, j, k] = 0.5 * (uz_b + uz_t)
            elif uz_t is not None:
                uz[i, j, k] = uz_t
            elif uz_b is not None:
                uz[i, j, k] = uz_b

        self.ux, self.uy, self.uz = ux, uy, uz
        self.speed = np.sqrt(ux**2 + uy**2 + uz**2)
        self.net_outflow = net_outflow
        return ux, uy, uz

    # -----------------------------------------------------------------------
    def interstitial_speed(self):
        """Pore (interstitial) velocity magnitude v = |u| / phi  [m/s]."""
        return self.speed / np.maximum(self.props.phi, 1e-6)

    # -----------------------------------------------------------------------
    def boundary_face_flux(self, pressure, i_face, p_ghost):
        """
        Volumetric flux [m^3/s] LEAVING the core across an x end-face into a
        Dirichlet ghost held at pressure `p_ghost`, summed over the active
        cells of that face.

        This mirrors EXACTLY how the solver modeled the end face: a ghost
        half-cell at distance dx/2, transmissibility T_b = (k/mu)*(A_x/(dx/2)).
        The outward flux from boundary cell P is

            q_out = T_b * (P_P - p_ghost)            [m^3/s]

        Sign: positive q_out means fluid leaves the core (correct for the
        outlet, where P_P > p_ghost). At the inlet this returns a negative
        number (fluid enters), so the inlet INFLOW is -q_out.

        Using the ghost flux -- not the cell-center averaged velocity -- is the
        discretely consistent way to read off throughput, because the
        cell-center ux of an end cell only sees its interior neighbor.
        """
        mesh, props = self.mesh, self.props
        Ax, dx = mesh.area_x, mesh.dx
        Q = 0.0
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                if mesh.active[i_face, j, k]:
                    mob = props.k[i_face, j, k] / props.mu[i_face, j, k]
                    T_b = mob * (Ax / (0.5 * dx))
                    Q += T_b * (pressure[i_face, j, k] - p_ghost)
        return Q

    def total_throughput(self, pressure, bc):
        """
        Total volumetric flow rate through the core [m^3/s], read as the flux
        leaving the OUTLET face into its Dirichlet ghost. For a converged
        incompressible solve this equals the injected rate (mass balance), so
        it is the headline physical cross-check on the whole pipeline.
        """
        i_out = self.mesh.nx - 1
        return self.boundary_face_flux(pressure, i_out, bc.p_outlet)
