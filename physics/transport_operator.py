"""
ASPiRe-3D : core/transport_operator.py
===============================================================================
Sparse finite-volume advection-dispersion OPERATOR (VECTORIZED).

This is the performance-hardened version of the Phase 2b operator. The physics
and sign conventions are IDENTICAL and still validated by the Phase 2b/3 tests;
only the assembly is rewritten from per-cell Python loops to vectorized NumPy +
a single sparse build. This matters because the coupled formation-damage loop
(Phase 4) re-assembles transport operators as permeability evolves, so assembly
speed is now on the critical path.

KEY IDEA OF THE VECTORIZATION
-----------------------------
The mesh topology (which faces connect which DOFs) is FIXED. We precompute, once,
the integer arrays of face owner-DOFs and neighbour-DOFs for the +x/+y/+z faces.
Each time a fresh operator is needed, we compute per-face flux and dispersion
transmissibility as ARRAY operations and emit the COO triplets by array
concatenation -- no Python-level per-cell loop.

MATHEMATICAL FORM (unchanged from Phase 2b)
-------------------------------------------
Backward-Euler implicit ADE -> per-step system  M C = r  with
    M = diag(phi V/dt) - L + diag(outlet_flux)
where L is the spatial operator built here:
    (L C)_P = sum_f ( -adv_f - disp_f )     (net spatial flux into P)

Per +face between P and N with signed Darcy flux Q (>0 => P->N):
  ADV (first-order upwind):
     Q>=0 : L[P,P]-=Q ; L[N,P]+=Q          (upwind = P)
     Q<0  : L[N,N]+=Q ; L[P,N]-=Q          (upwind = N)
  DISP (central), Td=(phiD)_face*A/d:
     L[P,P]-=Td ; L[P,N]+=Td ; L[N,N]-=Td ; L[N,P]+=Td

These signs make M a (non-symmetric) M-matrix -> monotone, bounded solutions.
===============================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix


class TransportOperator:
    """
    Vectorized sparse advection-dispersion operator on a fixed Darcy field.
    Acts on the compact active-cell DOF vector (same numbering as pressure).
    """

    def __init__(self, mesh, properties, velocity, dispersion_field):
        self.mesh = mesh
        self.props = properties
        self.vel = velocity
        self.D = dispersion_field

        self._build_face_topology()      # once; independent of properties
        self._compute_face_fluxes()
        self._compute_boundary_fluxes()
        self.L = None
        self._assemble_spatial_operator()

    # =======================================================================
    #  FACE TOPOLOGY  (computed once)
    # =======================================================================
    def _build_face_topology(self):
        """Integer arrays describing every interior +face (owner P, neighbour N)."""
        mesh = self.mesh
        cid = mesh.cell_id
        act = mesh.active
        I, J, K = np.meshgrid(np.arange(mesh.nx), np.arange(mesh.ny),
                              np.arange(mesh.nz), indexing="ij")
        faces = {}
        # +x faces
        m = act[:-1, :, :] & act[1:, :, :]
        faces["x"] = dict(
            fo=cid[:-1, :, :][m], fn=cid[1:, :, :][m],
            oi=I[:-1, :, :][m], oj=J[:-1, :, :][m], ok=K[:-1, :, :][m],
            ni=I[1:, :, :][m], nj=J[1:, :, :][m], nk=K[1:, :, :][m],
            spacing=mesh.dx, area=mesh.area_x)
        # +y faces
        m = act[:, :-1, :] & act[:, 1:, :]
        faces["y"] = dict(
            fo=cid[:, :-1, :][m], fn=cid[:, 1:, :][m],
            oi=I[:, :-1, :][m], oj=J[:, :-1, :][m], ok=K[:, :-1, :][m],
            ni=I[:, 1:, :][m], nj=J[:, 1:, :][m], nk=K[:, 1:, :][m],
            spacing=mesh.dy, area=mesh.area_y)
        # +z faces
        m = act[:, :, :-1] & act[:, :, 1:]
        faces["z"] = dict(
            fo=cid[:, :, :-1][m], fn=cid[:, :, 1:][m],
            oi=I[:, :, :-1][m], oj=J[:, :, :-1][m], ok=K[:, :, :-1][m],
            ni=I[:, :, 1:][m], nj=J[:, :, 1:][m], nk=K[:, :, 1:][m],
            spacing=mesh.dz, area=mesh.area_z)
        self._faces = faces

    # =======================================================================
    #  FACE & BOUNDARY FLUXES (vectorized)
    # =======================================================================
    def _compute_face_fluxes(self):
        ux, uy, uz = self.vel.ux, self.vel.uy, self.vel.uz
        for axis, u in (("x", ux), ("y", uy), ("z", uz)):
            f = self._faces[axis]
            f["Q"] = 0.5 * (u[f["oi"], f["oj"], f["ok"]]
                            + u[f["ni"], f["nj"], f["nk"]]) * f["area"]

    def _compute_boundary_fluxes(self):
        mesh = self.mesh
        n = mesh.n_active
        ux = self.vel.ux
        inlet = np.zeros(n); outlet = np.zeros(n)
        a0 = mesh.active[0, :, :]
        inlet[mesh.cell_id[0, :, :][a0]] = ux[0, :, :][a0] * mesh.area_x
        io = mesh.nx - 1
        aL = mesh.active[io, :, :]
        outlet[mesh.cell_id[io, :, :][aL]] = ux[io, :, :][aL] * mesh.area_x
        self._inlet_face_flux = inlet
        self._outlet_face_flux = outlet

    # =======================================================================
    #  VECTORIZED ASSEMBLY OF L
    # =======================================================================
    def _assemble_spatial_operator(self):
        props, D = self.props, self.D
        n = self.mesh.n_active
        rows_all, cols_all, vals_all = [], [], []

        for axis in ("x", "y", "z"):
            f = self._faces[axis]
            P = f["fo"].astype(np.int64); N = f["fn"].astype(np.int64); Q = f["Q"]
            if P.size == 0:
                continue
            phiD_o = props.phi[f["oi"], f["oj"], f["ok"]] * D[f["oi"], f["oj"], f["ok"]]
            phiD_n = props.phi[f["ni"], f["nj"], f["nk"]] * D[f["ni"], f["nj"], f["nk"]]
            Td = 0.5 * (phiD_o + phiD_n) * f["area"] / f["spacing"]

            pos = Q >= 0.0
            Qpos = np.where(pos, Q, 0.0)
            Qneg = np.where(pos, 0.0, Q)

            # advection
            rows_all += [P, N, N, P]
            cols_all += [P, P, N, N]
            vals_all += [-Qpos, +Qpos, +Qneg, -Qneg]
            # dispersion
            rows_all += [P, P, N, N]
            cols_all += [P, N, N, P]
            vals_all += [-Td, +Td, -Td, +Td]

        if rows_all:
            rows = np.concatenate(rows_all)
            cols = np.concatenate(cols_all)
            vals = np.concatenate(vals_all)
        else:
            rows = cols = np.array([], dtype=np.int64); vals = np.array([], dtype=float)
        self.L = csr_matrix((vals, (rows, cols)), shape=(n, n))

    # -----------------------------------------------------------------------
    def update(self, velocity=None, dispersion_field=None):
        """Rebuild after Darcy field / dispersion change (topology reused)."""
        if velocity is not None:
            self.vel = velocity
        if dispersion_field is not None:
            self.D = dispersion_field
        self._compute_face_fluxes()
        self._compute_boundary_fluxes()
        self._assemble_spatial_operator()

    # =======================================================================
    #  DIAGONALS / RHS HELPERS (vectorized)
    # =======================================================================
    def storage_diagonal(self):
        mesh, props = self.mesh, self.props
        ijk = mesh.dof_to_ijk
        return props.phi[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * mesh.cell_volume

    def outlet_diagonal(self):
        return self._outlet_face_flux.copy()

    def inlet_rhs(self, c_inject):
        return self._inlet_face_flux * c_inject

    def to_grid(self, c_vec):
        mesh = self.mesh
        C = np.full((mesh.nx, mesh.ny, mesh.nz), np.nan)
        ijk = mesh.dof_to_ijk
        C[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = c_vec
        return C

    def from_grid(self, C):
        mesh = self.mesh
        ijk = mesh.dof_to_ijk
        return C[ijk[:, 0], ijk[:, 1], ijk[:, 2]].copy()
