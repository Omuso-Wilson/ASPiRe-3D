"""
ASPiRe-3D : core/pressure_solver.py
===============================================================================
Implicit finite-volume pressure solver for incompressible single-phase Darcy
flow.

GOVERNING EQUATION (Phase 1, steady, incompressible, constant density)
----------------------------------------------------------------------
Continuity + Darcy's law give the elliptic pressure (Poisson) equation:

        div( (k/mu) grad P ) = - q_tilde            ... (1)

where q_tilde is a volumetric source per unit volume [1/s] (zero in the core
interior; boundary fluxes enter through the BCs, not as interior sources).

FINITE-VOLUME DISCRETIZATION
----------------------------
Integrate (1) over a cell P and apply the divergence theorem -> the volume
integral of the divergence becomes a SUM OF FACE FLUXES:

        sum_f [ (k/mu) grad P . n ]_f * A_f  =  -q_P

Approximate the face-normal gradient by a two-point central difference between
the cell and its neighbor. For the face shared by P and neighbor N:

        flux_f = T_f * (P_N - P_P)

with the TRANSMISSIBILITY

        T_f = (k_f / mu_f) * (A_f / d_f)            [m^3 / (Pa . s)]

   * A_f  = face area,
   * d_f  = center-to-center distance,
   * k_f  = HARMONIC mean of the two cell permeabilities (series resistance),
   * mu_f = arithmetic mean viscosity (smooth field; harmonic optional later).

Summed over the (up to 6) faces of cell P, the discrete equation is:

        sum_N T_f (P_N - P_P) = -q_P

This places, on matrix row `row(P)`:
        diagonal     A[P,P]  -= sum_N T_f
        off-diagonal A[P,N]  += T_f
RHS b[P] starts at -q_P (=0 interior) and is modified by boundary conditions.

PROPERTIES OF THE SYSTEM
------------------------
A is SPARSE (<=7 nonzeros/row), SYMMETRIC, and (after at least one Dirichlet
anchor) NEGATIVE-DEFINITE -> well-conditioned and uniquely solvable. For
core-plug sizes we use a sparse DIRECT solve (scipy spsolve / LU), which is
exact to round-off and avoids iterative-solver tolerance bookkeeping while we
validate the physics. An iterative CG path is trivial to add later for very
large grids.

NO-FLOW BOUNDARIES ARE AUTOMATIC
--------------------------------
A face whose neighbor is inactive (outside the cylinder) or off-grid in y/z is
simply NOT added to the assembly. Omitting a face = zero flux across it = a
no-flow (sealed sleeve) boundary, with no special-case code. This is one of the
elegances of finite volume on a masked grid.
===============================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from utils.helpers import harmonic_mean
from physics.boundary_conditions import BCMode


class PressureSolver:
    """Assembles and solves A p = b for the cell-centered pressure field."""

    def __init__(self, mesh, properties):
        self.mesh = mesh
        self.props = properties

        # Solution and bookkeeping (filled by solve()).
        self.pressure = None            # full-grid (nx,ny,nz) array, NaN outside
        self.A = None                   # sparse system matrix
        self.b = None                   # RHS vector
        self._diag = None               # diagonal accumulator during assembly

    # =======================================================================
    #  ASSEMBLY
    # =======================================================================
    def _face_transmissibility(self, ijk_P, ijk_N, area, spacing):
        """
        Transmissibility T_f for the face between cells P and N.

            T_f = (k_f / mu_f) * (A_f / d_f)

        k_f : harmonic mean (series flow resistance, correctly chokes on
              low-k / plugged cells -- essential for damage modelling).
        mu_f: arithmetic mean (viscosity is a smooth fluid field).
        """
        kP = self.props.k[ijk_P]; kN = self.props.k[ijk_N]
        muP = self.props.mu[ijk_P]; muN = self.props.mu[ijk_N]
        k_face = harmonic_mean(kP, kN)
        mu_face = 0.5 * (muP + muN)
        return (k_face / mu_face) * (area / spacing)

    def _assemble(self, bc):
        """
        Build the sparse matrix A and RHS b for the given boundary conditions.

        We accumulate COO-style triplets (rows, cols, vals) for interior
        face couplings, track each row's diagonal separately, then add boundary
        contributions, and finally build a CSR matrix in one shot (efficient
        and avoids repeated sparse-structure edits).
        """
        mesh, props = self.mesh, self.props
        n = mesh.n_active

        rows, cols, vals = [], [], []
        diag = np.zeros(n, dtype=np.float64)     # diagonal accumulator
        b = np.zeros(n, dtype=np.float64)        # RHS

        # ---- 1) Interior face couplings -----------------------------------
        # Loop over active cells; for each in-bounds ACTIVE neighbor add the
        # symmetric coupling. We add each off-diagonal once per (P,N) ordering;
        # since we visit every active cell, both (P->N) and (N->P) get added,
        # which is exactly the symmetric pair the matrix needs.
        for dof in range(n):
            i, j, k = mesh.dof_to_ijk[dof]
            row = mesh.cell_id[i, j, k]
            for area, spacing, (ni, nj, nk) in mesh.neighbors(i, j, k):
                if not mesh.active[ni, nj, nk]:
                    # Inactive neighbor => sealed face => no-flow => skip.
                    continue
                col = mesh.cell_id[ni, nj, nk]
                T = self._face_transmissibility((i, j, k), (ni, nj, nk),
                                                area, spacing)
                # Off-diagonal: +T ; diagonal accumulates -T.
                rows.append(row); cols.append(col); vals.append(T)
                diag[row] -= T

        # ---- 2) Boundary conditions on the x end-faces --------------------
        self._apply_boundaries(bc, diag, b)

        # ---- 3) Add diagonal entries and finalize CSR ---------------------
        rows.extend(range(n))
        cols.extend(range(n))
        vals.extend(diag.tolist())

        self._diag = diag
        self.A = csr_matrix((vals, (rows, cols)), shape=(n, n))
        self.b = b

    def _apply_boundaries(self, bc, diag, b):
        """
        Modify (diag, b) for the inlet (x=0, i=0) and outlet (x=L, i=nx-1)
        faces according to the BC mode.

        DIRICHLET (prescribed face pressure P_bc) on a boundary cell P:
        We model a "ghost" half-cell at the prescribed pressure, one half-cell
        spacing (dx/2) away from the cell center, sharing the x-face. Its
        transmissibility is
                T_b = (k_P / mu_P) * (A_x / (dx/2))
        and it contributes  T_b*(P_bc - P_P), i.e.
                diag[P] -= T_b ;  b[P] -= T_b * P_bc.

        NEUMANN (prescribed injection rate) on the inlet:
        A fixed total rate Q is split uniformly over the active inlet-face
        cells. Each such cell receives a source +Q_cell on the RHS:
                b[P] -= (-Q_cell)  ->  b[P] += ... handled with sign of (1).
        Recall the discrete eqn is  sum T(P_N-P_P) = -q_P, and a positive
        injection into the cell is a positive q_P, so it appears as b[P] -=
        q_cell with q_cell>0 for injection. We implement that sign explicitly
        below.
        """
        mesh, props = self.mesh, self.props
        Ax, dx = mesh.area_x, mesh.dx

        # Identify active inlet/outlet face cells (i = 0 and i = nx-1).
        inlet_cells = [(0, j, k) for j in range(mesh.ny) for k in range(mesh.nz)
                       if mesh.active[0, j, k]]
        outlet_cells = [(mesh.nx - 1, j, k) for j in range(mesh.ny)
                        for k in range(mesh.nz)
                        if mesh.active[mesh.nx - 1, j, k]]

        def dirichlet(cell, p_bc):
            mob = props.k[cell] / props.mu[cell]
            T_b = mob * (Ax / (0.5 * dx))      # ghost half-cell distance dx/2
            row = mesh.cell_id[cell]
            diag[row] -= T_b
            b[row] -= T_b * p_bc

        if bc.mode == BCMode.CONSTANT_PRESSURE_DROP:
            # Dirichlet at both ends.
            for c in inlet_cells:
                dirichlet(c, bc.p_inlet)
            for c in outlet_cells:
                dirichlet(c, bc.p_outlet)

        elif bc.mode == BCMode.CONSTANT_RATE:
            # Neumann inlet (rate split uniformly across inlet cells) ...
            n_in = len(inlet_cells)
            if n_in == 0:
                raise RuntimeError("No active inlet cells found.")
            q_cell = bc.injection_rate / n_in      # [m^3/s] into each inlet cell
            for c in inlet_cells:
                row = mesh.cell_id[c]
                # Discrete eqn: sum T (P_N - P_P) = -q_P. Injection q_P>0
                # appears on RHS as b[row] -= q_P.
                b[row] -= q_cell
            # ... and ONE Dirichlet anchor at the outlet to remove the null
            # space (pure Neumann pressure is defined only up to a constant).
            for c in outlet_cells:
                dirichlet(c, bc.p_outlet)

    # =======================================================================
    #  SOLVE
    # =======================================================================
    def solve(self, bc, verbose=True):
        """
        Assemble and solve the pressure system, then scatter the compact DOF
        solution back onto the full (nx,ny,nz) grid (inactive cells = NaN).
        """
        self._assemble(bc)

        if verbose:
            nnz = self.A.nnz
            print(f"[pressure] system: {self.mesh.n_active} unknowns, "
                  f"{nnz} nonzeros ({nnz / self.mesh.n_active:.2f} per row)")

        # Sparse direct solve. A is symmetric negative-definite; spsolve uses a
        # sparse LU (SuperLU) which is robust for this size. Exact to round-off.
        p_dof = spsolve(self.A.tocsc(), self.b)

        # Scatter back to grid.
        P = np.full((self.mesh.nx, self.mesh.ny, self.mesh.nz), np.nan)
        for dof in range(self.mesh.n_active):
            i, j, k = self.mesh.dof_to_ijk[dof]
            P[i, j, k] = p_dof[dof]
        self.pressure = P

        if verbose:
            act = self.mesh.active
            print(f"[pressure] solved. P range over core: "
                  f"[{np.nanmin(P[act]):.4e}, {np.nanmax(P[act]):.4e}] Pa")
        return P
