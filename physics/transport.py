"""
ASPiRe-3D : core/transport.py
===============================================================================
Phase 2 : passive tracer transport by advection-dispersion.

GOVERNING EQUATION
------------------
A single passive, conservative, non-reactive tracer of concentration C rides on
the Phase 1 Darcy velocity field:

        phi * dC/dt + div(u C) - div(phi D grad C) = 0          ...(ADE)

    term 1  phi dC/dt        : storage in pore space (porosity scales storage)
    term 2  div(u C)         : ADVECTION (tracer carried by Darcy flux u)
    term 3  div(phi D grad C): DISPERSION / diffusion (Fickian)

This is the prerequisite operator for ALL later reactive species (fines, ions,
polymer): they advect and disperse identically; chemistry only adds
source/sink terms to the right-hand side. Building and validating it cleanly
now de-risks every later phase.

NUMERICAL METHOD (finite volume, explicit in time)
--------------------------------------------------
Integrate the ADE over cell P and apply the divergence theorem -> a balance of
face fluxes. For each face f with outward volumetric Darcy flux Q_f [m^3/s]:

  * ADVECTION -> FIRST-ORDER UPWIND.
      C_face = C_upwind   (the cell on the side flow is coming FROM)
      adv_flux_f = Q_f * C_face        [ (kg/m^3)*(m^3/s) = kg/s ]
    Upwinding is chosen because central differencing of advection is
    UNCONDITIONALLY UNSTABLE for advection-dominated transport (it generates
    oscillations and negative concentrations -- physically meaningless and
    fatal for later reaction chemistry). Upwinding is unconditionally stable
    and MONOTONE: it guarantees 0 <= C <= C_inj with no over/undershoot. The
    price is numerical diffusion (the front smears); this is a well-understood,
    defensible first-order choice. Flux-limited higher-order schemes are a
    documented future refinement.

  * DISPERSION -> CENTRAL DIFFERENCE.
      disp_flux_f = -(phi D)_f * (C_N - C_P)/d_f * A_f          [kg/s]
    The diffusion operator is symmetric and well-behaved with central
    differencing (exactly as in the Phase 1 pressure Laplacian).

TIME INTEGRATION + STABILITY (the "stable timestep framework")
--------------------------------------------------------------
Explicit forward Euler. Explicit advection obeys the Courant-Friedrichs-Lewy
(CFL) limit; explicit diffusion obeys the parabolic (von Neumann) limit:

      dt_adv  = CFL * dx_min / v_max         (v = interstitial velocity u/phi)
      dt_diff = safety * dx_min^2 / (2*ndim*D_eff)
      dt      = min(dt_adv, dt_diff)

The solver recomputes dt each step from the current field, MONITORS the Courant
and diffusion numbers, and reports them -- so stability is observed, not merely
assumed. An implicit transport option is a documented extension for stiff
(diffusion- or reaction-dominated) regimes.

CONSERVATIVE FLUX FORMULATION
-----------------------------
The cell update is written as a pure flux balance:

  C_P^{n+1} = C_P^n + (dt/(phi_P V_P)) * sum_f ( -adv_flux_f - disp_flux_f )

Each interior face flux enters one cell with +sign and its neighbour with
-sign (identical magnitude), so interior fluxes cancel exactly when summed over
the domain. Total tracer mass therefore changes ONLY through boundary
(inlet/outlet) fluxes -> discrete global conservation to round-off (tested).

BOUNDARY CONDITIONS
-------------------
  inlet  (x=0)   : Dirichlet C = C_inj (injected tracer enters advectively)
  outlet (x=L)   : outflow / zero-gradient (dC/dx = 0); tracer leaves at its
                   own concentration, no artificial reflection
  sleeve (sides) : no-flux (inherited from the masked no-flow geometry)
===============================================================================
"""

import numpy as np


class TracerTransport:
    """
    Explicit finite-volume solver for passive tracer advection-dispersion on a
    precomputed Darcy velocity field.

    Parameters
    ----------
    mesh : StructuredMesh
    properties : FluidRockProperties     (porosity used for storage & v)
    velocity : VelocityField             (already .compute()'d)
    longitudinal_dispersivity : float    alpha_L [m]
    transverse_dispersivity   : float    alpha_T [m]
    molecular_diffusion       : float    D_m [m^2/s]
    cfl : float                          Courant target (<=1; default 0.5)
    diffusion_safety : float             parabolic-limit safety factor (<=1)
    """

    def __init__(self, mesh, properties, velocity,
                 longitudinal_dispersivity=1.0e-3,
                 transverse_dispersivity=1.0e-4,
                 molecular_diffusion=1.0e-9,
                 cfl=0.5, diffusion_safety=0.5):
        self.mesh = mesh
        self.props = properties
        self.vel = velocity
        self.alpha_L = float(longitudinal_dispersivity)
        self.alpha_T = float(transverse_dispersivity)
        self.D_m = float(molecular_diffusion)
        self.cfl = float(cfl)
        self.diffusion_safety = float(diffusion_safety)

        # Concentration field over the full grid; NaN outside the core so plots
        # mask cleanly and accidental use of dead cells is obvious.
        self.C = np.full((mesh.nx, mesh.ny, mesh.nz), np.nan)
        self.C[mesh.active] = 0.0          # initially tracer-free core

        # Diagnostics history (for timestep monitoring / thesis plots).
        self.time = 0.0
        self.history = {"time": [], "dt": [], "courant": [],
                        "diffusion_number": [], "mass": [], "outlet_C": []}

        # Precompute per-face volumetric Darcy fluxes Q_f once (velocity field
        # is steady in Phase 2). Stored as dictionaries keyed by the "owner"
        # cell and the axis, holding the flux across its + face.
        self._precompute_face_fluxes()
        # Effective scalar dispersion coefficient (isotropic-mechanical model;
        # see _effective_dispersion). Used for the diffusion stability limit.
        self._D_eff_field = self._effective_dispersion()

    # =======================================================================
    #  PRECOMPUTATION
    # =======================================================================
    def _precompute_face_fluxes(self):
        """
        Compute the signed volumetric Darcy flux across every interior face,
        consistent with the solved pressure field.

        We store, for each cell, the flux across its +x, +y, +z faces (the
        "-" faces of a cell are the "+" faces of its lower neighbour, so we
        never double-store). Flux sign convention: positive = flow in the
        +axis direction across that face.

        Q across the +x face of cell (i,j,k):
            Q = u_face_x * A_x
        where u_face_x is the face Darcy velocity. We reuse the velocity
        field's own face mobilities for discrete consistency with the pressure
        solve, recomputing the face velocity from the pressure gradient is
        equivalent; here we use the cell-centred velocities' face average,
        which for a divergence-free reconstructed field is consistent.

        IMPLEMENTATION: we recompute face velocities directly from the stored
        cell-centred Darcy velocity by simple face interpolation between the
        two active cells, which is second-order on a uniform grid and matches
        the reconstructed field. (For a uniform incompressible field this is
        the natural face flux.)
        """
        mesh = self.mesh
        nx, ny, nz = mesh.nx, mesh.ny, mesh.nz

        # +face fluxes; NaN where the +neighbour is inactive/off-grid (no face)
        self.Qx = np.zeros((nx, ny, nz))   # flux across +x face of each cell
        self.Qy = np.zeros((nx, ny, nz))
        self.Qz = np.zeros((nx, ny, nz))
        self.has_x = np.zeros((nx, ny, nz), dtype=bool)
        self.has_y = np.zeros((nx, ny, nz), dtype=bool)
        self.has_z = np.zeros((nx, ny, nz), dtype=bool)

        ux, uy, uz = self.vel.ux, self.vel.uy, self.vel.uz
        A_x, A_y, A_z = mesh.area_x, mesh.area_y, mesh.area_z

        for dof in range(mesh.n_active):
            i, j, k = mesh.dof_to_ijk[dof]
            # +x face
            if i + 1 < nx and mesh.active[i + 1, j, k]:
                u_face = 0.5 * (ux[i, j, k] + ux[i + 1, j, k])
                self.Qx[i, j, k] = u_face * A_x
                self.has_x[i, j, k] = True
            # +y face
            if j + 1 < ny and mesh.active[i, j + 1, k]:
                u_face = 0.5 * (uy[i, j, k] + uy[i, j + 1, k])
                self.Qy[i, j, k] = u_face * A_y
                self.has_y[i, j, k] = True
            # +z face
            if k + 1 < nz and mesh.active[i, j, k + 1]:
                u_face = 0.5 * (uz[i, j, k] + uz[i, j, k + 1])
                self.Qz[i, j, k] = u_face * A_z
                self.has_z[i, j, k] = True

    def _effective_dispersion(self):
        """
        Effective scalar dispersion coefficient D_eff [m^2/s] per cell, used for
        the diffusion stability limit and as the (isotropic) dispersion used in
        the flux.

        ISOTROPIC-MECHANICAL MODEL (Phase 2 default, clearly stated):
            D_eff = D_m + alpha_L * |v|
        i.e. molecular diffusion plus longitudinal mechanical dispersion driven
        by the interstitial speed |v| = |u|/phi. We use alpha_L (the larger,
        flow-aligned dispersivity) as a conservative scalar so the front
        spreading is not under-estimated. The FULL anisotropic Bear tensor
            D = (D_m + alpha_T|v|) I + (alpha_L - alpha_T) v⊗v/|v|
        is documented in the README as an extension; the isotropic form keeps
        the Phase 2 flux assembly transparent and is standard for 1D-dominated
        core floods where transport is essentially axial.
        """
        v_mag = self.vel.speed / np.maximum(self.props.phi, 1e-6)
        return self.D_m + self.alpha_L * v_mag

    # =======================================================================
    #  STABLE TIMESTEP (CFL + parabolic)
    # =======================================================================
    def stable_timestep(self):
        """
        Compute the explicit-stable timestep as the minimum of the advective
        (CFL) and diffusive (von Neumann) limits, and return it along with the
        governing Courant and diffusion numbers for monitoring.
        """
        mesh = self.mesh
        dx_min = min(mesh.dx, mesh.dy, mesh.dz)
        ndim = 3

        # Advective limit from interstitial velocity.
        v_mag = self.vel.speed[mesh.active] / np.maximum(
            self.props.phi[mesh.active], 1e-6)
        v_max = float(np.max(v_mag)) if v_mag.size else 0.0
        dt_adv = (self.cfl * dx_min / v_max) if v_max > 0 else np.inf

        # Diffusive (parabolic) limit: dt <= safety * dx^2 / (2*ndim*D_eff).
        D_max = float(np.max(self._D_eff_field[mesh.active]))
        dt_diff = (self.diffusion_safety * dx_min ** 2 /
                   (2.0 * ndim * D_max)) if D_max > 0 else np.inf

        dt = min(dt_adv, dt_diff)
        courant = v_max * dt / dx_min
        diff_num = 2.0 * ndim * D_max * dt / dx_min ** 2
        return dt, courant, diff_num, dt_adv, dt_diff

    # =======================================================================
    #  SINGLE EXPLICIT STEP (conservative flux balance)
    # =======================================================================
    def step(self, dt, c_inject):
        """
        Advance the tracer field by one explicit forward-Euler step of size dt.

        Implements the conservative flux balance:
            C_P^{n+1} = C_P^n + (dt/(phi_P V_P)) * sum_f (-adv_f - disp_f)

        with first-order upwind advection and central-difference dispersion.
        Returns the new total tracer mass [kg] and the flux-weighted outlet
        concentration (for breakthrough curves).
        """
        mesh, props = self.mesh, self.props
        C = self.C
        V = mesh.cell_volume

        # net mass rate into each cell = -sum(out fluxes) [kg/s]; we accumulate
        # per-cell net inflow then apply the time update.
        net = np.zeros((mesh.nx, mesh.ny, mesh.nz))

        D = self._D_eff_field

        def disp_coeff_face(a, b, axis_spacing, area):
            # (phi D)_face with arithmetic mean (smooth field) * A / d
            phiD_a = props.phi[a] * D[a]
            phiD_b = props.phi[b] * D[b]
            phiD_face = 0.5 * (phiD_a + phiD_b)
            return phiD_face * area / axis_spacing

        # ---- Interior face loop (each +face handled once, applied to both) --
        for dof in range(mesh.n_active):
            i, j, k = mesh.dof_to_ijk[dof]

            # +x face between (i,j,k) and (i+1,j,k)
            if self.has_x[i, j, k]:
                P = (i, j, k); N = (i + 1, j, k)
                Q = self.Qx[i, j, k]                  # +ve => flow P->N
                # Upwind advected concentration.
                C_up = C[P] if Q >= 0 else C[N]
                adv = Q * C_up                        # kg/s, leaving P toward N
                disp_T = disp_coeff_face(P, N, mesh.dx, mesh.area_x)
                disp = disp_T * (C[N] - C[P])         # kg/s, +ve => into P? sign:
                # dispersive flux from P to N is -disp_T*(C_N - C_P)*... define
                # flux_PtoN = -disp_T*(C_P - C_N) = disp_T*(C_N - C_P) ... use:
                flux_PtoN = adv - disp                # net kg/s P->N
                net[P] -= flux_PtoN                   # P loses what goes to N
                net[N] += flux_PtoN                   # N gains it

            # +y face
            if self.has_y[i, j, k]:
                P = (i, j, k); N = (i, j + 1, k)
                Q = self.Qy[i, j, k]
                C_up = C[P] if Q >= 0 else C[N]
                adv = Q * C_up
                disp_T = disp_coeff_face(P, N, mesh.dy, mesh.area_y)
                disp = disp_T * (C[N] - C[P])
                flux_PtoN = adv - disp
                net[P] -= flux_PtoN
                net[N] += flux_PtoN

            # +z face
            if self.has_z[i, j, k]:
                P = (i, j, k); N = (i, j, k + 1)
                Q = self.Qz[i, j, k]
                C_up = C[P] if Q >= 0 else C[N]
                adv = Q * C_up
                disp_T = disp_coeff_face(P, N, mesh.dz, mesh.area_z)
                disp = disp_T * (C[N] - C[P])
                flux_PtoN = adv - disp
                net[P] -= flux_PtoN
                net[N] += flux_PtoN

        # ---- Boundary fluxes on x end-faces --------------------------------
        # Inlet (i=0): Dirichlet C=c_inject. The inlet face advective flux
        # carries c_inject in. The inlet Darcy flux per cell = injected Q split
        # uniformly; we read it from the cell's own +x interior continuity:
        # for an incompressible field the inlet face flux equals the sum of the
        # cell's outgoing interior fluxes. Simpler & exact: inlet face flux =
        # (cell Darcy ux at i=0) * A_x, directed +x (into the core).
        ux = self.vel.ux
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                if mesh.active[0, j, k]:
                    q_in = ux[0, j, k] * mesh.area_x      # +ve into core
                    # advective inflow carries injected concentration
                    net[0, j, k] += q_in * c_inject
                    # (dispersive inlet flux neglected: advection-dominated
                    #  inlet; documented simplification)

        # Outlet (i=nx-1): zero-gradient outflow. Tracer leaves advectively at
        # the cell's own concentration: flux_out = ux*A_x*C_cell (>=0).
        i_out = mesh.nx - 1
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                if mesh.active[i_out, j, k]:
                    q_out = ux[i_out, j, k] * mesh.area_x  # +ve out of core
                    net[i_out, j, k] -= q_out * C[i_out, j, k]

        # ---- Apply explicit update (per active cell) -----------------------
        phi = props.phi
        upd = np.zeros_like(net)
        upd[mesh.active] = dt * net[mesh.active] / (phi[mesh.active] * V)
        self.C[mesh.active] = C[mesh.active] + upd[mesh.active]

        # Numerical safety: clip tiny negative round-off (upwind is monotone so
        # any negativity is float noise, not scheme failure).
        np.clip(self.C, 0.0, None, out=self.C, where=mesh.active)

        self.time += dt

        # ---- Diagnostics ---------------------------------------------------
        total_mass = float(np.sum(phi[mesh.active] * V * self.C[mesh.active]))
        # Flux-weighted outlet concentration (breakthrough signal).
        num = den = 0.0
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                if mesh.active[i_out, j, k]:
                    q = ux[i_out, j, k] * mesh.area_x
                    num += q * self.C[i_out, j, k]
                    den += q
        outlet_C = num / den if den > 0 else 0.0
        return total_mass, outlet_C

    # =======================================================================
    #  DRIVER
    # =======================================================================
    def run(self, total_time=None, n_pore_volumes=None, c_inject=1.0,
            verbose=True, monitor_every=50, snapshot_pvs=None):
        """
        Time-march the tracer to a target simulation time, or to a target
        number of injected pore volumes (PV) -- the natural clock for core
        floods. 1 PV = pore volume / volumetric rate.

        Recomputes the stable dt each step (here the velocity field is steady,
        so dt is effectively constant, but the per-step recompute is the
        correct pattern for when properties evolve in later phases).

        snapshot_pvs : optional list of pore-volume marks at which to capture a
            copy of the concentration field (for mid-flood visualization). The
            captured snapshots are returned as a list of (label, grid) tuples.
        """
        mesh, props = self.mesh, self.props
        pore_volume = float(np.sum(props.phi[mesh.active]) * mesh.cell_volume)
        # Total injected volumetric rate (sum of inlet face fluxes).
        Q_total = 0.0
        for j in range(mesh.ny):
            for k in range(mesh.nz):
                if mesh.active[0, j, k]:
                    Q_total += self.vel.ux[0, j, k] * mesh.area_x

        if n_pore_volumes is not None:
            total_time = n_pore_volumes * pore_volume / Q_total

        if total_time is None:
            raise ValueError("Provide either total_time or n_pore_volumes.")

        # Snapshot scheduling: convert PV marks to times, capture when crossed.
        snapshots = []
        pending = sorted(snapshot_pvs) if snapshot_pvs else []
        pending_times = [pv * pore_volume / Q_total for pv in pending]

        if verbose:
            print(f"[transport] pore volume = {pore_volume:.4e} m^3, "
                  f"Q = {Q_total:.4e} m^3/s, "
                  f"1 PV = {pore_volume / Q_total:.2f} s")
            dt0, c0, d0, da, dd = self.stable_timestep()
            print(f"[transport] stable dt = {dt0:.3e} s "
                  f"(adv-limit {da:.3e}, diff-limit {dd:.3e}); "
                  f"Courant={c0:.3f}, diffusion#={d0:.3f}")
            limiter = "advection" if da < dd else "diffusion"
            print(f"[transport] timestep limited by: {limiter}")

        step_idx = 0
        while self.time < total_time - 1e-15:
            dt, courant, diff_num, _, _ = self.stable_timestep()
            dt = min(dt, total_time - self.time)   # land exactly on target
            mass, outlet_C = self.step(dt, c_inject)

            # Capture any snapshots whose time has been reached this step.
            while pending_times and self.time >= pending_times[0] - 1e-12:
                pv_mark = pending.pop(0)
                pending_times.pop(0)
                snapshots.append((f"{pv_mark:.2f} PV", self.C.copy()))

            self.history["time"].append(self.time)
            self.history["dt"].append(dt)
            self.history["courant"].append(courant)
            self.history["diffusion_number"].append(diff_num)
            self.history["mass"].append(mass)
            self.history["outlet_C"].append(outlet_C)

            if verbose and (step_idx % monitor_every == 0):
                pv = self.time * Q_total / pore_volume
                print(f"  step {step_idx:5d}  t={self.time:8.2f}s  "
                      f"PV={pv:5.3f}  Courant={courant:.3f}  "
                      f"outletC={outlet_C:.4f}")
            step_idx += 1

        if verbose:
            pv = self.time * Q_total / pore_volume
            print(f"[transport] done: {step_idx} steps, t={self.time:.2f}s "
                  f"({pv:.3f} PV), outletC={self.history['outlet_C'][-1]:.4f}")
        return self.C, snapshots


# ===========================================================================
#  IMPLICIT SINGLE-TRACER SOLVER (unconditionally stable, backward Euler)
# ===========================================================================
class ImplicitTracerTransport:
    """
    Backward-Euler implicit solver for the passive tracer, built on the shared
    sparse TransportOperator. Unconditionally stable: dt is limited by ACCURACY,
    not stability, so large steps (high Courant) are admissible.

    Per step we solve  M C^{n+1} = r  with
        M = diag(phi V / dt) - L + diag(outlet_flux)
        r = (phi V / dt) C^n + inlet_flux * C_inj

    M is assembled ONCE (steady velocity) and only its storage diagonal changes
    with dt; if dt is held constant, M is fully constant and we factorize once.

    Two solve paths:
      * 'direct'  : sparse LU (scipy splu) -- exact, factorized once.
      * 'bicgstab': iterative, for large/multi-species systems, with residual
                    and iteration-count monitoring.
    """

    def __init__(self, mesh, properties, velocity,
                 longitudinal_dispersivity=1.0e-3,
                 transverse_dispersivity=1.0e-4,
                 molecular_diffusion=1.0e-9,
                 solver="direct", iterative_tol=1.0e-10):
        from physics.transport_operator import TransportOperator
        self.mesh = mesh
        self.props = properties
        self.vel = velocity
        self.alpha_L = float(longitudinal_dispersivity)
        self.D_m = float(molecular_diffusion)
        self.solver = solver
        self.iterative_tol = float(iterative_tol)

        # Effective (isotropic-mechanical) dispersion field, as in Phase 2.
        v_mag = velocity.speed / np.maximum(properties.phi, 1e-6)
        self.D_eff = self.D_m + self.alpha_L * v_mag

        self.op = TransportOperator(mesh, properties, velocity, self.D_eff)
        self.storage = self.op.storage_diagonal()         # phi V per DOF
        self.outlet = self.op.outlet_diagonal()           # outlet flux per DOF

        # State.
        self.c = np.zeros(mesh.n_active)                  # compact DOF vector
        self.C = self.op.to_grid(self.c)
        self.time = 0.0
        self._dt_factorized = None
        self._lu = None

        self.history = {"time": [], "dt": [], "courant": [],
                        "mass": [], "outlet_C": [],
                        "lin_residual": [], "lin_iters": []}

    # -----------------------------------------------------------------------
    def _build_M(self, dt):
        """M = diag(phi V/dt) - L + diag(outlet_flux)."""
        from scipy.sparse import diags
        n = self.mesh.n_active
        storage_diag = self.storage / dt
        M = (diags(storage_diag) - self.op.L + diags(self.outlet)).tocsc()
        return M

    def _courant(self, dt):
        mesh = self.mesh
        dx_min = min(mesh.dx, mesh.dy, mesh.dz)
        v_mag = self.vel.speed[mesh.active] / np.maximum(
            self.props.phi[mesh.active], 1e-6)
        v_max = float(np.max(v_mag)) if v_mag.size else 0.0
        return v_max * dt / dx_min

    # -----------------------------------------------------------------------
    def step(self, dt, c_inject):
        """One implicit step. Returns (total_mass, outlet_C, lin_residual, iters)."""
        from scipy.sparse.linalg import splu

        # Right-hand side.
        r = (self.storage / dt) * self.c + self.op.inlet_rhs(c_inject)

        if self.solver == "direct":
            # Factorize once per distinct dt (here dt is constant -> once).
            if self._lu is None or self._dt_factorized != dt:
                M = self._build_M(dt)
                self._lu = splu(M)
                self._dt_factorized = dt
                self._M_cache = M
            c_new = self._lu.solve(r)
            # Report the true linear residual for monitoring.
            res = float(np.linalg.norm(self._M_cache @ c_new - r) /
                        (np.linalg.norm(r) + 1e-30))
            iters = 1
        else:  # iterative: restarted GMRES (robust for non-symmetric M)
            from scipy.sparse.linalg import gmres, LinearOperator
            M = self._build_M(dt)
            # Jacobi (diagonal) preconditioner: M is a diagonally-dominant
            # M-matrix, so the inverse diagonal is an excellent cheap
            # preconditioner that sharply cuts iteration counts. GMRES is used
            # rather than BiCGSTAB because BiCGSTAB can break down (stagnate)
            # when the RHS norm is tiny, e.g. the first near-empty step; GMRES
            # is breakdown-free and monotone in residual.
            d = M.diagonal()
            Minv = LinearOperator(M.shape, matvec=lambda x: x / d)
            counter = {"n": 0}
            def cb(rk):
                counter["n"] += 1
            c_new, info = gmres(M, r, x0=self.c, M=Minv,
                                rtol=self.iterative_tol, atol=1e-14,
                                restart=50, maxiter=5000, callback=cb,
                                callback_type="pr_norm")
            res = float(np.linalg.norm(M @ c_new - r) /
                        (np.linalg.norm(r) + 1e-30))
            iters = counter["n"]

        # Monotone clip of float round-off (M-matrix guarantees boundedness).
        np.clip(c_new, 0.0, None, out=c_new)
        self.c = c_new
        self.time += dt

        # Diagnostics.
        total_mass = float(np.sum(self.storage * self.c))
        # Flux-weighted outlet concentration.
        num = float(np.sum(self.outlet * self.c))
        den = float(np.sum(self.outlet))
        outlet_C = num / den if den > 0 else 0.0
        return total_mass, outlet_C, res, iters

    # -----------------------------------------------------------------------
    def run(self, dt, n_pore_volumes=None, total_time=None, c_inject=1.0,
            verbose=True, monitor_every=20):
        """
        March with a FIXED user-chosen dt (the whole point of implicit: dt is an
        accuracy choice, free of the stability limit). Provide n_pore_volumes or
        total_time as the stopping clock.
        """
        mesh, props = self.mesh, self.props
        pore_volume = float(np.sum(props.phi[mesh.active]) * mesh.cell_volume)
        Q_total = float(np.sum(self.op._inlet_face_flux))
        if n_pore_volumes is not None:
            total_time = n_pore_volumes * pore_volume / Q_total
        if total_time is None:
            raise ValueError("Provide n_pore_volumes or total_time.")

        courant = self._courant(dt)
        if verbose:
            print(f"[implicit] dt={dt:.3e}s  Courant={courant:.3f} "
                  f"(unconditionally stable)  solver={self.solver}")
            print(f"[implicit] 1 PV = {pore_volume/Q_total:.2f}s, "
                  f"target = {total_time:.2f}s")

        step_idx = 0
        while self.time < total_time - 1e-12:
            step_dt = min(dt, total_time - self.time)
            mass, outlet_C, res, iters = self.step(step_dt, c_inject)
            self.history["time"].append(self.time)
            self.history["dt"].append(step_dt)
            self.history["courant"].append(self._courant(step_dt))
            self.history["mass"].append(mass)
            self.history["outlet_C"].append(outlet_C)
            self.history["lin_residual"].append(res)
            self.history["lin_iters"].append(iters)
            if verbose and step_idx % monitor_every == 0:
                pv = self.time * Q_total / pore_volume
                print(f"  step {step_idx:4d}  PV={pv:5.3f}  "
                      f"Courant={self._courant(step_dt):6.3f}  "
                      f"outletC={outlet_C:.4f}  "
                      f"lin_res={res:.1e} iters={iters}")
            step_idx += 1

        self.C = self.op.to_grid(self.c)
        if verbose:
            pv = self.time * Q_total / pore_volume
            print(f"[implicit] done: {step_idx} steps, {pv:.3f} PV, "
                  f"outletC={self.history['outlet_C'][-1]:.4f}")
        return self.C
