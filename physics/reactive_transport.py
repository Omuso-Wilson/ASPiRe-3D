"""
ASPiRe-3D : core/reactive_transport.py
===============================================================================
Multi-species reactive transport BACKBONE.

This is the extensible core onto which formation-damage physics (precipitation,
fines migration, permeability damage) will later attach. It steps N species
through the SHARED sparse advection-dispersion operator and applies a pluggable
reaction model by OPERATOR SPLITTING.

SPECIES HANDLED (Phase 3 demonstration set)
-------------------------------------------
    salinity        (mobile)  -- key brine ion / TDS surrogate
    alkali          (mobile)  -- 'A' of ASP
    surfactant      (mobile)  -- 'S' of ASP
    polymer         (mobile)  -- 'P' of ASP (would raise viscosity later)
    fines_suspended (mobile)  -- detached fines travelling in suspension
    precipitate     (IMMOBILE)-- scale attached to rock (transport skipped)

Phase 3 uses the NullReactionModel => pure conservative transport (no chemistry
yet), per the stated objective: establish a stable, extensible backbone BEFORE
formation-damage coupling.

NUMERICAL FORMULATION
---------------------
Per species s, the conservative ADE (with a reaction source r_s) is:

        phi dC_s/dt + div(u C_s) - div(phi D_s grad C_s) = r_s

Spatial discretization: the SAME finite-volume operator validated in Phase 2b
(first-order upwind advection + central dispersion, M-matrix implicit form).
Each MOBILE species reuses the operator structure; only its inlet value and
dispersion field differ. IMMOBILE species skip transport entirely.

TIME INTEGRATION -- SEQUENTIAL OPERATOR SPLITTING (Lie splitting):
    step 1 (transport): advance each mobile species by dt using the implicit
                        transport solve  M_s C_s^{*} = r_s(C_s^n, inlet)
    step 2 (reaction):  integrate dC/dt = reaction_rates(C^{*}) over dt for all
                        species (mobile and immobile), explicitly.
This cleanly separates the (linear) transport solve from the (generally
nonlinear) reaction update, so each is independently testable and replaceable.
Splitting is first-order accurate in dt; Strang (2nd-order) splitting is a
documented future refinement.

STATE LAYOUT
------------
    self.C  : (n_species, n_active) concentration matrix in compact DOF order.
Sparse, per-species transport matrices are assembled once (steady velocity);
with a fixed dt they are factorized once and reused across all steps and
species that share the same dispersion field.
===============================================================================
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from physics.transport_operator import TransportOperator
from physics.species import SpeciesRegistry, NullReactionModel


class ReactiveTransport:
    """
    Backbone solver: N species, shared FVM transport operator, pluggable
    reaction model, implicit time stepping with operator splitting.
    """

    def __init__(self, mesh, properties, velocity, registry,
                 reaction_model=None,
                 longitudinal_dispersivity=1.0e-3):
        """
        Parameters
        ----------
        mesh, properties, velocity : simulator context (velocity .compute()'d)
        registry : SpeciesRegistry
        reaction_model : ReactionModel or None (defaults to NullReactionModel)
        longitudinal_dispersivity : alpha_L [m] used in mechanical dispersion.
        """
        self.mesh = mesh
        self.props = properties
        self.vel = velocity
        self.registry = registry
        self.reaction = reaction_model or NullReactionModel()
        self.alpha_L = float(longitudinal_dispersivity)

        n_species = len(registry)
        n_active = mesh.n_active

        # Interstitial speed field (for mechanical dispersion D = D_m + aL|v|).
        self._v_mag = velocity.speed / np.maximum(properties.phi, 1e-6)

        # ---- Build a transport operator PER DISTINCT dispersion field ------
        # In Phase 3 species differ in molecular diffusion D_m. We build one
        # operator per species (cheap; they share structure). Each operator
        # holds L, inlet/outlet flux vectors, and the storage diagonal.
        self._ops = {}            # species_name -> TransportOperator
        for s in registry:
            if not s.mobile:
                continue
            D_eff = s.molecular_diffusion + self.alpha_L * self._v_mag
            self._ops[s.name] = TransportOperator(mesh, properties,
                                                  velocity, D_eff)

        # Storage diagonal (phi V) is species-independent; grab from any op or
        # build directly.
        any_op = next(iter(self._ops.values())) if self._ops else None
        if any_op is not None:
            self.storage = any_op.storage_diagonal()
            self.outlet = any_op.outlet_diagonal()
            self._inlet_flux = any_op._inlet_face_flux
        else:
            # No mobile species (degenerate) -- still define storage.
            V = mesh.cell_volume
            self.storage = np.array([properties.phi[tuple(mesh.dof_to_ijk[d])] * V
                                     for d in range(n_active)])
            self.outlet = np.zeros(n_active)
            self._inlet_flux = np.zeros(n_active)

        # ---- State: (n_species, n_active), initialised to resident values --
        self.C = np.zeros((n_species, n_active))
        for s in registry:
            self.C[registry.index(s.name), :] = s.initial_value

        self.time = 0.0
        self._lu = {}             # species_name -> factorized M (per dt)
        self._dt_factorized = None

        # Diagnostics.
        self.history = {"time": [], "dt": [], "courant": [],
                        "species_mass": {s.name: [] for s in registry},
                        "outlet": {s.name: [] for s in registry},
                        "max_lin_residual": []}

    # -----------------------------------------------------------------------
    def _build_M(self, op, dt):
        """M_s = diag(phi V/dt) - L_s + diag(outlet_flux)."""
        return (diags(self.storage / dt) - op.L + diags(self.outlet)).tocsc()

    def _courant(self, dt):
        dx_min = min(self.mesh.dx, self.mesh.dy, self.mesh.dz)
        v_max = float(np.max(self._v_mag[self.mesh.active]))
        return v_max * dt / dx_min

    # -----------------------------------------------------------------------
    def _factorize(self, dt):
        """Factorize each mobile species' transport matrix once for this dt."""
        self._lu = {}
        for name, op in self._ops.items():
            self._lu[name] = splu(self._build_M(op, dt))
        self._dt_factorized = dt

    # -----------------------------------------------------------------------
    def step(self, dt):
        """
        One reactive-transport step via sequential operator splitting.

        Returns the maximum linear-solve residual across species (monitoring).
        """
        reg = self.registry

        # Refactorize only if dt changed (constant dt -> once).
        if self._dt_factorized != dt:
            self._factorize(dt)

        max_res = 0.0

        # ---- STEP 1: TRANSPORT each mobile species -------------------------
        C_star = self.C.copy()
        for s in reg:
            if not s.mobile:
                continue           # immobile: no transport
            idx = reg.index(s.name)
            op = self._ops[s.name]
            c_n = self.C[idx, :]
            # RHS: storage carry-over + advective inlet inflow of this species.
            r = (self.storage / dt) * c_n + op.inlet_rhs(s.inlet_value)
            c_star = self._lu[s.name].solve(r)
            # Monotone clip of round-off (M-matrix => bounded).
            np.clip(c_star, 0.0, None, out=c_star)
            C_star[idx, :] = c_star
            # Linear residual for monitoring.
            M = self._build_M(op, dt)
            res = np.linalg.norm(M @ c_star - r) / (np.linalg.norm(r) + 1e-30)
            max_res = max(max_res, float(res))

        # ---- STEP 2: REACTION update (explicit over dt) --------------------
        rates = self.reaction.rates(C_star, reg, self.mesh, self.props)
        C_new = C_star + dt * rates
        np.clip(C_new, 0.0, None, out=C_new)     # concentrations non-negative

        self.C = C_new
        self.time += dt
        return max_res

    # -----------------------------------------------------------------------
    def run(self, dt, n_pore_volumes=None, total_time=None,
            verbose=True, monitor_every=20):
        """
        March the reactive-transport system with a fixed (implicit-stable) dt.
        """
        mesh, props, reg = self.mesh, self.props, self.registry
        pore_volume = float(np.sum(props.phi[mesh.active]) * mesh.cell_volume)
        Q_total = float(np.sum(self._inlet_flux))
        if n_pore_volumes is not None:
            total_time = n_pore_volumes * pore_volume / Q_total
        if total_time is None:
            raise ValueError("Provide n_pore_volumes or total_time.")

        if verbose:
            courant = self._courant(dt)
            print(f"[reactive] {len(reg)} species, dt={dt:.3e}s, "
                  f"Courant={courant:.2f} (implicit, unconditionally stable)")
            print(f"[reactive] reaction model: {self.reaction.name()}")
            print(f"[reactive] 1 PV = {pore_volume/Q_total:.2f}s, "
                  f"target {total_time:.2f}s")

        step_idx = 0
        while self.time < total_time - 1e-12:
            step_dt = min(dt, total_time - self.time)
            max_res = self.step(step_dt)

            # Diagnostics.
            self.history["time"].append(self.time)
            self.history["dt"].append(step_dt)
            self.history["courant"].append(self._courant(step_dt))
            self.history["max_lin_residual"].append(max_res)
            for s in reg:
                idx = reg.index(s.name)
                self.history["species_mass"][s.name].append(
                    float(np.sum(self.storage * self.C[idx, :])))
                # flux-weighted outlet concentration
                num = float(np.sum(self.outlet * self.C[idx, :]))
                den = float(np.sum(self.outlet))
                self.history["outlet"][s.name].append(num / den if den > 0 else 0.0)

            if verbose and step_idx % monitor_every == 0:
                pv = self.time * Q_total / pore_volume
                outs = "  ".join(
                    f"{s.name[:4]}={self.history['outlet'][s.name][-1]:.3f}"
                    for s in reg if s.mobile)
                print(f"  step {step_idx:4d}  PV={pv:5.3f}  res={max_res:.1e}  {outs}")
            step_idx += 1

        if verbose:
            pv = self.time * Q_total / pore_volume
            print(f"[reactive] done: {step_idx} steps, {pv:.3f} PV")
        return self.C

    # -----------------------------------------------------------------------
    def concentration_grid(self, species_name):
        """Return the full-grid (nx,ny,nz) concentration of a named species."""
        idx = self.registry.index(species_name)
        op = next(iter(self._ops.values())) if self._ops else None
        mesh = self.mesh
        C = np.full((mesh.nx, mesh.ny, mesh.nz), np.nan)
        for dof in range(mesh.n_active):
            i, j, k = mesh.dof_to_ijk[dof]
            C[i, j, k] = self.C[idx, dof]
        return C
