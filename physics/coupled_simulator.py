"""
ASPiRe-3D : core/coupled_simulator.py
===============================================================================
Coupled formation-damage reactive transport simulator (Phase 4 orchestrator).

This is the top-level driver that closes the formation-damage loop:

    for each timestep:
        (1) TRANSPORT  : advance mobile species on the current Darcy field
                         (implicit, M-matrix, vectorized operators)
        (2) REACT      : precipitation/dissolution + fines migration
                         (operator splitting; pluggable ReactionModel)
        (3) DAMAGE     : precipitate + deposited fines -> porosity reduction
                         -> permeability update (Kozeny-Carman / power / exp)
        (4) RE-FLOW    : if permeability changed enough, re-solve pressure and
                         recompute velocity, then rebuild transport operators
                         (this is what produces injectivity decline)

This is a SEQUENTIAL (loosely) coupled scheme -- the standard, defensible
approach in reactive-transport / formation-damage modelling. It is first-order
accurate in the coupling, and we expose a re-flow tolerance so the (expensive)
pressure re-solve is only triggered when permeability has drifted enough to
matter, balancing fidelity and cost.

ENGINEERING DIAGNOSTICS (recorded every step)
---------------------------------------------
  * species mass conservation residual (transport + reaction bookkeeping),
  * porosity bounds (min stays > floor) and permeability positivity,
  * Courant number (transport stability),
  * reaction stiffness number  Da = max|reaction_rate|*dt / typical_conc
    (a Damkohler-like measure: Da >> 1 warns the reaction step is stiff and the
     explicit splitting may need a smaller dt),
  * injectivity = Q / dP (proportional to overall conductance) and its decline.

The transport CORE is untouched: this orchestrator only composes the existing
modules and the pluggable reaction model.
===============================================================================
"""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu

from physics.pressure_solver import PressureSolver
from physics.velocity import VelocityField
from physics.transport_operator import TransportOperator
from physics.species import NullReactionModel


class CoupledSimulator:
    """Top-level coupled formation-damage reactive transport simulator."""

    def __init__(self, mesh, properties, registry, bc,
                 reaction_model=None,
                 formation_damage=None,
                 longitudinal_dispersivity=5.0e-4,
                 reflow_k_tolerance=0.02,
                 fines_model=None):
        """
        Parameters
        ----------
        mesh, properties, registry, bc : simulator context
        reaction_model : ReactionModel (composite of precip + fines, etc.)
        formation_damage : FormationDamage instance (or None to disable damage)
        longitudinal_dispersivity : alpha_L [m]
        reflow_k_tolerance : re-solve pressure when the mean permeability ratio
            changes by more than this fraction since the last re-flow.
        fines_model : optional FinesMigrationKinetics whose velocity must be
            refreshed after each re-flow (it depends on interstitial velocity).
        """
        self.mesh = mesh
        self.props = properties
        self.registry = registry
        self.bc = bc
        self.reaction = reaction_model or NullReactionModel()
        self.damage = formation_damage
        self.alpha_L = float(longitudinal_dispersivity)
        self.reflow_tol = float(reflow_k_tolerance)
        self.fines_model = fines_model

        # Initial flow solve.
        self.pressure_solver = PressureSolver(mesh, properties)
        self.velocity = VelocityField(mesh, properties)
        self._solve_flow()
        self._last_reflow_kmean = self._mean_active_k()

        # Build one transport operator per mobile species (dispersion differs).
        self._build_operators()

        # State.
        n_species, n_active = len(registry), mesh.n_active
        self.C = np.zeros((n_species, n_active))
        for s in registry:
            self.C[registry.index(s.name), :] = s.initial_value
        self.time = 0.0
        self._dt_lu = None
        self._lu = {}

        # Conserved-mass reference: total (mobile+immobile) per coupled family
        # is checked externally; here we record per-species mass.
        self.history = {
            "time": [], "pv": [], "dt": [],
            "courant": [], "stiffness_Da": [],
            "phi_min": [], "k_ratio_min": [], "k_ratio_mean": [],
            "dP": [], "injectivity": [], "injectivity_ratio": [],
            "species_mass": {s.name: [] for s in registry},
            "outlet": {s.name: [] for s in registry},
            "max_lin_residual": [],
        }
        self._injectivity0 = None

    # =======================================================================
    #  FLOW
    # =======================================================================
    def _solve_flow(self):
        self.P = self.pressure_solver.solve(self.bc, verbose=False)
        self.velocity.compute(self.P, self.bc)

    def _mean_active_k(self):
        return float(self.props.k[self.mesh.active].mean())

    def _build_operators(self):
        self._v_mag = self.velocity.speed / np.maximum(self.props.phi, 1e-6)
        self._ops = {}
        for s in self.registry:
            if not s.mobile:
                continue
            D_eff = s.molecular_diffusion + self.alpha_L * self._v_mag
            self._ops[s.name] = TransportOperator(self.mesh, self.props,
                                                  self.velocity, D_eff)
        any_op = next(iter(self._ops.values())) if self._ops else None
        self.storage = (any_op.storage_diagonal() if any_op is not None
                        else self._fallback_storage())
        self.outlet = (any_op.outlet_diagonal() if any_op is not None
                       else np.zeros(self.mesh.n_active))
        self._inlet_flux = (any_op._inlet_face_flux if any_op is not None
                            else np.zeros(self.mesh.n_active))
        self._dt_lu = None       # force refactorization after operator rebuild

    def _fallback_storage(self):
        mesh, props = self.mesh, self.props
        ijk = mesh.dof_to_ijk
        return props.phi[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * mesh.cell_volume

    def _rebuild_operators_after_damage(self):
        """Permeability changed -> re-solve flow, refresh dispersion & operators."""
        self._solve_flow()
        self._v_mag = self.velocity.speed / np.maximum(self.props.phi, 1e-6)
        for s in self.registry:
            if not s.mobile:
                continue
            D_eff = s.molecular_diffusion + self.alpha_L * self._v_mag
            self._ops[s.name].update(velocity=self.velocity,
                                     dispersion_field=D_eff)
        # Recompute storage (porosity changed) and boundary vectors.
        any_op = next(iter(self._ops.values()))
        self.storage = any_op.storage_diagonal()
        self.outlet = any_op.outlet_diagonal()
        self._inlet_flux = any_op._inlet_face_flux
        self._dt_lu = None
        # Refresh fines model's velocity dependence.
        if self.fines_model is not None:
            self.fines_model.update_velocity(self.velocity, self.props)

    # =======================================================================
    #  DIAGNOSTICS
    # =======================================================================
    def _courant(self, dt):
        dx_min = min(self.mesh.dx, self.mesh.dy, self.mesh.dz)
        return float(np.max(self._v_mag[self.mesh.active])) * dt / dx_min

    def _differential_pressure(self):
        act = self.mesh.active
        return float(np.nanmax(self.P[act]) - np.nanmin(self.P[act]))

    def _injectivity(self):
        """Injectivity ~ Q/dP [m^3/s/Pa]; for constant-rate Q fixed, so this
        falls as dP rises (the injectivity-decline signature)."""
        dP = self._differential_pressure()
        Q = float(np.sum(self._inlet_flux))
        return (Q / dP) if dP > 0 else np.inf, dP, Q

    # =======================================================================
    #  TIME STEP
    # =======================================================================
    def _factorize(self, dt):
        self._lu = {}
        for name, op in self._ops.items():
            M = (diags(self.storage / dt) - op.L + diags(self.outlet)).tocsc()
            self._lu[name] = (splu(M), M)
        self._dt_lu = dt

    def step(self, dt):
        reg = self.registry
        if self._dt_lu != dt:
            self._factorize(dt)

        max_res = 0.0
        # --- (1) TRANSPORT all mobile species ---
        C_star = self.C.copy()
        for s in reg:
            if not s.mobile:
                continue
            idx = reg.index(s.name)
            op = self._ops[s.name]
            lu, M = self._lu[s.name]
            r = (self.storage / dt) * self.C[idx, :] + op.inlet_rhs(s.inlet_value)
            c_star = lu.solve(r)
            np.clip(c_star, 0.0, None, out=c_star)
            C_star[idx, :] = c_star
            res = np.linalg.norm(M @ c_star - r) / (np.linalg.norm(r) + 1e-30)
            max_res = max(max_res, float(res))

        # --- (2) REACT (explicit, with stiffness-controlled sub-stepping) ---
        # The reaction ODE dC/dt = rates(C) is integrated explicitly within the
        # split step. Explicit integration of a (possibly stiff) reaction has
        # its own stability limit; rather than impose a global small dt, we
        # SUB-CYCLE the reaction over n_sub steps chosen so each sub-step keeps
        # the Damkohler number below ~0.5 (forward-Euler-stable for linear
        # driving-force / first-order kinetics). This keeps the scheme
        # unconditionally usable while preserving the operator-splitting
        # interface and any ReactionModel.
        rates0 = self.reaction.rates(C_star, reg, self.mesh, self.props)
        # Stiffness must be measured PER SPECIES relative to that species' own
        # scale: an immobile adsorbed species can have a tiny absolute
        # concentration but a large fractional change per step (the linear-
        # driving-force rate constant). Using a global typical concentration
        # would miss this and let the explicit reaction overshoot. We therefore
        # take the max over species of |rate|*dt / (species scale).
        species_scale = np.maximum(
            np.max(np.abs(C_star), axis=1),
            0.01 * max(1e-9, float(np.max(np.abs(C_star)))))   # (n_species,)
        rel_change = np.max(np.abs(rates0) * dt
                            / species_scale[:, None])
        stiffness_Da = float(rel_change)
        # Sub-cycle so each sub-step's relative change <= 0.25 (comfortably
        # forward-Euler-stable for first-order / linear-driving-force kinetics).
        n_sub = int(max(1, np.ceil(stiffness_Da / 0.25)))
        n_sub = min(n_sub, 5000)         # safety cap
        dt_sub = dt / n_sub
        C_react = C_star.copy()
        for _ in range(n_sub):
            rr = self.reaction.rates(C_react, reg, self.mesh, self.props)
            C_react = C_react + dt_sub * rr
            np.clip(C_react, 0.0, None, out=C_react)
        self.C = C_react
        self._n_reaction_substeps = n_sub

        # --- (3) DAMAGE: porosity + permeability update ---
        dmg = None
        if self.damage is not None:
            dmg = self.damage.apply(self.C)

        # --- (4) RE-FLOW if permeability drifted past tolerance ---
        reflowed = False
        if self.damage is not None:
            kmean = self._mean_active_k()
            rel_change = abs(kmean - self._last_reflow_kmean) / self._last_reflow_kmean
            if rel_change > self.reflow_tol:
                self._rebuild_operators_after_damage()
                self._last_reflow_kmean = kmean
                reflowed = True

        self.time += dt
        return max_res, stiffness_Da, dmg, reflowed

    # =======================================================================
    #  DRIVER
    # =======================================================================
    def run(self, dt, n_pore_volumes=None, total_time=None,
            verbose=True, monitor_every=10):
        mesh, props, reg = self.mesh, self.props, self.registry
        pore_volume = float(np.sum(self.props.phi0[mesh.active]) * mesh.cell_volume)
        Q_total = float(np.sum(self._inlet_flux))
        if n_pore_volumes is not None:
            total_time = n_pore_volumes * pore_volume / Q_total
        if total_time is None:
            raise ValueError("Provide n_pore_volumes or total_time.")

        inj0, dP0, _ = self._injectivity()
        self._injectivity0 = inj0
        if verbose:
            print(f"[coupled] {len(reg)} species, dt={dt:.3e}s, "
                  f"Courant={self._courant(dt):.2f}")
            print(f"[coupled] reaction: {self.reaction.name()}; "
                  f"damage: {self.damage.perm_model if self.damage else 'OFF'}")
            print(f"[coupled] 1 PV = {pore_volume/Q_total:.1f}s; "
                  f"initial dP = {dP0:.3e} Pa, injectivity = {inj0:.3e}")

        step_idx = 0
        while self.time < total_time - 1e-9:
            step_dt = min(dt, total_time - self.time)
            max_res, Da, dmg, reflowed = self.step(step_dt)

            inj, dP, _ = self._injectivity()
            pv = self.time * Q_total / pore_volume
            self.history["time"].append(self.time)
            self.history["pv"].append(pv)
            self.history["dt"].append(step_dt)
            self.history["courant"].append(self._courant(step_dt))
            self.history["stiffness_Da"].append(Da)
            self.history["max_lin_residual"].append(max_res)
            self.history["dP"].append(dP)
            self.history["injectivity"].append(inj)
            self.history["injectivity_ratio"].append(inj / self._injectivity0)
            self.history["phi_min"].append(dmg["phi_min"] if dmg else
                                           float(props.phi[mesh.active].min()))
            self.history["k_ratio_min"].append(dmg["k_ratio_min"] if dmg else 1.0)
            self.history["k_ratio_mean"].append(dmg["k_ratio_mean"] if dmg else 1.0)
            for s in reg:
                idx = reg.index(s.name)
                self.history["species_mass"][s.name].append(
                    float(np.sum(self.storage * self.C[idx, :])))
                den = float(np.sum(self.outlet))
                self.history["outlet"][s.name].append(
                    float(np.sum(self.outlet * self.C[idx, :])) / den if den > 0 else 0.0)

            if verbose and step_idx % monitor_every == 0:
                tag = " [reflow]" if reflowed else ""
                print(f"  step {step_idx:4d} PV={pv:5.3f} "
                      f"k/k0(min)={self.history['k_ratio_min'][-1]:.3f} "
                      f"phi_min={self.history['phi_min'][-1]:.3f} "
                      f"inj/inj0={self.history['injectivity_ratio'][-1]:.3f} "
                      f"Da={Da:.2e} res={max_res:.1e}{tag}")
            step_idx += 1

        if verbose:
            print(f"[coupled] done: {step_idx} steps, {pv:.3f} PV, "
                  f"final k/k0(min)={self.history['k_ratio_min'][-1]:.3f}, "
                  f"injectivity ratio={self.history['injectivity_ratio'][-1]:.3f}")
        return self.C

    # -----------------------------------------------------------------------
    def concentration_grid(self, species_name):
        idx = self.registry.index(species_name)
        mesh = self.mesh
        C = np.full((mesh.nx, mesh.ny, mesh.nz), np.nan)
        ijk = mesh.dof_to_ijk
        C[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = self.C[idx, :]
        return C
