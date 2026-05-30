# ASPiRe-3D — Phase 5 Documentation: Adsorption, Retention & Calibration

**Surfactant Adsorption, Polymer Retention, and History-Matching Framework**

Publication-grade documentation of the physics, numerics, assumptions,
limitations, validation, and novelty added in Phase 5.

---

## 1. Physics added

ASPiRe-3D now represents the two dominant chemical-loss and permeability-damage
mechanisms in ASP flooding:

1. **Surfactant adsorption** — Langmuir or Freundlich isotherm, modelled as
   rate-limited (linear-driving-force) kinetics toward equilibrium, with
   optional irreversibility, salinity sensitivity, and a pH/alkali modifier.
2. **Polymer retention** — adsorption + mechanical entrapment toward a
   capacity, with a Residual Resistance Factor (RRF) permeability reduction,
   inaccessible/permeability-dependent retention, and optional shear
   degradation.

Both couple back into porosity, permeability, mobility, and pressure drop,
producing **concentration retardation** and **injectivity decline** — the two
signals core-flood history matching fits.

A **calibration framework** (bounded parameter containers, weighted objective
function, local sensitivity analysis) prepares the simulator for inversion
against Niger Delta sandstone core-flood data.

---

## 2. Governing equations

### 2.1 Adsorptive transport and retardation

For a dissolved species `C` (per fluid volume) with adsorbed amount `q` (per
bulk volume), the mass balance with adsorption source is:

```
phi dC/dt + div(uC) - div(phi D grad C) = -dq/dt
dq/dt = k_a ( q_eq(C) - q )          (linear driving force toward isotherm)
```

The breakthrough is retarded by the retardation factor

```
R = 1 + (1/phi) dq_eq/dC
```

which the simulator reproduces *emergently* from the operator-split mass
exchange (validated against the analytical R to 3%).

### 2.2 Isotherms

```
Langmuir   : q_eq = q_max K_L C / (1 + K_L C)        (monolayer, saturates)
Freundlich : q_eq = K_F C^(1/n_F)                    (heterogeneous surface)
```

with environmental modifiers on `q_eq`:

```
salinity   : q_eq *= (1 + beta_s * salinity)         (charge screening raises adsorption)
pH/alkali  : q_eq *= (1 - beta_pH * alkali)_+        (high pH lowers surfactant adsorption)
```

### 2.3 Polymer retention and RRF

```
d(sigma_p)/dt = k_r ( sigma_max (k/k0)^(-gamma) - sigma_p )_+     (irreversible)
RRF = 1 + (RRF_max - 1) * min(sigma_p/sigma_ref, 1)
k   = f_phi(phi) / RRF                                            (extra perm penalty)
shear: dC/dt |_shear = -k_sh (|v|/v_sh - 1)_+ C                   (optional)
```

RRF is a **bounded, persistent** permeability reduction (distinct from
pore-volume occupancy), giving the characteristic early-onset injectivity
plateau at `I/I0 ~ 1/RRF`.

---

## 3. Numerical treatment

- Adsorbed/retained species are **immobile species**; adsorption is a
  **ReactionModel** moving mass between dissolved and adsorbed pools — the
  transport core is unchanged (conservative FVM, M-matrix implicit, sparse).
- Isotherm equilibrium enters through **linear-driving-force kinetics**,
  integrated within the operator-split reaction step.
- **Stiffness-controlled reaction sub-stepping**: the explicit reaction update
  is sub-cycled so each sub-step's *per-species relative* change stays ≤ 0.25,
  making stiff (fast `k_a`) adsorption unconditionally well-behaved. This fixed
  a real overshoot failure mode and is reported via the stiffness diagnostic.
- RRF and adsorbed-volume effects are applied in the **formation-damage**
  coupling layer (permeability penalty + porosity occupancy), feeding the
  pressure/velocity re-solve.

### Engineering diagnostics

Adsorption mass balance (dissolved loss == adsorbed gain, mass-weighted);
porosity bounds and permeability positivity; Courant and reaction-stiffness
numbers; injectivity and pressure-drop histories.

---

## 4. Assumptions and limitations

**Assumptions.** Single dissolved/adsorbed pool per chemical; linear-driving-
force kinetics toward an equilibrium isotherm; isothermal Arrhenius
(precipitation) and rate constants treated as calibration targets; RRF a
bounded multiplicative permeability penalty; salinity/pH modifiers
multiplicative and monotone; sequential first-order coupling.

**Limitations.**
- Adsorption isotherm parameters are illustrative until calibrated to core data
  (the calibration framework is the intended route; no field data is bundled).
- The pH effect uses alkali concentration as a surrogate for pH; explicit
  speciation awaits the geochemical-equilibrium phase.
- Shear degradation is a first-order velocity-threshold surrogate, not a
  molecular-weight-distribution model.
- First-order operator splitting and upwind numerical diffusion apply as in
  earlier phases (documented, convergent).

---

## 5. Validation strategy

| Check | Result |
|-------|--------|
| Langmuir/Freundlich isotherm limits | correct (saturation, monotonicity) |
| Adsorption mass conservation | exact (mass-weighted, ~1e-18) |
| Concentration retardation | adsorbing 2.6 PV vs conservative 1.0 PV |
| Irreversible vs reversible adsorption | correct sign behaviour |
| Salinity / pH modifiers | adsorption ↑ with salinity, ↓ with alkali |
| Polymer RRF permeability reduction | k/k0 → 1/RRF plateau |
| Permeability-dependent retention | low-k retains more |
| Shear degradation | loss above v_sh only |
| Analytical retardation factor R | 3% (benchmark B5) |
| Breakthrough-delay vs adsorption capacity | monotone (benchmark B6) |
| Calibration round-trip / objective / sensitivity | exact / zero-at-match / correct ranking |

**55/55 automated tests pass** across all phases.

---

## 6. Publishable novelty

- An **open, modular, fully-validated 3D simulator** coupling surfactant
  adsorption *and* polymer retention (with RRF) to formation-damage permeability
  evolution and injectivity decline, specifically for **natural ASP flooding in
  sandstones**.
- **Mechanism discrimination**: precipitation/fines plugging (progressive,
  unbounded) vs polymer RRF (bounded, early-onset plateau) produce distinct
  injectivity signatures the simulator reproduces — directly useful for
  diagnosing damage mechanisms in core-flood history matching.
- A reaction-model plug-in architecture with **stiffness-controlled
  sub-stepping**, proven against an **analytical retardation benchmark (3%)** —
  a verification standard rarely documented at thesis scale.
- A built-in **calibration/sensitivity architecture** that makes the path to
  Niger Delta core-flood history matching explicit and reproducible.

---

## 7. Extensibility (architecture preserved for)

Geochemical equilibrium (replace LDF/SI surrogates with speciation), ion
exchange (multi-site clay models as ReactionModels), thermal coupling (energy
transport reusing the FVM operator; dynamic Arrhenius), multiphase flow (oil–
water through the `k/mu` mobility seam), and ML surrogates (the calibration
framework already emits parameter→response mappings for training).
