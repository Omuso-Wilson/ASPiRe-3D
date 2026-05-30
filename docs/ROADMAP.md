# ASPiRe-3D — Engineering Roadmap & Phase 4 Physics Report

**Coupled Formation-Damage Reactive Transport**

This document summarises the physics added in Phase 4, the governing equations,
the numerical scheme, the assumptions and limitations, the publishable novelty,
and the industrial applications — written to support thesis examination and
journal submission.

---

## 1. Physics added in Phase 4

ASPiRe-3D has evolved from multi-species transport into a **fully coupled
formation-damage reactive transport simulator**. The new physics:

1. **Precipitation / dissolution kinetics** — a lumped mineral-scale species
   forms where injected alkali mixes with saline formation brine, with
   reversible dissolution where conditions reverse, and an optional Arrhenius
   temperature dependence.
2. **Fines migration** — colloidal fines detach above a critical interstitial
   velocity, travel in suspension, and re-deposit by filtration; deposited
   fines are tracked as an immobile species.
3. **Formation-damage coupling** — precipitate and deposited-fines volumes
   reduce porosity; permeability follows porosity via a selectable law
   (Kozeny–Carman, power-law, or exponential damage); the reduced permeability
   feeds back into the pressure/velocity solution, producing injectivity
   decline.

The result is the closed loop **transport → reaction → damage → re-flow** that
is the scientific core of the thesis.

---

## 2. Governing equations

### 2.1 Flow (unchanged from Phase 1)

Incompressible single-phase Darcy flow with spatially- and temporally-varying
permeability `k(x,t)`:

```
div( (k/mu) grad P ) = -q~ ,    u = -(k/mu) grad P
```

### 2.2 Multi-species reactive transport

For each species `s` with concentration `C_s`, mobility flag `m_s`, dispersion
`D_s`, and reaction source `r_s`:

```
phi dC_s/dt + m_s [ div(u C_s) - div(phi D_s grad C_s) ] = r_s
```

Immobile species (`m_s = 0`) drop the transport terms and evolve by reaction
only. Dispersion uses the isotropic-mechanical model `D_s = D_m,s + alpha_L |v|`,
`v = u/phi`.

### 2.3 Precipitation / dissolution kinetics

A dimensionless saturation index encodes the two triggers:

```
SI = (alkali / alkali_ref) * (salinity / salinity_ref)
```

Net precipitate rate (`c_p`):

```
R_p = f_T [ k_p (SI - 1)_+  -  k_d (1 - SI)_+  H(c_p) ]
H(c_p) = c_p / (c_p + eps)        (dissolution only where precipitate exists)
f_T   = exp( -Ea/R (1/T - 1/T_ref) )    (Arrhenius; Ea=0 disables)
```

with conservative coupling to the dissolved pool:
`r_dissolved = -stoich * R_p`.

### 2.4 Fines migration kinetics

Suspended `c_s` (mobile), deposited `sigma` (immobile):

```
R_det = k_det (|v|/v_c - 1)_+ sigma      (detachment above critical velocity)
R_dep = k_dep |v| c_s                    (filtration deposition)
dc_s/dt|_rxn = R_det - R_dep ,   dsigma/dt = R_dep - R_det   (mass conserving)
```

### 2.5 Formation-damage coupling

```
phi = clip( phi0 - c_p/rho_p - sigma/rho_f , phi_min , 1 )
k   = f(phi) ,   f in { Kozeny-Carman, power-law, exponential }     (k > 0)
```

Kozeny–Carman: `k/k0 = (phi/phi0)^3 ((1-phi0)/(1-phi))^2`.

---

## 3. Numerical scheme

- **Spatial discretization**: cell-centred finite volume; first-order upwind
  advection (monotone), central-difference dispersion; harmonic-mean face
  permeability for flow. Conservative by construction.
- **Transport time integration**: implicit backward Euler; the system matrix is
  a non-symmetric **M-matrix**, guaranteeing monotone, bounded concentrations
  at any timestep. Sparse direct (LU) or preconditioned GMRES solves.
- **Reaction integration**: explicit over the step, applied by **sequential
  (Lie) operator splitting** after transport. First-order in time.
- **Damage / re-flow coupling**: after reaction, porosity and permeability are
  updated; the pressure–velocity field and transport operators are rebuilt when
  the mean permeability has drifted beyond a tolerance (vectorized assembly
  keeps this affordable).
- **Performance**: operator assembly is fully vectorized (face-topology arrays +
  single sparse build), so the per-damage-step operator rebuild is inexpensive.

### Engineering diagnostics (recorded every step)

- discrete mass-conservation residuals (transport + reaction couples),
- porosity-bounds enforcement (`phi >= phi_min`) and permeability positivity,
- Courant number (transport stability),
- reaction stiffness `Da = max|rate| dt / typical_conc` (a Damköhler-like
  warning that the explicit reaction step is becoming stiff),
- injectivity `I = Q/Δp` and its decline ratio `I/I0`.

---

## 4. Assumptions and limitations

**Assumptions.** Single-phase incompressible aqueous flow; isotropic
permeability; isothermal (temperature enters only as a constant Arrhenius
multiplier); lumped surrogate scaling chemistry (a single precipitate fed by a
salinity×alkali saturation index rather than a full ion-activity product);
mechanical-isotropic dispersion; staircased cylinder boundary; sequential
(first-order) coupling between transport, reaction, and damage.

**Limitations (and the honest caveats for examination).**
- The saturation index is a transparent *surrogate*, not thermodynamic
  equilibrium — deliberately so; equilibrium chemistry is the next phase.
- First-order upwinding adds numerical diffusion that smears sharp fronts; it is
  monotone and safe, but quantitative front sharpness should be reported with a
  grid-refinement note (the convergence test already demonstrates the trend).
- Operator splitting is first-order in time; for very stiff reactions the
  stiffness diagnostic flags when a smaller step or Strang/implicit-reaction
  coupling would be warranted.
- Damage is irreversible in permeability except through explicit dissolution;
  no mechanical re-opening of plugged pores is modelled.
- Rate and dispersivity parameters are illustrative; quantitative use requires
  calibration to core-flood data (the intended validation route).

---

## 5. Publishable novelty

- A **modular, open, reproducible** 3D coupled formation-damage reactive
  transport simulator targeted specifically at **natural ASP flooding in
  sandstones**, with a clean reaction-model plug-in architecture and a verified
  conservative M-matrix transport core.
- Simultaneous treatment of **two coupled damage mechanisms** — alkali/salinity-
  triggered precipitation *and* critical-velocity fines migration — feeding a
  shared porosity→permeability damage law, with quantitative injectivity-decline
  output.
- A **verification-and-validation suite** (42 automated physics tests including
  analytical Ogata–Banks agreement to <1% and exact Kozeny–Carman reproduction)
  of a standard rarely matched in thesis-scale reservoir codes — strengthening
  reproducibility claims for publication.
- A demonstrated path from **mechanistic kinetics to field-relevant injectivity
  curves** on standard hardware, suitable as a screening tool complementary to
  commercial simulators.

---

## 6. Industrial applications

- **ASP / chemical-EOR screening**: predicting injectivity loss and formation
  damage risk for candidate ASP formulations and salinity designs before field
  trials.
- **Injectivity-decline diagnosis**: matching laboratory core-flood Δp histories
  to attribute damage between scaling and fines plugging.
- **Water-injection & low-salinity studies**: the critical-velocity fines model
  and salinity transport apply directly to injection-water compatibility and
  low-salinity EOR.
- **Scale-management planning**: temperature- and chemistry-sensitive
  precipitation supports scaling-tendency assessment in mixing zones.
- **Niger Delta sandstone reservoirs** specifically (the thesis target): a
  calibratable, defensible tool for formation-damage prediction during natural
  ASP flooding.

---

## 7. Forward roadmap (architecture is ready for)

The reaction-model interface and per-cell property arrays make the following
incremental, each as a new `ReactionModel` subclass or property update with **no
transport-core changes**:

1. **Adsorption** — surfactant/polymer retention isotherms (Langmuir) as a
   reaction sink + immobile adsorbed species.
2. **Ion exchange** — multi-site exchange on clays, coupling cations to fines
   stability.
3. **Geochemical equilibrium** — replace the surrogate SI with an
   ion-activity-product / Ksp speciation solver (operator-split with kinetics).
4. **Multiphase flow** — extend the flow core to oil–water with relative
   permeability; mobility already factored through the `k/mu` seam.
5. **Thermal coupling** — an energy-balance transport equation reusing the same
   FVM operator; temperature then drives the Arrhenius factor dynamically.
6. **ML surrogate models** — the simulator can generate labelled
   (design → injectivity-decline) datasets for surrogate training and
   history-matching acceleration.
```
Phase 1  flow core ✅
Phase 2  passive transport ✅
Phase 2b implicit + stability hardening ✅
Phase 3  multi-species backbone ✅
Phase 4  coupled formation-damage reactive transport ✅
Phase 5  adsorption, retention & calibration framework ✅
Next     geochemical equilibrium → ion exchange → multiphase
         → thermal → ML surrogates
```
