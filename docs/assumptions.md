# ASPiRe-3D — Assumptions & Limitations

## Physical assumptions
- Single-phase incompressible aqueous flow; isotropic permeability.
- Isothermal; temperature enters only through an Arrhenius multiplier.
- Lumped surrogate scaling chemistry (salinity×alkali saturation index), not
  full thermodynamic speciation (deliberately deferred).
- pH represented through alkali concentration as a surrogate.
- Mechanical-isotropic dispersion; homogeneous baseline rock properties.
- Adsorption/retention as linear-driving-force kinetics toward equilibrium
  isotherms; polymer retention largely irreversible; RRF a bounded
  multiplicative permeability penalty.
- Sequential (first-order) operator splitting between transport, reaction, and
  damage.

## Numerical limitations
- First-order upwind advection adds numerical diffusion (monotone, convergent;
  report with grid-refinement where front sharpness matters).
- Operator splitting is first-order in time; stiff reactions handled by
  stiffness-controlled sub-stepping (diagnostic reported).
- Damage is irreversible in permeability except through explicit dissolution.
- Staircased cylindrical boundary from Cartesian masking.

## Calibration limitations
- Identifiability is dataset-dependent; the diagnostics flag non-identifiable
  parameters (e.g. Langmuir q_max/K_L from a near-saturated effluent, or the
  porosity-permeability exponent when RRF dominates).
- Linearized (Jacobian) confidence intervals assume locally-Gaussian residuals;
  the Bayesian path is provided where that is doubtful.
- R² is unreliable for near-constant plateau signals; judge those by RMSE
  (reports auto-flag this).

## Data limitations
- Quantitative reservoir claims require real core-flood data; the loader is
  ready and the machinery is validated against synthetic truth.
- Cleaning thresholds (outlier σ, smoothing) are user-tunable and logged; their
  effect on the match should be reported as a robustness check.

## Explicitly out of scope (by design, for this stage)
- Geochemical equilibrium/speciation, ion exchange, multiphase flow, thermal
  coupling, machine-learning surrogates, and any GUI. The architecture is
  extensible toward these without disturbing the validated transport core.
