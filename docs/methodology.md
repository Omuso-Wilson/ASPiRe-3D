# ASPiRe-3D — Methodology

## Scope
A research-grade simulator for formation damage during natural ASP flooding in
Niger Delta sandstone cores. It couples single-phase Darcy flow, multi-species
advection–dispersion transport, reaction kinetics (precipitation, fines),
adsorption/retention (surfactant Langmuir/Freundlich, polymer RRF), and
porosity→permeability damage, with optimizer-driven history matching,
sensitivity, and uncertainty quantification.

## Numerical method
Cell-centred finite volume on a structured grid masked to a cylinder.
Conservative flux balances; harmonic-mean face permeabilities; implicit sparse
pressure solve; first-order upwind advection + central dispersion; implicit
backward-Euler transport (M-matrix → bounded, stable); sequential operator
splitting for reactions with stiffness-controlled sub-stepping; permeability
feedback through a pressure/velocity re-solve. See `equations.md`.

## Calibration methodology
A `ForwardModel` maps parameters → simulated signals (Δp history, k/k0,
injectivity, effluent curves). A weighted, normalised least-squares objective is
minimised by `least_squares` (local, gives Jacobian → covariance/CIs) or
`differential_evolution` (global). Bayesian Metropolis-Hastings provides
posterior uncertainty. Identifiability is reported via parameter relative-std,
correlation matrix, and JᵀJ condition number. Global sensitivity uses Sobol
total-effect indices. Full detail in `docs_phase6_history_matching.md`.

## Experimental data integration
Real core-flood CSVs are described by a JSON config (column names, units, flood
metadata) and ingested through a robust pipeline: delimiter/header detection,
unit & clock→pore-volume conversion, NaN/duplicate/outlier cleaning, smoothing/
resampling, and inverse-variance uncertainty weighting — all logged for
auditability. Validated end-to-end against a messy synthetic dataset. Detail in
`docs_phase7_data_integration.md`.

## Scientific integrity
No experimental data is fabricated. Calibration and data-integration machinery
is validated by synthetic-truth recovery (recover known parameters; honestly
flag non-identifiable ones). Real Niger Delta datasets integrate via config with
no code changes.

## Validation
82 automated tests: physics/benchmarks (incl. analytical Ogata–Banks to <1%,
exact Kozeny–Carman reproduction, retardation-factor benchmark) and framework
integration (config, imports, workflow, end-to-end). See `tests/`.
