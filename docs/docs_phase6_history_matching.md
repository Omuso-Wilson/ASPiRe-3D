# ASPiRe-3D — Phase 6 Documentation: History Matching & Calibration

**Optimizer-Driven Calibration, Uncertainty Quantification, and Reproducible
Validation Workflow**

Publication-grade documentation of the history-matching methodology, objective
formulation, optimization strategy, uncertainty quantification, assumptions,
and reproducibility for ASPiRe-3D.

---

## 1. Scientific-integrity statement

ASPiRe-3D ships **no experimental data**, and none is fabricated. Inventing
"laboratory" measurements and then matching them would be scientific
misconduct. The calibration engine therefore operates on **user-supplied
observed data**, and its correctness is established by **synthetic-truth
recovery**: data is generated from the simulator at a known parameter set, and
the engine is shown to recover the identifiable parameters and to *correctly
flag* the non-identifiable ones. This is the standard, defensible way to qualify
an inversion workflow before applying it to real core-flood data. The path to
real-data use is a single configuration switch (`SYNTHETIC=False`) plus a CSV
loader — no code changes downstream.

---

## 2. History-matching methodology

### 2.1 Forward model

A `ForwardModel` maps a parameter dictionary to simulated signals by building
and running a fresh `CoupledSimulator` (conservative FVM transport, implicit
M-matrix solve, operator splitting with stiffness-controlled sub-stepping) and
extracting the comparable signals: differential-pressure history, effluent
concentration curves, permeability ratio, and injectivity. The mapping from
calibration-parameter names to physical model parameters is centralised in
`core/experiment.py` for auditability.

### 2.2 Objective-function formulation

For observed signals `o_s(t)` and simulated `ŷ_s(t)`, the weighted, normalised
residual vector is

```
r = concat_s [ sqrt(w_s)/scale_s * ( interp(ŷ_s, t_s) − o_s ) ]
scale_s = RMS(o_s)        (normalisation so disparate signals weigh comparably)
```

`least_squares` minimises `‖r‖²` exploiting the residual structure;
`differential_evolution` minimises the scalar `‖r‖²`. Normalisation ensures a
kPa-scale pressure signal and an O(1) concentration signal contribute on equal
footing.

### 2.3 Optimization strategy

- **`least_squares`** (Trust Region Reflective): local, gradient-based, fast,
  and returns the Jacobian → linearized covariance → confidence intervals,
  correlation matrix, and condition number. Preferred when a reasonable initial
  guess exists.
- **`differential_evolution`**: global, gradient-free, robust to multimodal
  misfit surfaces and poor initial guesses (Sobol-initialised population),
  optionally polished by a local solve. Preferred for difficult or poorly-known
  parameter spaces.
- **Bayesian (Metropolis-Hastings)**: posterior parameter distributions and
  credible intervals plus posterior-predictive uncertainty envelopes — a
  fuller uncertainty statement than the linearized covariance.

### 2.4 Calibratable parameters

Surfactant: `q_max`, `K_L` (or `K_F`, `n_F`), `k_ads`, salinity/pH
coefficients. Polymer: `sigma_max`, `k_ret`, `rrf_max`, `perm_gamma`, shear
parameters. Precipitation: `k_precip`, `k_dissolve`. Fines: `k_detach`,
`k_deposit`, `critical_velocity`. Damage: `perm_exponent` / law parameters.
Parameters carry physical bounds and optional log-scaling (for quantities
spanning orders of magnitude).

---

## 3. Uncertainty quantification & identifiability

- **Confidence intervals**: `cov = σ²(JᵀJ)⁻¹`, `σ² = SSR/(m−p)`; 95% CI =
  estimate ± 1.96·√diag(cov).
- **Identifiability**: per-parameter relative standard deviation (CV);
  CV < 0.5 well-identified, < 2 weak, ≥ 2 poor. The `JᵀJ` condition number
  flags an ill-conditioned (correlated) system globally.
- **Correlation matrix**: exposes parameter trade-offs (e.g. Langmuir `q_max`
  and `K_L` are individually non-identifiable from a near-saturated effluent —
  only their low-concentration product is constrained — which the engine
  reports as a high correlation and large CV rather than a false-confident
  estimate).
- **Global sensitivity (Sobol)**: variance-based first-order (`S1`) and
  total-effect (`ST`) indices via the Saltelli scheme, ranking parameter
  importance over the entire bounded space; the rigorous complement to local
  one-at-a-time sensitivity.
- **Bayesian posterior**: credible intervals and posterior-predictive bands.

---

## 4. Engineering outputs

RMSE and R² per signal and aggregate; parameter estimates with 95% CIs;
identifiability flags; `JᵀJ` condition number; correlation matrix; Sobol
ranking; posterior credible intervals; and uncertainty envelopes. All are
emitted into an auto-generated Markdown `calibration_report.md` and a set of
thesis-ready figures (observed-vs-simulated, residuals, convergence, tornado,
uncertainty band).

---

## 5. Validation results (synthetic-truth)

| Check | Result |
|-------|--------|
| Zero residual at the true parameters | SSR = 0 (machinery consistent) |
| Identifiable-parameter recovery (RRF) | 0.0% error, R² = 1.0000 |
| Three-parameter joint recovery (RRF, n, k_ret) | all 0.0% error, R² = 1.0000 |
| Confidence intervals / condition number | produced, finite, bracket estimate |
| Non-identifiability flagged (q_max, K_L) | CV ≫ 1 and correlation −0.96 |
| differential_evolution global recovery | 0.0% error from a poor guess |
| Convergence history | monotone decreasing |
| Sobol ranking | influential ≫ non-influential |
| Bayesian posterior brackets truth | yes |

**63/63 automated tests pass across all phases.**

---

## 6. Assumptions & limitations

- Synthetic-truth validation demonstrates the *machinery*; quantitative claims
  about real reservoirs require real core-flood data (the loader is ready).
- Linearized (Jacobian) confidence intervals assume locally-Gaussian residuals;
  the Bayesian path is provided where that assumption is doubtful.
- Identifiability is dataset-dependent: which parameters are constrained depends
  on which signals are measured and how informative they are — the diagnostics
  report this honestly per case.
- Sequential first-order coupling and first-order upwind numerical diffusion
  apply as documented in earlier phases (convergent, monotone, conservative).
- Measurement-noise models for real data (heteroscedastic, correlated) can be
  added through per-signal weights and the Bayesian `sigma`.

---

## 7. Reproducibility notes

- Fixed experiment configuration object (`ASPExperiment`); deterministic seeds
  for differential evolution, Sobol, and MCMC.
- Forward model is fully deterministic given parameters.
- `calibration_workflow.py` regenerates all figures and the report from scratch.
- Real-data use: implement `load_real_data()` with CSVs and set
  `SYNTHETIC=False`; everything else is identical.

---

## 8. Publishable novelty

- A complete, open, **reproducible history-matching workflow** for coupled ASP
  formation-damage reactive transport, integrating local, global, and Bayesian
  calibration with rigorous identifiability and global-sensitivity diagnostics.
- **Honest identifiability reporting**: the engine distinguishes constrained
  from unconstrained parameters and exposes parameter trade-offs, rather than
  returning confident point estimates — a standard rarely met at thesis scale.
- A demonstrated, validated path from **mechanistic ASP physics to quantitative
  core-flood history matching**, ready for Niger Delta sandstone datasets.
```
Phase 1–5 ✅  →  Phase 6 history matching & calibration ✅
Next: geochemical equilibrium → ion exchange → multiphase → thermal → ML surrogates
```
