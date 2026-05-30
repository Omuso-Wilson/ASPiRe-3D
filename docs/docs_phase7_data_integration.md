# ASPiRe-3D — Phase 7 Documentation: Experimental Data Integration

**Plug-and-Play Real Core-Flood Data Ingestion, Cleaning, and Validation**

How to integrate real Niger Delta (or any) laboratory ASP core-flood data into
ASPiRe-3D for quantitative, defensible history matching — plus the methodology,
assumptions, and reproducibility notes.

---

## 1. What this phase adds

A complete pipeline from a **raw laboratory CSV** to a **calibrated,
uncertainty-quantified, fully-reported validation**, with minimal restructuring:

```
lab CSVs + JSON config
   → schema mapping (columns, units, metadata)       core/data_schema.py
   → robust ingest + clean (units, PV clock, outliers) core/data_ingestion.py
   → ObservedData with inverse-variance weights        core/calibration.py
   → forward model matched to the experiment geometry  core/experiment.py
   → history match (least_squares / diff. evolution)   core/history_matching.py
   → identifiability, Sobol sensitivity, uncertainty   core/uncertainty.py
   → automated validation report + figures             core/experimental_validation.py
```

Entry point: `experimental_validation_workflow.py`.

---

## 2. Integrating real data (three steps, no code changes)

1. **Place your CSVs** in a folder, e.g. `data/coreA/`. Each signal is a small
   table: a clock column and a value column (optionally a per-point
   uncertainty column). Column names are free.
2. **Describe your file** by copying `data_templates/example_config.json` and
   editing the `metadata` (core dimensions, porosity, flow rate, baseline
   permeability, injected concentrations) and one `signals` entry per measured
   curve (which column, what unit, which clock, normalisation, uncertainty).
3. **Run**
   `python experimental_validation_workflow.py --config data/coreA/config.json`.

Supported canonical signals: `dp_history`, `injectivity_history`,
`k_ratio_history`, `surfactant_effluent`, `polymer_effluent`,
`salinity_effluent`. Supported raw units — pressure: Pa/kPa/MPa/bar/psi/atm;
clock: s/min/hr/PV. Effluent is normalised to C/C₀ by a declared injected
reference.

---

## 3. Preprocessing & cleaning (all logged)

Each signal passes through, with every operation recorded in a provenance log:
delimiter/header auto-detection and comment skipping; unit and clock→PV
conversion; NaN/inf removal; sorting by clock; duplicate-clock averaging;
MAD/Hampel outlier removal (>4 robust-σ by default); optional smoothing and
uniform-PV resampling; and uncertainty attachment. The provenance log is
embedded in the validation report so the data treatment is fully auditable —
answering the inevitable "what did you do to the raw measurements?".

---

## 4. Experimental uncertainty handling

Per-point 1-σ comes from a declared uncertainty column, a constant, or (failing
both) an estimated noise floor. Signals are weighted by **inverse variance**
(maximum-likelihood weighting for Gaussian noise): a cleaner signal contributes
more to the fit than a noisy one. Weights feed the existing
weighted-normalised objective so disparate signals (kPa-scale Δp, O(1)
concentration) are comparable.

---

## 5. Matching targets

`dp_history` (pressure-drop history), `k_ratio_history` (permeability
reduction), `injectivity_history` (injectivity decline), and effluent
concentration curves (`surfactant_effluent`, `polymer_effluent`,
`salinity_effluent`) — any subset present in the data is matched jointly.

---

## 6. Outputs

RMSE and R² per signal and aggregate; calibrated parameters with 95% CIs,
identifiability flags, JᵀJ condition number, and correlation matrix; Sobol
global sensitivity; and an auto-generated Markdown **validation report** plus
figures (observed-vs-simulated, residuals, convergence, tornado).

**On R² for plateau signals:** RRF-dominated Δp and k/k₀ are near-constant; R²
is statistically unreliable for low-variance data (tiny SS_tot) and may be
near-zero or negative even for an excellent fit. The report **auto-flags** such
signals and directs interpretation to RMSE. This is a known, defensible
statistical point, not a fit failure.

---

## 7. Validation of the pipeline itself

Because no real data is bundled (and fabricating it would be misconduct), the
ingestion+cleaning+calibration chain is validated against a **messy synthetic
dataset** generated from known parameters: deliberately non-SI units (psi),
a minutes clock, shuffled non-monotonic rows, duplicate rows, injected
outliers, and Gaussian noise. The pipeline parses, converts, cleans, weights,
and matches it, recovering identifiable parameters within the noise tolerance
(e.g. RRF to ~0.5–3%) and **correctly flagging** non-identifiable ones (large
CI, POOR identifiability, ill-conditioned JᵀJ). This is the standard way to
qualify a data-integration and inversion workflow before applying it to real
laboratory measurements. **72/72 automated tests pass across all phases.**

---

## 8. Assumptions & limitations

- The forward model is configured to match the experiment's geometry, rate, and
  porosity; sub-core heterogeneity beyond the homogeneous baseline is not yet
  represented (a future extension).
- Identifiability is dataset- and design-dependent: which parameters are
  constrained depends on which signals were measured and how informative they
  are. The diagnostics report this honestly per case; calibrating
  non-identifiable parameters (e.g. the porosity-permeability exponent when RRF
  dominates) is intentionally discouraged.
- Linearized (Jacobian) confidence intervals assume locally-Gaussian residuals;
  the Bayesian path quantifies uncertainty where that assumption is doubtful.
- Cleaning thresholds (outlier σ, smoothing) are user-tunable and logged; their
  effect on the match should be reported as a robustness check.

---

## 9. Reproducibility

Fixed experiment metadata; deterministic optimizer/Sobol/MCMC seeds; a fully
deterministic forward model; provenance logging of all data transformations;
and a single-command workflow that regenerates the report and figures. Real-data
use is a configuration change only — the validated code path is identical to the
synthetic-validation path.

---

## 10. Publishable contribution

A reproducible, auditable **experimental-validation framework** that takes raw,
messy core-flood data to a calibrated, uncertainty-quantified ASP formation-
damage history match, with honest identifiability reporting and automated
documentation — directly enabling thesis-grade experimental validation against
Niger Delta sandstone datasets.

```
Phase 1–6 ✅  →  Phase 7 experimental data integration ✅
Next: geochemical equilibrium → ion exchange → multiphase → thermal → ML
```
