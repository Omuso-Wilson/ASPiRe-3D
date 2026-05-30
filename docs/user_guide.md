# ASPiRe-3D — User Guide

ASPiRe-3D is a config-driven 3D porous-media reactive-transport simulator for
predicting formation damage during natural ASP (Alkaline–Surfactant–Polymer)
flooding in sandstone cores, with built-in history matching, sensitivity, and
uncertainty quantification.

---

## 1. Installation

```bash
# Python 3.10+ recommended
pip install -r requirements.txt
```

Dependencies are minimal: NumPy, SciPy, Matplotlib.

---

## 2. Running a simulation (single entry point)

Everything runs through one entry point, driven by a JSON config:

```bash
# list bundled example configs
python -m main.run_aspire3d --list

# run a case
python -m main.run_aspire3d --config experiments/configs/baseline_case.json

# quieter console (still logs to file)
python -m main.run_aspire3d --config experiments/configs/validation_case.json --quiet
```

Outputs (figures, Markdown report, log) are written to the `output.outdir`
declared in the config (default `experiments/outputs/`).

---

## 3. The three example cases

- **baseline_case.json** — a single forward ASP flood (adsorption + polymer
  retention + Kozeny–Carman damage); produces injectivity-decline and
  permeability/porosity figures and a run report.
- **sensitivity_case.json** — enables a Sobol global sensitivity analysis over
  the listed parameters, ranking their effect on final injectivity.
- **validation_case.json** — enables history matching in synthetic-truth mode
  (generates observations from known parameters, then recovers them), with
  calibration statistics, identifiability, and sensitivity.

---

## 4. Writing your own config

Copy an example and edit. Sections (only `core` and `injection` are strictly
required; everything else has sensible defaults):

```json
{
  "core":     { "length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2,
                "permeability_mD": 100, "clay_fraction": 0.05 },
  "fluid":    { "salinity_ppm": 10000, "temperature_C": 25, "pH": 9.5 },
  "asp":      { "surfactant": 1.0, "polymer": 1.0, "alkali": 1.0,
                "salinity": 0.3, "initial_salinity": 1.0 },
  "injection":{ "flow_rate_ccmin": 2.0, "pore_volumes": 3.0 },
  "numerics": { "courant": 1.0, "dispersivity_m": 5e-4,
                "reflow_tolerance": 0.03, "grid": [30, 12, 12] },
  "physics":  { "enable_surfactant_adsorption": true,
                "enable_polymer_retention": true,
                "enable_precipitation": false, "enable_fines": false,
                "enable_damage": true, "perm_model": "kozeny_carman",
                "isotherm": "langmuir" },
  "parameters": { "q_max": 1.5e-4, "K_L": 4.0, "rrf_max": 4.0,
                  "perm_exponent": 3.0, "k_ret": 2e-2 },
  "calibration": { "enabled": false },
  "sensitivity": { "enabled": false },
  "uncertainty": { "enabled": false },
  "output": { "outdir": "experiments/outputs", "prefix": "myrun",
              "save_figures": true, "save_report": true }
}
```

The loader validates ranges, units, and enums, and prints a clear error if
something is wrong (e.g. porosity outside 0–0.8, an unknown `perm_model`, or a
calibration block with no parameters).

---

## 5. Calibrating against laboratory data

Two modes:

1. **Synthetic-truth** (validation of the machinery): add a
   `calibration.synthetic_truth` block of known parameters; the workflow
   generates observations from the model and recovers them.
2. **Real core-flood data**: provide a `data` block describing your CSV files
   (see `data/templates/` and `docs/methodology.md` §experimental integration).
   Place CSVs under `experiments/raw_data/`. No code changes are needed.

Calibration reports include RMSE, R², 95% confidence intervals, identifiability
flags, and a JᵀJ condition number.

---

## 6. Project layout

```
main/          single entry point, workflow controller, config loader
physics/       validated FVM flow/transport/reaction/damage solvers
calibration/   history matching, sensitivity, uncertainty, experiment factory
data/          CSV ingestion, cleaning, schema/templates
visualization/ plotting (fields, breakthrough, damage, calibration, sensitivity)
experiments/   configs, raw_data, processed_data, outputs
reports/       exported markdown + figures
tests/         unit_tests (framework) + validation_tests (physics/benchmarks)
docs/          methodology, equations, assumptions, user_guide
```

---

## 7. Running the tests

```bash
# framework integration tests
python tests/unit_tests/test_integration.py

# physics & calibration validation + benchmarks
python tests/validation_tests/test_phase1.py
python tests/validation_tests/test_benchmarks.py
# ... (test_phase1 .. test_phase7, test_benchmarks)
```

All 82 tests should pass (72 physics/calibration + 10 framework integration).

---

## 8. Reproducibility

Every run is fully specified by its JSON config; optimizer, Sobol, and MCMC
seeds are deterministic; the forward model is deterministic given parameters;
and every run writes a log plus a Markdown report capturing configuration and
results. Re-running the same config reproduces the same outputs.
