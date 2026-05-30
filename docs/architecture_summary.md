# ASPiRe-3D — Software Architecture Summary Report

_Transition from generated scripts to a structured scientific-software
framework. Phases 1–3 (organize, single entry point, config-driven execution)._

---

## 1. Objective and outcome

ASPiRe-3D's validated physics and calibration capabilities were reorganized into
a professional, reproducible research-software framework **without changing any
physics**. The result: one entry point, fully config-driven execution, clean
package separation, and a complete test/documentation suite. **82/82 tests pass**
(72 physics/calibration validation + 10 framework integration).

---

## 2. Package architecture

| Package | Responsibility | Key public modules |
|---|---|---|
| `main/` | execution control | `run_aspire3d.py` (sole entry point), `workflow.py`, `config_loader.py` |
| `physics/` | validated solvers | `darcy_solver`, `transport_solver`, `reactive_transport`, `fines_migration`, `permeability_update` (+ mesh, geometry, properties, boundary_conditions, species, adsorption, coupled_simulator) |
| `calibration/` | inversion & UQ | `history_matching`, `optimization`, `sensitivity`, `uncertainty`, `experiment`, `interpretation` |
| `data/` | experimental data | `ingestion`, `preprocessing`, `schemas`, `templates/` |
| `visualization/` | figures | `plots`, `validation_plots`, `sensitivity_plots` (+ damage/adsorption/calibration plots) |
| `experiments/` | run artifacts | `configs/`, `raw_data/`, `processed_data/`, `outputs/` |
| `reports/`, `docs/`, `tests/`, `utils/` | reports, docs, tests, shared helpers | |

The role-named modules requested by the architecture (`darcy_solver`,
`transport_solver`, etc.) are thin **facade modules** that re-export the
validated implementations under their original, verified names. This satisfies
the structure and provides clean public entry points **without rewriting any
validated numerical solver** — a deliberate risk-minimising decision.

---

## 3. Phase 1 — codebase organization

- 23 `core/` modules + `postprocessing/` + `utils/` were relocated into
  `physics/`, `calibration/`, `data/`, `visualization/`, `utils/`.
- All internal imports were refactored programmatically to the new packages.
- Facade modules added for the requested public names; duplicated/role names
  consolidated by re-export rather than copy.
- The legacy top-level scripts (`main.py`, `calibration_workflow.py`,
  `experimental_validation_workflow.py`) were retired; their capabilities are
  now reached through the single config-driven entry point and the `data`
  package functions.
- All prior validation suites migrated to `tests/validation_tests/` and pass
  unchanged (import paths updated).

## 4. Phase 2 — single entry point

- `main/run_aspire3d.py` is the **only** top-level execution file (CLI:
  `--config`, `--list`, `--quiet`).
- `main/workflow.py` is a `Workflow` controller executing the ordered pipeline:
  validate → load data → init physics → simulate → calibrate → sensitivity →
  plots → report → logs.
- Structured logging (file + concise console), robust exception handling (fatal
  stages abort; optional stages degrade gracefully), and deterministic ordering
  for reproducibility.

## 5. Phase 3 — config-driven execution

- `main/config_loader.py` parses JSON, fills defaults for optional parameters,
  validates required keys/ranges/enums and unit/grid consistency, converts to SI,
  and raises clear `ConfigError` messages.
- Schema covers core, fluid, ASP composition, injection, numerics, physics
  toggles, model parameters, calibration, sensitivity, uncertainty, and output.
- Three example configs provided and validated: `baseline_case.json`,
  `sensitivity_case.json`, `validation_case.json`.
- No hardcoded experimental values remain in the execution path.

---

## 6. Testing

| Suite | Location | Tests |
|---|---|---|
| Framework integration | `tests/unit_tests/test_integration.py` | 10 |
| Phase 1–7 physics/calibration | `tests/validation_tests/test_phase*.py` | 56 |
| Benchmarks | `tests/validation_tests/test_benchmarks.py` | 6 |
| **Total** | | **82** |

Integration tests cover module imports, config loading (defaults, ranges, enums,
bad JSON, calibration validation), the three example configs, and end-to-end
workflow runs (baseline and synthetic-truth calibration recovering RRF to 0.0%).

---

## 7. Reproducibility & extensibility

- Each run fully specified by its JSON config; deterministic seeds; deterministic
  forward model; every run writes a log and a Markdown report.
- The validated transport core is untouched; future physics (geochemical
  equilibrium, ion exchange, multiphase, thermal) attaches as new
  `ReactionModel`/property modules under `physics/`, and new optimizers/UQ under
  `calibration/`, with no change to the framework layer.

---

## 8. Constraints honoured

No new physics, no geochemical equilibrium/speciation, no GUI, no ML, no
rewriting of validated solvers, and no unnecessary complexity. The work was
strictly software architecture, integration, and execution stability.
