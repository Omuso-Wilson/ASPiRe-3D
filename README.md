# ASPiRe-3D

**Alkaline–Surfactant–Polymer Reactive Emulator in 3D** — a config-driven,
research-grade simulator for predicting formation damage during natural ASP
flooding in sandstone cores, with built-in history matching, sensitivity, and
uncertainty quantification.

Developed for doctoral research on formation damage during natural ASP flooding
in Niger Delta sandstone reservoirs.

---

## Quick start

```bash
pip install -r requirements.txt

# list bundled example configurations
python -m main.run_aspire3d --list

# run the baseline ASP flood
python -m main.run_aspire3d --config experiments/configs/baseline_case.json

# run a calibration + sensitivity validation case
python -m main.run_aspire3d --config experiments/configs/validation_case.json
```

Outputs (figures, Markdown report, log) land in `experiments/outputs/`.
Everything is driven by JSON configs — no hardcoded experimental values.

---

## What it does

- **Flow**: single-phase Darcy, implicit sparse pressure solve, conservative FVM.
- **Transport**: multi-species advection–dispersion (implicit, M-matrix, monotone).
- **Reactions**: precipitation/dissolution, fines migration, surfactant
  adsorption (Langmuir/Freundlich), polymer retention with Residual Resistance
  Factor.
- **Damage**: porosity→permeability (Kozeny–Carman / power-law / exponential),
  injectivity decline, pressure buildup.
- **Calibration**: optimizer-driven history matching (least-squares + global
  differential evolution), Bayesian uncertainty, Sobol sensitivity,
  identifiability diagnostics.
- **Data**: robust real core-flood CSV ingestion, cleaning, and validation with
  full provenance.

No new physics is introduced by the framework layer; it organizes and drives the
previously validated solvers.

---

## Architecture

```
ASPiRe-3D/
├── main/                  single entry point + workflow controller + config
│   ├── run_aspire3d.py        the ONLY top-level execution file
│   ├── workflow.py            ordered, logged, exception-handled pipeline
│   └── config_loader.py       JSON config: validation, units, defaults
├── physics/               validated FVM solvers (flow, transport, reaction, damage)
│   ├── darcy_solver.py        pressure + velocity (Darcy)
│   ├── transport_solver.py    advection–dispersion operators
│   ├── reactive_transport.py  multi-species backbone
│   ├── fines_migration.py     fines + precipitation kinetics
│   ├── permeability_update.py porosity→permeability damage
│   └── (mesh, geometry, properties, boundary_conditions, species,
│        adsorption, coupled_simulator, ...)
├── calibration/           history matching, sensitivity, uncertainty, optimization
├── data/                  ingestion, preprocessing, schemas, templates/
├── visualization/         plots, validation_plots, sensitivity_plots
├── experiments/           configs/, raw_data/, processed_data/, outputs/
├── reports/               markdown/, exported_figures/
├── tests/                 unit_tests/ (framework) + validation_tests/ (physics)
├── docs/                  methodology, equations, assumptions, user_guide
├── utils/                 shared constants + helpers
├── requirements.txt
├── README.md
└── LICENSE
```

See `docs/architecture_summary.md` for the full software-architecture report and
`docs/user_guide.md` for usage.

---

## Configuration

A config has sections for `core`, `fluid`, `asp`, `injection`, `numerics`,
`physics`, `parameters`, `calibration`, `sensitivity`, `uncertainty`, and
`output`. Only `core` and `injection` are strictly required; everything else has
validated defaults. Example:

```json
{
  "core":     { "length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100 },
  "fluid":    { "salinity_ppm": 10000, "temperature_C": 25, "pH": 9.5 },
  "injection":{ "flow_rate_ccmin": 2.0, "pore_volumes": 3.0 },
  "calibration": { "enabled": false }
}
```

The loader validates ranges, units, and enums, with clear error messages.

---

## Testing

```bash
python tests/unit_tests/test_integration.py        # framework (10 tests)
python tests/validation_tests/test_phase1.py       # flow core (11)
python tests/validation_tests/test_benchmarks.py   # analytical benchmarks (6)
# ... test_phase1 .. test_phase7, test_benchmarks
```

**82/82 tests pass** (72 physics/calibration + 10 framework integration).

---

## Documentation

- `docs/user_guide.md` — installation, running, writing configs, calibration.
- `docs/methodology.md` — numerical method and calibration methodology.
- `docs/equations.md` — governing equations and discretization.
- `docs/assumptions.md` — assumptions, limitations, and out-of-scope items.
- `docs/architecture_summary.md` — software-architecture summary report.

---

## Status & scope

Validated physics and calibration through experimental-data integration, now
organized as a reproducible, modular research-software framework. Out of scope
by design at this stage (architecture is extensible toward them): geochemical
equilibrium/speciation, ion exchange, multiphase flow, thermal coupling, ML
surrogates, and any GUI.

---

## 🖥️ Streamlit Web GUI

A professional web interface for ASPiRe-3D is included. No command-line expertise required.

### Launch the GUI

```bash
streamlit run gui/app.py
```

Or use the launch script:
```bash
./examples/run_gui.sh
```

Opens at `http://localhost:8501`.

### GUI Features

- **Configuration Management**: Load bundled cases or upload custom JSON configs
- **Real-time Parameter Editing**: Adjust core, fluid, injection, and physics settings directly
- **One-Click Execution**: Run simulation with a single button click
- **Real-time Logging**: Watch progress in the execution log panel
- **Results Visualization**: Automatic display of metrics and generated figures
- **File Downloads**: Export Markdown reports and execution logs
- **Data Upload** (prepared): Upload experimental CSV files for future calibration

### Quick Start with GUI

1. Open the GUI: `streamlit run gui/app.py`
2. Select **Bundled example** → **baseline_case** in the sidebar
3. Click **▶️ Run Simulation**
4. View results, metrics, and figures in the main panel
5. Download the report if desired

See `gui/QUICKSTART.md` for detailed instructions.

---
