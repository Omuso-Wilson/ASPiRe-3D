# ASPiRe-3D Streamlit GUI

Professional web interface for the ASPiRe-3D formation-damage simulator. Provides configuration, execution, results visualization, and file management without modifying any physics modules.

## Installation

```bash
# Install dependencies (already in requirements.txt)
pip install -r requirements.txt

# Or manually
pip install streamlit>=1.28
```

## Running the GUI

```bash
streamlit run gui/app.py
```

The GUI will open in your browser at `http://localhost:8501` (or the URL shown in the terminal).

## Features

### 🔧 Configuration Management
- **Bundled Examples**: Select from pre-configured baseline, sensitivity, and validation cases
- **Custom JSON**: Upload your own configuration files
- **Real-time Editing**: Adjust core properties, fluid parameters, injection schedule, physics toggles, and output settings directly in the sidebar
- **Live Validation**: Config errors are caught immediately

### ▶️ Simulation Control
- **Single Button Execution**: Run simulation with one click
- **Real-time Logging**: Watch progress in the execution log panel
- **State Management**: Simulation state persists across interface interactions

### 📊 Results Visualization
- **Key Metrics**: Final injectivity ratio, minimum permeability, final porosity, total steps
- **Calibration Metrics** (if enabled): R², RMSE, number of forward runs
- **Generated Figures**: Automatic display of pressure, permeability, injectivity, breakthrough curves
- **Multi-panel Layout**: Results organized for easy interpretation

### 📤 Data Management
- **CSV Upload**: Upload experimental core-flood data (pressure, concentration, permeability)
- **Column Mapping**: Manually map CSV columns to signal types
- **Preview & Inspection**: View uploaded data before processing
- **Future Integration**: Data upload ready for real calibration workflows

### 📥 Download & Export
- **Markdown Report**: Download generated validation report
- **Execution Log**: Download the full execution log
- **Figures**: Save generated figures via browser (right-click → Save Image)

### 📝 Execution Monitoring
- **Color-coded Log**: INFO, WARNING, ERROR, SUCCESS messages with icons
- **Scrollable Display**: View the complete execution history
- **Clear Button**: Reset the log at any time

## Architecture

```
gui/
├── app.py                    ← Main entry point (streamlit run gui/app.py)
├── session.py                ← Session state management
└── components/
    ├── sidebar.py            ← Configuration selector & parameter editing
    ├── simulator.py          ← Simulation execution & results display
    ├── logging.py            ← Real-time logging panel
    └── data_upload.py        ← Experimental data ingestion
```

## How It Works

1. **Configuration Loading**: The sidebar loads either a bundled example or custom JSON config
2. **Parameter Editing**: Users adjust any setting in the sidebar (no code changes)
3. **Simulation Execution**: Clicking "Run Simulation" invokes the existing `main/workflow.py` pipeline
4. **Results Capture**: The GUI extracts metrics, figures, and reports from the workflow output
5. **Display**: Results render in real-time with metrics cards and figure galleries
6. **Export**: Users can download reports and logs directly

## Integration with Core Physics

The GUI **does not modify** any validated physics modules:

- All solvers live in `physics/` and are unchanged
- All calibration code lives in `calibration/` and is unchanged
- The GUI only **orchestrates** execution via the existing `main/workflow.py`
- Config validation happens in `main/config_loader.py` (unchanged)
- Results extraction is done transparently by the GUI components

## Configuration File Format

Configs are JSON files with sections for:

```json
{
  "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.20, "permeability_mD": 100},
  "fluid": {"salinity_ppm": 10000, "temperature_C": 25, "pH": 9.5},
  "asp": {"surfactant": 1.0, "polymer": 1.0, "alkali": 1.0},
  "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
  "numerics": {"courant": 1.0, "grid": [30, 12, 12]},
  "physics": {"enable_surfactant_adsorption": true, "enable_damage": true},
  "parameters": {"rrf_max": 4.0},
  "calibration": {"enabled": false},
  "sensitivity": {"enabled": false},
  "output": {"save_figures": true, "save_report": true}
}
```

All optional parameters have defaults in the config loader.

## Troubleshooting

**Port already in use:**
```bash
streamlit run gui/app.py --server.port 8502
```

**Memory issues with large runs:**
- Reduce grid size (numerics.grid)
- Reduce pore_volumes
- Close other applications

**Config validation errors:**
- Check the error message in the sidebar
- Ensure required keys (core.*, injection.*) are present
- Verify ranges (porosity 0–0.8, permeability > 0, etc.)

## For Developers

The GUI is modular and extensible:

- **New parameters**: Add to `render_*_parameters()` functions in `sidebar.py`
- **New result displays**: Add to `render_results_panel()` in `simulator.py`
- **New tabs**: Add to the `st.tabs()` list in `app.py`
- **Custom metrics**: Extract them in `get_metrics_from_result()` in `session.py`

All without touching physics modules or the core workflow.

## Limitations & Future Work

**Currently not implemented but easy to add:**
- Real-time convergence plotting (update as simulation runs)
- Parameter sensitivity visualization (interactive tornado charts)
- Multi-run batch processing (sweep parameter space)
- Real CSV calibration workflow (wire up data_upload.py)
- 3D visualization of fields and pressure distribution

These can be added as additional components without changing the physics layer.
