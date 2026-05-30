# ASPiRe-3D Streamlit GUI — Quick Start

## Installation (30 seconds)

```bash
pip install -r requirements.txt
```

This installs Streamlit and other dependencies needed for the GUI.

## Running (10 seconds)

```bash
streamlit run gui/app.py
```

Or use the launch script:

```bash
./gui/run.sh
```

The GUI opens at `http://localhost:8501` in your browser.

## Basic Workflow

### 1. Load or Create a Configuration

**In the sidebar:**
- Select **"Bundled example"** to use a pre-configured case (baseline, sensitivity, validation)
- Or select **"Upload JSON"** to load your own config file

### 2. Adjust Parameters (Optional)

All sidebar sections are editable:
- **Core geometry & properties**: Length, diameter, porosity, permeability
- **Fluid properties**: Salinity, temperature, pH
- **Injection schedule**: Flow rate, pore volumes
- **Physics models**: Toggle surfactant adsorption, polymer, precipitation, fines, damage
- **Output controls**: Prefix for output files

Parameters are validated in real-time. Any errors appear in red.

### 3. Run the Simulation

Click the blue **"▶️ Run Simulation"** button.

- The **Execution Log** panel (right side) shows real-time progress
- A spinner indicates the simulation is running
- Look for "✅ Simulation completed!" when done

### 4. View Results

Once complete, the **Simulator** tab shows:

- **Key Metrics**: Injectivity ratio, permeability reduction, porosity, total steps
- **Calibration Metrics** (if enabled): R², RMSE, number of forward runs
- **Generated Figures**: Pressure, permeability, injectivity, breakthrough curves
- **Download Buttons**: Export the Markdown report and log file

### 5. Upload Experimental Data (Optional)

Switch to the **"📤 Data Upload"** tab to:
- Upload your core-flood CSV files
- Preview the data
- Map columns to signal types (pressure, concentration, permeability)

*(Data integration with calibration coming soon)*

---

## Example Cases

### Baseline Case
- Simple ASP flood with adsorption + damage
- Kozeny-Carman damage model
- No calibration
- **Run time**: ~4 seconds

### Sensitivity Case
- Same physics as baseline
- Enables **Sobol global sensitivity** analysis
- Generates a tornado chart showing parameter importance
- **Run time**: ~15 seconds

### Validation Case
- Enables **history matching** (least-squares calibration)
- Synthetic-truth mode: generates synthetic observations, then recovers the parameters
- Includes sensitivity analysis
- **Run time**: ~30 seconds

---

## Configuration File Format

Configs are JSON with these sections:

```json
{
  "core": {
    "length_cm": 10,
    "diameter_cm": 3.8,
    "porosity": 0.20,
    "permeability_mD": 100
  },
  "fluid": {
    "salinity_ppm": 10000,
    "temperature_C": 25,
    "pH": 9.5
  },
  "injection": {
    "flow_rate_ccmin": 2.0,
    "pore_volumes": 3.0
  },
  "physics": {
    "enable_surfactant_adsorption": true,
    "enable_polymer_retention": true,
    "enable_damage": true,
    "perm_model": "kozeny_carman"
  },
  "output": {
    "prefix": "my_run"
  }
}
```

**Required**: `core.{length_cm, diameter_cm, porosity, permeability_mD}`, `injection.{flow_rate_ccmin, pore_volumes}`

**Optional**: Everything else has defaults.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8501 in use | `streamlit run gui/app.py --server.port 8502` |
| Slow simulation | Reduce `numerics.grid` (default 30×12×12) or `pore_volumes` |
| Config error | Check sidebar error message; ensure required keys are present |
| Missing figures | Enable `output.save_figures: true` (default) |
| Can't upload files | Ensure browser allows file uploads; try a different file |

---

## What's Happening Behind the Scenes

1. **Configuration**: GUI loads/validates JSON config via `main/config_loader.py`
2. **Execution**: GUI invokes `main/workflow.py` (the standard CLI workflow)
3. **Physics**: Solvers run unchanged from `physics/` package (FVM, transport, reaction, damage)
4. **Results**: Workflow writes figures and reports to a temp directory
5. **Display**: GUI loads and displays figures, metrics, and logs

No physics modules are modified. The GUI is purely an orchestration layer.

---

## Tips for Users

- **Reproducibility**: Every run is fully specified by the JSON config. Save configs for cases you want to reproduce.
- **Parameter sweeps**: Copy a baseline config, tweak one or two parameters, run again. Each run is independent.
- **Download reports**: Markdown reports can be opened in any text editor or viewed on GitHub.
- **Batch runs**: Use the CLI (`python -m main.run_aspire3d --config <file.json>`) for automated sweeps.

---

## For Developers

The GUI is modular:
- `gui/app.py` — Main entry point
- `gui/session.py` — Session state management
- `gui/components/sidebar.py` — Configuration controls
- `gui/components/simulator.py` — Execution and results display
- `gui/components/logging.py` — Real-time log panel
- `gui/components/data_upload.py` — CSV ingestion (future work)

To add a new feature (e.g., 3D visualization, real-time plotting), add a component without touching physics modules.

---

## Next Steps

- Read `docs/user_guide.md` for detailed documentation
- Explore `docs/methodology.md` to understand the physics
- Try the **validation case** to see calibration in action
- Upload your own core-flood data (once CSV integration is complete)
