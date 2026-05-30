# ASPiRe-3D GUI — Developer Guide

## Architecture Overview

The GUI is built as a modular Streamlit application that wraps the existing `main/workflow.py` orchestration layer. It does not modify any physics modules.

```
┌─────────────────────────────────────────────────────┐
│               Streamlit Web App (gui/)              │
│  ┌────────────────────────────────────────────────┐ │
│  │  gui/app.py  — Main entry point & layout       │ │
│  ├────────────────────────────────────────────────┤ │
│  │  gui/session.py  — Session state management    │ │
│  ├────────────────────────────────────────────────┤ │
│  │  gui/components/  — UI components              │ │
│  │    ├── sidebar.py      (config controls)       │ │
│  │    ├── simulator.py    (execution & display)   │ │
│  │    ├── logging.py      (progress monitor)      │ │
│  │    └── data_upload.py  (CSV ingestion)         │ │
│  └────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
                           │
                           ↓
        ┌──────────────────────────────────────┐
        │  main/workflow.py  (Orchestration)   │
        │  main/config_loader.py (Validation)  │
        └──────────────────────────────────────┘
                           │
                           ↓
        ┌──────────────────────────────────────┐
        │  physics/ calibration/ data/ etc.    │
        │  (Unchanged validated modules)       │
        └──────────────────────────────────────┘
```

## Module Reference

### `gui/app.py`

**Main entry point. Runs on `streamlit run gui/app.py`.**

Functions:
- **Page setup**: Configures Streamlit page (title, layout, custom CSS)
- **Session initialization**: Calls `init_session_state()`
- **Layout rendering**: Three-column layout (sidebar + main + logs)
- **Tab management**: Simulator, Data Upload, Information tabs

```python
# To add a new tab:
tab_name = st.tabs(["🚀 Simulator", "📤 Data", "ℹ️ Info", "🆕 New Tab"])
with tab_name[3]:  # Fourth tab
    render_new_tab()  # Your function here
```

### `gui/session.py`

**Session state and utility functions.**

Key functions:
- `init_session_state()` — Initialize streamlit session variables
- `add_log(message, level)` — Append to execution log
- `clear_logs()` — Clear the log
- `get_bundled_config(name)` — Load example config from JSON
- `save_temp_file(filename, content)` — Save to temp directory
- `get_metrics_from_result(result)` — Extract metrics from simulation result
- `reset_simulation_state()` — Clear results (keep config)

Session state keys:
```python
st.session_state.config              # Dict: the loaded/edited config
st.session_state.simulation_complete # Bool: has simulation run?
st.session_state.simulation_result   # Dict: workflow result
st.session_state.metrics             # Dict: extracted key metrics
st.session_state.figures             # Dict: base64 images
st.session_state.log_messages        # List: execution log entries
```

### `gui/components/sidebar.py`

**Configuration loading and parameter editing.**

Functions:
- `render_config_selector()` — Dropdown for bundled vs. uploaded configs
- `render_core_parameters(config)` — Editable core geometry/properties
- `render_fluid_parameters(config)` — Editable fluid settings
- `render_injection_parameters(config)` — Editable injection schedule
- `render_physics_toggles(config)` — Physics model checkboxes
- `render_output_controls(config)` — Output prefix, figure/report toggles
- `render_sidebar(config)` — Main sidebar function

Example: Adding a new parameter:
```python
def render_asp_composition(config):
    with st.sidebar.expander("ASP composition"):
        asp = config.get("asp", {})
        asp["surfactant"] = st.slider("Surfactant", 0.1, 5.0, step=0.1)
        config["asp"] = asp
    return config

# Call from render_sidebar():
config = render_asp_composition(config)
```

### `gui/components/simulator.py`

**Simulation execution and results display.**

Functions:
- `run_simulation(config)` — Execute the workflow (main entry point for simulation)
- `load_figures_from_results(results, output_dir)` — Cache PNG figures
- `render_simulation_button()` — Run/Run Again button with state management
- `render_results_panel()` — Display metrics, figures, download buttons

The key flow:
```
render_simulation_button()
    ↓ user clicks "Run"
    ↓
run_simulation(config)
    ↓
load_config_dict(config)  # Validate
    ↓
Workflow(cfg).run()       # Execute
    ↓
get_metrics_from_result() # Extract metrics
    ↓
load_figures_from_results() # Cache images
    ↓
render_results_panel()    # Display
```

### `gui/components/logging.py`

**Real-time execution log display.**

Functions:
- `render_logging_panel()` — Display color-coded log entries
- `render_error_panel()` — Show error messages if present

Log entry format (in session state):
```python
{
    "message": "Starting simulation...",
    "level": "INFO"  # or "SUCCESS", "WARNING", "ERROR"
}
```

### `gui/components/data_upload.py`

**CSV file upload and column mapping (prepared for future work).**

Functions:
- `render_data_upload_tab()` — File upload, preview, and column mapping
- `get_uploaded_data_config()` — Generate data block for config

Currently this is scaffolded; to enable real CSV calibration:
1. Create a `data_config` section in the JSON schema
2. Modify `run_simulation()` to pass uploaded data metadata to workflow
3. The `data/experimental_validation.py` module is ready to ingest the CSVs

---

## Adding a New Feature

### Example 1: Add a new physics toggle

In `gui/components/sidebar.py`:

```python
def render_new_physics_model(config):
    with st.sidebar.expander("New physics"):
        phys = config.get("physics", {})
        phys["enable_new_model"] = st.checkbox("New model", value=phys.get("enable_new_model", False))
        config["physics"] = phys
    return config
```

In `gui/app.py`, call it from `render_sidebar()`:
```python
config = render_new_physics_model(config)
```

### Example 2: Add a new results visualization

In `gui/components/simulator.py`:

```python
def render_3d_field_visualization():
    """Display 3D pressure/concentration field."""
    if st.session_state.get("simulation_result") is None:
        return
    
    result = st.session_state.simulation_result
    sim = result.get("simulation")
    if sim is None:
        return
    
    # Your visualization code here
    st.plotly_chart(create_3d_plot(sim))  # Example
```

In `gui/app.py`, add to the results panel:
```python
with tab_sim:
    # ... existing code ...
    render_3d_field_visualization()
```

### Example 3: Add a download option

In `gui/components/simulator.py`:

```python
def render_download_panel():
    st.subheader("📥 Downloads")
    
    results = st.session_state.simulation_result
    if results and results.get("figures"):
        # Create a ZIP of all figures
        import zipfile, io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as z:
            for fig_path in results["figures"]:
                z.write(fig_path, arcname=os.path.basename(fig_path))
        
        zip_buffer.seek(0)
        st.download_button(
            "📦 Download All Figures",
            data=zip_buffer.getvalue(),
            file_name="aspire3d_figures.zip",
            mime="application/zip"
        )
```

---

## Testing the GUI

### Unit tests (no Streamlit required)
```bash
python tests/unit_tests/test_gui_logic.py
```

Tests config modification, metrics extraction, bundled configs, parameter ranges.

### Manual testing (requires Streamlit)
```bash
streamlit run gui/app.py
```

1. Load each bundled case
2. Edit parameters and verify they're preserved through the run
3. Run simulation and check metrics display
4. Download report/log and verify content
5. Test error handling (e.g., invalid JSON upload)

---

## Deployment Considerations

### For local use (laptop/workstation):
```bash
streamlit run gui/app.py
```

### For remote server (e.g., campus server):
```bash
streamlit run gui/app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.baseUrlPath /aspire3d
```

Then access at `http://server-ip:8501/aspire3d`.

### For Docker containerization:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "gui/app.py"]
```

---

## Common Pitfalls

1. **Modifying physics in the GUI**: Don't. All physics is in `physics/` and is immutable.
2. **Session state not persisting**: Streamlit reruns the entire script on each interaction. Use `st.session_state` to persist data across reruns.
3. **Blocking the UI**: Long operations should use `st.spinner()` or progress indicators. The workflow runs synchronously; consider caching for faster reruns.
4. **File paths**: Use absolute paths or `Path(__file__).parent` to ensure paths work regardless of where the script is run from.

---

## Performance Optimization

- **Caching**: Streamlit's `@st.cache_data` decorator caches function results across reruns
- **Session state**: Pre-compute expensive results and store in `st.session_state`
- **Progressive disclosure**: Use `st.expander()` to hide details until needed
- **Lazy loading**: Don't load all figures if the user hasn't scrolled there yet

---

## Future Enhancements

1. **Real CSV calibration**: Wire up `gui/components/data_upload.py` to ingest and process CSVs
2. **Interactive plots**: Use Plotly for interactive injectivity/permeability curves
3. **Batch runs**: Upload multiple configs, run in sequence, compare results
4. **Real-time updates**: Stream simulation progress (currently shows after completion)
5. **3D visualization**: Use VTK or Plotly for 3D pressure/concentration fields
6. **Parameter sweep**: UI for range-based sensitivity studies
7. **Exporting configs**: Save edited configs back to JSON for reproducibility

All can be added without touching physics modules.
