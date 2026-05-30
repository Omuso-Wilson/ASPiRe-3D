"""
ASPiRe-3D GUI : gui/components/sidebar.py
===============================================================================
Sidebar configuration controls for case selection and parameter adjustment.

Provides dropdowns for bundled configs, CSV upload, and real-time parameter
editing without modifying the core Config object until simulation is run.
===============================================================================
"""

import streamlit as st
import json
from gui.session import get_bundled_config, add_log


def render_config_selector():
    """Render the configuration selector in the sidebar."""
    st.sidebar.header("⚙️ Configuration")
    
    config_source = st.sidebar.radio(
        "Config source",
        ["Bundled example", "Upload JSON"],
        key="config_source_radio"
    )
    
    config = None
    
    if config_source == "Bundled example":
        example = st.sidebar.selectbox(
            "Example case",
            ["baseline_case", "sensitivity_case", "validation_case"],
            help="Bundled configurations for quick testing or validation studies"
        )
        config = get_bundled_config(example)
        if config:
            st.sidebar.success(f"Loaded: {example}.json")
            add_log(f"Loaded bundled config: {example}.json")
        else:
            st.sidebar.error(f"Could not load {example}.json")
    
    else:  # Upload JSON
        uploaded_file = st.sidebar.file_uploader(
            "Upload JSON config",
            type=["json"],
            help="Custom ASPiRe-3D configuration file"
        )
        if uploaded_file is not None:
            try:
                config = json.load(uploaded_file)
                st.sidebar.success(f"Loaded: {uploaded_file.name}")
                add_log(f"Uploaded config: {uploaded_file.name}")
            except json.JSONDecodeError:
                st.sidebar.error("Invalid JSON file")
                config = None
    
    return config


def render_core_parameters(config):
    """Render editable core parameters in the sidebar."""
    if config is None:
        return config
    
    st.sidebar.subheader("Core Properties")
    
    with st.sidebar.expander("Core geometry & properties", expanded=True):
        core = config.get("core", {})
        core["length_cm"] = st.number_input(
            "Length (cm)", value=core.get("length_cm", 10.0), min_value=0.1, max_value=100.0, step=0.1
        )
        core["diameter_cm"] = st.number_input(
            "Diameter (cm)", value=core.get("diameter_cm", 3.8), min_value=0.1, max_value=30.0, step=0.1
        )
        core["porosity"] = st.slider(
            "Porosity", value=core.get("porosity", 0.20), min_value=0.001, max_value=0.8, step=0.01
        )
        core["permeability_mD"] = st.number_input(
            "Permeability (mD)", value=core.get("permeability_mD", 100.0), min_value=1e-3, max_value=1e6, step=1.0
        )
        config["core"] = core
    
    return config


def render_fluid_parameters(config):
    """Render editable fluid parameters in the sidebar."""
    if config is None:
        return config
    
    with st.sidebar.expander("Fluid properties"):
        fluid = config.get("fluid", {})
        fluid["salinity_ppm"] = st.number_input(
            "Salinity (ppm)", value=fluid.get("salinity_ppm", 10000.0), min_value=0.0, max_value=300000.0, step=100.0
        )
        fluid["temperature_C"] = st.slider(
            "Temperature (°C)", value=fluid.get("temperature_C", 25.0), min_value=0.0, max_value=200.0, step=1.0
        )
        fluid["pH"] = st.slider(
            "pH", value=fluid.get("pH", 9.5), min_value=0.0, max_value=14.0, step=0.1
        )
        config["fluid"] = fluid
    
    return config


def render_injection_parameters(config):
    """Render editable injection parameters in the sidebar."""
    if config is None:
        return config
    
    with st.sidebar.expander("Injection schedule", expanded=True):
        inj = config.get("injection", {})
        inj["flow_rate_ccmin"] = st.number_input(
            "Flow rate (cc/min)", value=inj.get("flow_rate_ccmin", 2.0), min_value=1e-3, max_value=100.0, step=0.1
        )
        inj["pore_volumes"] = st.number_input(
            "Pore volumes", value=inj.get("pore_volumes", 3.0), min_value=1e-3, max_value=1000.0, step=0.1
        )
        config["injection"] = inj
    
    return config


def render_physics_toggles(config):
    """Render physics model toggles in the sidebar."""
    if config is None:
        return config
    
    with st.sidebar.expander("Physics models"):
        phys = config.get("physics", {})
        phys["enable_surfactant_adsorption"] = st.checkbox(
            "Surfactant adsorption", value=phys.get("enable_surfactant_adsorption", True)
        )
        phys["enable_polymer_retention"] = st.checkbox(
            "Polymer retention + RRF", value=phys.get("enable_polymer_retention", True)
        )
        phys["enable_precipitation"] = st.checkbox(
            "Precipitation/dissolution", value=phys.get("enable_precipitation", False)
        )
        phys["enable_fines"] = st.checkbox(
            "Fines migration", value=phys.get("enable_fines", False)
        )
        phys["enable_damage"] = st.checkbox(
            "Formation damage", value=phys.get("enable_damage", True)
        )
        phys["perm_model"] = st.selectbox(
            "Permeability model",
            ["kozeny_carman", "power_law", "exponential"],
            index=["kozeny_carman", "power_law", "exponential"].index(phys.get("perm_model", "kozeny_carman"))
        )
        config["physics"] = phys
    
    return config


def render_output_controls(config):
    """Render output/visualization controls in the sidebar."""
    if config is None:
        return config
    
    with st.sidebar.expander("Output controls"):
        output = config.get("output", {})
        output["save_figures"] = st.checkbox(
            "Save figures", value=output.get("save_figures", True)
        )
        output["save_report"] = st.checkbox(
            "Save report", value=output.get("save_report", True)
        )
        output["prefix"] = st.text_input(
            "Output prefix", value=output.get("prefix", "run"), help="Prefix for output filenames"
        )
        config["output"] = output
    
    return config


def render_sidebar(config):
    """Main sidebar rendering function."""
    st.set_page_config(
        page_title="ASPiRe-3D",
        page_icon="⚗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Load config
    config = render_config_selector()
    
    if config is not None:
        # Render parameter panels
        config = render_core_parameters(config)
        config = render_fluid_parameters(config)
        config = render_injection_parameters(config)
        config = render_physics_toggles(config)
        config = render_output_controls(config)
        
        st.sidebar.divider()
        st.sidebar.info("⚠️ Simulation parameters locked during execution", icon="ℹ️")
    
    return config
