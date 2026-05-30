"""
ASPiRe-3D GUI : gui/components/simulator.py
===============================================================================
Simulation execution and results display.

Wraps the existing run_aspire3d.py workflow without modifying physics modules.
Handles logging, progress indication, and results extraction.
===============================================================================
"""

import streamlit as st
import sys
import io
import json
import os
from pathlib import Path
from main.config_loader import load_config_dict, ConfigError
from main.workflow import Workflow
from gui.session import (
    add_log, save_temp_file, get_metrics_from_result,
    reset_simulation_state, get_temp_dir
)


def run_simulation(config):
    """Execute the ASPiRe-3D simulation with the given config."""
    if config is None:
        st.error("Configuration is required to run simulation")
        return None
    
    # Reset state
    reset_simulation_state()
    
    # Create a placeholder for progress
    progress_container = st.empty()
    log_container = st.empty()
    
    try:
        # Validate config
        progress_container.info("⏳ Validating configuration...")
        add_log("Validating configuration...")
        try:
            cfg = load_config_dict(config)
        except ConfigError as e:
            st.error(f"Configuration error: {e}")
            add_log(f"Config validation failed: {e}", level="ERROR")
            st.session_state.error_message = str(e)
            return None
        
        # Set output directory
        output_dir = get_temp_dir()
        cfg.raw["output"]["outdir"] = output_dir
        
        # Run workflow
        progress_container.info("▶️  Running simulation...")
        add_log("Starting simulation...")
        
        # Capture logging from workflow
        workflow = Workflow(cfg, verbose=False)
        results = workflow.run()
        
        # Extract results
        progress_container.success("✅ Simulation completed!")
        add_log("Simulation completed successfully", level="SUCCESS")
        
        # Extract metrics
        metrics = get_metrics_from_result(results)
        st.session_state.metrics = metrics
        st.session_state.simulation_result = results
        st.session_state.simulation_complete = True
        
        # Load generated figures
        load_figures_from_results(results, output_dir)
        
        return results
    
    except Exception as e:
        progress_container.error(f"Simulation failed: {str(e)}")
        add_log(f"Simulation error: {str(e)}", level="ERROR")
        st.session_state.error_message = str(e)
        return None


def load_figures_from_results(results, output_dir):
    """Load and cache matplotlib figures from the results."""
    figures_dir = Path(output_dir)
    for fig_file in figures_dir.glob("*.png"):
        try:
            with open(fig_file, "rb") as f:
                st.session_state.figures[fig_file.stem] = f.read()
        except Exception as e:
            add_log(f"Could not load figure {fig_file.name}: {e}", level="WARNING")


def render_simulation_button():
    """Render the simulation run button."""
    if st.session_state.get("config") is None:
        st.button("▶️ Run Simulation", disabled=True, help="Load a configuration first")
        return False
    
    if st.session_state.get("simulation_complete"):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Run Again"):
                return True
        with col2:
            if st.button("🔄 Reset Results"):
                reset_simulation_state()
                st.rerun()
        return False
    else:
        if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
            return True
    
    return False


def render_results_panel():
    """Render the results display panel."""
    if not st.session_state.get("simulation_complete"):
        st.info("👈 Configure and run a simulation to see results", icon="ℹ️")
        return
    
    results = st.session_state.get("simulation_result")
    metrics = st.session_state.get("metrics", {})
    
    # Key metrics
    st.subheader("📊 Simulation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        val = metrics.get("final_injectivity")
        st.metric("Final Injectivity (I/I₀)", f"{val:.3f}" if val else "—")
    
    with col2:
        val = metrics.get("min_permeability_ratio")
        st.metric("Min Permeability (k/k₀)", f"{val:.3f}" if val else "—")
    
    with col3:
        val = metrics.get("final_porosity")
        st.metric("Final Porosity", f"{val:.4f}" if val else "—")
    
    with col4:
        val = metrics.get("total_steps")
        st.metric("Total Steps", f"{val}" if val else "—")
    
    # Calibration metrics if present
    if metrics.get("calibration_r2") is not None:
        st.subheader("🎯 Calibration Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² (total)", f"{metrics['calibration_r2']:.4f}")
        
        with col2:
            st.metric("RMSE", f"{metrics['calibration_rmse']:.3e}")
        
        with col3:
            st.metric("Forward runs", f"{metrics['calibration_runs']}")
    
    # Figures
    if st.session_state.get("figures"):
        st.subheader("📈 Generated Figures")
        figures = st.session_state.get("figures", {})
        
        cols = st.columns(2)
        for i, (name, image_data) in enumerate(figures.items()):
            with cols[i % 2]:
                st.image(image_data, use_column_width=True, caption=name)
    
    # Download section
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)
    
    with col1:
        # Download report
        if results and results.get("report"):
            report_path = results["report"]
            if os.path.exists(report_path):
                with open(report_path, "r") as f:
                    report_text = f.read()
                st.download_button(
                    "📄 Download Report",
                    data=report_text,
                    file_name="aspire3d_report.md",
                    mime="text/markdown"
                )
    
    with col2:
        # Download log
        if results and results.get("logfile"):
            log_path = results["logfile"]
            if os.path.exists(log_path):
                with open(log_path, "r") as f:
                    log_text = f.read()
                st.download_button(
                    "📋 Download Log",
                    data=log_text,
                    file_name="aspire3d_log.txt",
                    mime="text/plain"
                )
    
    # Output directory info
    if st.session_state.get("output_dir"):
        st.info(f"📁 Results saved to: {st.session_state.output_dir}", icon="ℹ️")
