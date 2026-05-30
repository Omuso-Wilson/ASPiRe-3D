"""
ASPiRe-3D GUI : gui/session.py
===============================================================================
Session state management and utilities for the Streamlit GUI.

Manages simulation state, results caching, and file handling across the app
lifecycle without modifying the core physics modules.
===============================================================================
"""

import os
import tempfile
import json
import streamlit as st
from pathlib import Path


def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        "config": None,
        "config_source": "bundled",  # 'bundled' or 'uploaded'
        "simulation_complete": False,
        "calibration_complete": False,
        "simulation_result": None,
        "calibration_result": None,
        "sensitivity_result": None,
        "log_messages": [],
        "error_message": None,
        "output_dir": None,
        "figures": {},
        "metrics": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_temp_dir():
    """Get or create a persistent temp directory for this session."""
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = tempfile.mkdtemp(prefix="aspire3d_")
    return st.session_state.temp_dir


def add_log(message, level="INFO"):
    """Add a message to the session log."""
    timestamp = st.session_state.get("log_messages", [])
    timestamp.append({"message": message, "level": level})
    st.session_state.log_messages = timestamp


def clear_logs():
    """Clear the session log."""
    st.session_state.log_messages = []


def get_bundled_config(name):
    """Load a bundled example config by name."""
    config_dir = Path(__file__).parent.parent / "experiments" / "configs"
    config_path = config_dir / f"{name}.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


def save_temp_file(filename, content, binary=False):
    """Save a file to the session temp directory."""
    temp_dir = get_temp_dir()
    path = os.path.join(temp_dir, filename)
    mode = "wb" if binary else "w"
    with open(path, mode) as f:
        f.write(content)
    return path


def get_metrics_from_result(result):
    """Extract key metrics from a simulation result."""
    if result is None:
        return {}
    metrics = {}
    if "simulation" in result and result["simulation"] is not None:
        sim = result["simulation"]
        if hasattr(sim, "history"):
            h = sim.history
            metrics["final_injectivity"] = float(h["injectivity_ratio"][-1]) if "injectivity_ratio" in h else None
            metrics["min_permeability_ratio"] = float(h["k_ratio_min"][-1]) if "k_ratio_min" in h else None
            metrics["final_porosity"] = float(h["phi_mean"][-1]) if "phi_mean" in h else None
            metrics["total_steps"] = len(h.get("pv", []))
    if "calibration" in result and result["calibration"] is not None:
        cal = result["calibration"]
        metrics["calibration_r2"] = cal.r2_total
        metrics["calibration_rmse"] = cal.rmse_total
        metrics["calibration_runs"] = cal.n_eval
    if "sensitivity" in result and result["sensitivity"] is not None:
        sen = result["sensitivity"]
        if isinstance(sen, dict) and "order" in sen:
            metrics["most_influential_param"] = sen["order"][0] if sen["order"] else None
    return metrics


def reset_simulation_state():
    """Reset simulation results (keep config)."""
    st.session_state.simulation_complete = False
    st.session_state.calibration_complete = False
    st.session_state.simulation_result = None
    st.session_state.calibration_result = None
    st.session_state.sensitivity_result = None
    st.session_state.figures = {}
    st.session_state.metrics = {}
    st.session_state.error_message = None
    clear_logs()
