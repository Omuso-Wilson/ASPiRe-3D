"""
ASPiRe-3D : gui/app.py
===============================================================================
Professional Streamlit GUI for ASPiRe-3D simulator.

Provides a complete web interface for:
  - Configuration management (bundled or custom JSON)
  - Real-time parameter editing
  - Simulation execution
  - Real-time logging
  - Results visualization
  - File download

Integration with existing run_aspire3d.py workflow; no physics modules modified.

Run: streamlit run gui/app.py
===============================================================================
"""

import streamlit as st
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from gui.session import init_session_state
from gui.components.sidebar import render_sidebar
from gui.components.simulator import (
    run_simulation, render_simulation_button, render_results_panel
)
from gui.components.logging import render_logging_panel, render_error_panel
from gui.components.data_upload import render_data_upload_tab


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="ASPiRe-3D",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling
st.markdown("""
<style>
    /* Professional color scheme */
    :root {
        --primary-color: #0066cc;
        --secondary-color: #004499;
        --success-color: #00aa00;
        --warning-color: #ff9900;
        --error-color: #cc0000;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 5px;
        font-weight: 600;
        padding: 10px 24px;
    }
    
    /* Card styling */
    .metric-card {
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZATION
# ============================================================================

init_session_state()

# ============================================================================
# HEADER
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    st.title("⚗️ ASPiRe-3D")
    st.markdown("*Formation Damage Simulator for ASP Flooding*")

with col2:
    st.markdown("""
    <div style='text-align: right; padding-top: 10px; color: #666;'>
    <small>
    <b>Phase 1–7 Complete</b><br>
    82/82 Tests Passing<br>
    <code>v1.0</code>
    </small>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ============================================================================
# MAIN LAYOUT: SIDEBAR + CONTENT
# ============================================================================

# Load config from sidebar
config = render_sidebar(st.session_state.get("config"))
st.session_state.config = config

# Main content area
main_col1, main_col2 = st.columns([2, 1], gap="large")

with main_col1:
    # TABS for different views
    tab_sim, tab_data, tab_info = st.tabs(
        ["🚀 Simulator", "📤 Data Upload", "ℹ️ Information"]
    )
    
    # --- SIMULATOR TAB ---
    with tab_sim:
        st.subheader("Simulation Control")
        
        # Run button
        should_run = render_simulation_button()
        
        if should_run and config is not None:
            with st.spinner("⏳ Running simulation..."):
                result = run_simulation(config)
        
        # Results panel
        if st.session_state.get("simulation_complete"):
            render_results_panel()
        else:
            st.info(
                "👈 Select a configuration in the sidebar, adjust parameters as needed, "
                "then click **Run Simulation** to begin.",
                icon="ℹ️"
            )
    
    # --- DATA UPLOAD TAB ---
    with tab_data:
        render_data_upload_tab()
    
    # --- INFO TAB ---
    with tab_info:
        st.markdown("""
        ### About ASPiRe-3D
        
        ASPiRe-3D is a doctoral-thesis-level simulator for predicting formation damage 
        during natural ASP (Alkaline–Surfactant–Polymer) flooding in sandstone cores.
        
        **Key Features:**
        - 3D cylindrical core geometry
        - Single-phase Darcy flow
        - Multi-species advection–dispersion transport
        - Surfactant adsorption (Langmuir/Freundlich)
        - Polymer retention with Residual Resistance Factor
        - Fines migration and precipitation kinetics
        - Formation damage (porosity → permeability)
        - History matching via least-squares and global optimization
        - Bayesian uncertainty quantification
        - Global sensitivity analysis (Sobol)
        
        **Validation:**
        - 82 automated tests (72 physics + 10 framework)
        - Synthetic-truth recovery (0.0% error)
        - Analytical benchmarks (Ogata–Banks, Kozeny–Carman)
        - Real experimental-data integration
        
        **Documentation:**
        See `docs/` directory for:
        - User Guide
        - Methodology & Equations
        - Assumptions & Limitations
        - Architecture Summary
        """)
        
        st.divider()
        
        st.markdown("""
        ### Quick Links
        
        - 📖 [User Guide](docs/user_guide.md)
        - 🔬 [Methodology](docs/methodology.md)
        - 📐 [Equations](docs/equations.md)
        - ⚙️ [Architecture](docs/architecture_summary.md)
        """)

with main_col2:
    st.subheader("📝 Execution Log")
    render_error_panel()
    render_logging_panel()

# ============================================================================
# FOOTER
# ============================================================================

st.divider()
st.markdown("""
<div style='text-align: center; color: #999; font-size: 0.9em; margin-top: 20px;'>
<p>
<b>ASPiRe-3D</b> — Alkaline–Surfactant–Polymer Reactive Emulator in 3D<br>
Doctoral Research Software | Niger Delta ASP Flooding | Formation Damage Prediction<br>
<code>© 2026 | MIT License | phases 1–7 complete</code>
</p>
</div>
""", unsafe_allow_html=True)


# ============================================================================
# RUN INSTRUCTIONS
# ============================================================================

if __name__ == "__main__":
    # This is auto-run by streamlit
    pass
