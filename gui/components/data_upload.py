"""
ASPiRe-3D GUI : gui/components/data_upload.py
===============================================================================
Experimental data upload and preview component.
===============================================================================
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from gui.session import get_temp_dir, add_log


def render_data_upload_tab():
    """Render the data upload tab."""
    st.header("📤 Upload Experimental Data")
    
    st.markdown("""
    Upload your core-flood experimental CSV files to calibrate ASPiRe-3D against real data.
    
    **Supported formats:**
    - Differential pressure history (Δp vs time/PV)
    - Produced concentration curves (effluent vs time/PV)
    - Permeability ratio (k/k₀ vs time/PV)
    - Injectivity ratio (I/I₀ vs time/PV)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload CSV Files")
        
        uploaded_files = st.file_uploader(
            "Select CSV files",
            type=["csv"],
            accept_multiple_files=True,
            help="Upload your core-flood measurement CSV files"
        )
        
        if uploaded_files:
            temp_dir = get_temp_dir()
            data_dir = os.path.join(temp_dir, "raw_data")
            os.makedirs(data_dir, exist_ok=True)
            
            for uploaded_file in uploaded_files:
                # Save file
                save_path = os.path.join(data_dir, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"✅ Saved: {uploaded_file.name}")
                add_log(f"Uploaded data: {uploaded_file.name}")
            
            # Store paths in session
            if "uploaded_data_files" not in st.session_state:
                st.session_state.uploaded_data_files = []
            st.session_state.uploaded_data_files = [f.name for f in uploaded_files]
    
    with col2:
        st.subheader("Preview & Map Columns")
        
        if st.session_state.get("uploaded_data_files"):
            files = st.session_state.uploaded_data_files
            selected_file = st.selectbox("Select file to preview", files)
            
            if selected_file:
                # Load and preview
                temp_dir = get_temp_dir()
                file_path = os.path.join(temp_dir, "raw_data", selected_file)
                
                try:
                    df = pd.read_csv(file_path)
                    st.write("**Preview** (first 10 rows):")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Column mapping
                    st.write("**Column mapping:**")
                    col_map = {}
                    cols = list(df.columns)
                    
                    signal_type = st.selectbox(
                        "Signal type",
                        ["Pressure (Δp)", "Concentration", "Permeability (k/k₀)"],
                        help="What does this CSV represent?"
                    )
                    
                    clock_col = st.selectbox("Time/PV column", cols, help="The independent variable (time or PV)")
                    value_col = st.selectbox("Value column", cols, help="The measured quantity")
                    
                    col_map[selected_file] = {
                        "signal_type": signal_type,
                        "clock_column": clock_col,
                        "value_column": value_col
                    }
                    st.session_state.column_mappings = col_map
                    
                except Exception as e:
                    st.error(f"Could not read file: {e}")
        
        else:
            st.info("Upload CSV files to preview and map columns.", icon="ℹ️")


def get_uploaded_data_config():
    """Generate a data block for the config from uploaded files."""
    if not st.session_state.get("uploaded_data_files"):
        return None
    
    # This would be used by the simulator to generate a proper data config
    # For now, return metadata about uploaded files
    return {
        "files": st.session_state.uploaded_data_files,
        "mappings": st.session_state.get("column_mappings", {})
    }
