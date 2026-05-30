"""
ASPiRe-3D GUI : gui/components/logging.py
===============================================================================
Real-time logging panel for simulation progress and debugging.
===============================================================================
"""

import streamlit as st
from gui.session import st as session_st


def render_logging_panel():
    """Render the real-time logging panel."""
    st.subheader("📝 Execution Log")
    
    # Log viewer
    log_messages = st.session_state.get("log_messages", [])
    
    if not log_messages:
        st.info("No log messages yet. Run a simulation to see progress.", icon="ℹ️")
        return
    
    # Create log display with color coding
    log_text = []
    for entry in log_messages:
        msg = entry.get("message", "")
        level = entry.get("level", "INFO")
        
        # Color code by level
        if level == "ERROR":
            prefix = "❌ ERROR"
        elif level == "WARNING":
            prefix = "⚠️  WARNING"
        elif level == "SUCCESS":
            prefix = "✅ SUCCESS"
        else:
            prefix = "ℹ️  INFO"
        
        log_text.append(f"{prefix}: {msg}")
    
    # Display in a scrollable container
    log_display = st.empty()
    log_display.code("\n".join(log_text), language="text")
    
    # Clear button
    if st.button("🗑️ Clear Logs", help="Clear the execution log"):
        st.session_state.log_messages = []
        st.rerun()


def render_error_panel():
    """Render error message if present."""
    error_msg = st.session_state.get("error_message")
    if error_msg:
        st.error(f"⚠️ Error: {error_msg}")
