#!/bin/bash
# Quick launch script for the ASPiRe-3D GUI
echo "Starting ASPiRe-3D Streamlit GUI..."
streamlit run gui/app.py --theme.primaryColor="#0066cc" --theme.backgroundColor="#ffffff" --theme.secondaryBackgroundColor="#f0f2f6"
