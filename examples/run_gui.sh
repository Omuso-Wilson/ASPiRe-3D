#!/bin/bash
# examples/run_gui.sh
# Quick-launch script for the ASPiRe-3D Streamlit GUI

set -e

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                     ASPiRe-3D Streamlit GUI                        ║"
echo "║              Alkaline–Surfactant–Polymer Reactive Emulator         ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python $(python3 --version | cut -d' ' -f2)"

# Check Streamlit
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "ℹ️  Streamlit not installed. Installing dependencies..."
    pip install -q streamlit>=1.28 pillow
fi

echo "✓ Dependencies ready"
echo ""
echo "🚀 Launching GUI at http://localhost:8501"
echo "   Press Ctrl+C to stop"
echo ""

# Launch Streamlit
cd "$(dirname "$0")/.."
streamlit run gui/app.py \
    --theme.primaryColor="#0066cc" \
    --theme.backgroundColor="#ffffff" \
    --theme.secondaryBackgroundColor="#f0f2f6" \
    --client.showErrorDetails=true
