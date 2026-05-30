"""
ASPiRe-3D : visualization/sensitivity_plots.py
===============================================================================
Public sensitivity plotting interface. Re-exports the tornado chart (local
elasticities or Sobol total-effect indices).
===============================================================================
"""
from visualization.calibration_plots import plot_tornado

__all__ = ["plot_tornado"]
