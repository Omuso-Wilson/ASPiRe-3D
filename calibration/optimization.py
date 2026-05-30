"""
ASPiRe-3D : calibration/optimization.py
===============================================================================
Public optimization interface for history matching. Re-exports the validated
optimizer-driven engine (least_squares + differential_evolution) and result
container, preserving the verified calibration numerics.
===============================================================================
"""
from calibration.history_matching import (ForwardModel, HistoryMatcher,
                                          CalibrationResult)

__all__ = ["ForwardModel", "HistoryMatcher", "CalibrationResult"]
