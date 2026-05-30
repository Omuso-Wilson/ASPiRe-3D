"""
ASPiRe-3D : physics/darcy_solver.py
===============================================================================
Public Darcy single-phase flow interface. Re-exports the validated pressure
solver and velocity reconstruction (kept in pressure_solver.py / velocity.py to
preserve the verified numerics). This facade gives the package the role-named
entry point requested by the framework architecture without rewriting solvers.

Governing physics (unchanged): u = -(k/mu) grad P, div(u) = q, with harmonic-
mean face transmissibilities and an implicit sparse pressure-Poisson solve.
===============================================================================
"""
from physics.pressure_solver import PressureSolver
from physics.velocity import VelocityField

__all__ = ["PressureSolver", "VelocityField"]
