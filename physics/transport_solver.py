"""
ASPiRe-3D : physics/transport_solver.py
===============================================================================
Public advection-dispersion transport interface. Re-exports the validated
explicit and implicit tracer transport and the sparse advection-dispersion
operator (M-matrix), preserving the verified conservative finite-volume scheme.
===============================================================================
"""
from physics.transport import TracerTransport, ImplicitTracerTransport
from physics.transport_operator import TransportOperator

__all__ = ["TracerTransport", "ImplicitTracerTransport", "TransportOperator"]
