"""
ASPiRe-3D : core/boundary_conditions.py
===============================================================================
Boundary-condition definitions for a single-phase core flood.

PHYSICAL SETUP
--------------
A core plug sits in a holder. Fluid enters the INLET face (x = 0) and leaves
the OUTLET face (x = L). The cylindrical lateral surface is sealed by the
sleeve -> NO-FLOW. In this masked Cartesian scheme, no-flow is the DEFAULT:
any face whose neighbor is inactive (outside the cylinder) or is the box edge
in y/z contributes ZERO flux simply by being omitted from the assembly. So we
never need an explicit equation for the sleeve -- a clean consequence of FVM.

That leaves only the two end faces to specify. Core-flood experiments are run
in one of two control modes, and we support both because validation data comes
in both forms:

  1. CONSTANT_PRESSURE_DROP (Dirichlet-Dirichlet)
     Fix P = P_in at the inlet face and P = P_out at the outlet face. The flow
     rate is an OUTPUT. Implemented by adding a transmissibility from each
     boundary cell to a ghost value held at the prescribed pressure.

  2. CONSTANT_RATE (Neumann inlet, Dirichlet outlet)
     Inject a fixed total volumetric rate Q [m^3/s] uniformly across the inlet
     face (a flux/Neumann condition -> enters the RHS source vector), and pin
     the outlet at a reference pressure P_out (one Dirichlet anchor is required
     to make the otherwise pure-Neumann system non-singular). The inlet
     pressure -- hence the differential pressure dP that the experiment
     measures -- is an OUTPUT. This is the most common lab configuration.

These objects are pure DATA describing intent. The pressure solver consumes
them during assembly; keeping them declarative keeps the solver readable.
===============================================================================
"""

from enum import Enum


class BCMode(Enum):
    CONSTANT_PRESSURE_DROP = "constant_pressure_drop"
    CONSTANT_RATE = "constant_rate"


class CoreFloodBC:
    """Declarative description of inlet/outlet conditions along the x-axis."""

    def __init__(self, mode,
                 p_inlet=None, p_outlet=0.0,
                 injection_rate=None):
        """
        Parameters
        ----------
        mode : BCMode
        p_inlet : float, optional
            Inlet pressure [Pa] (CONSTANT_PRESSURE_DROP only).
        p_outlet : float
            Outlet (reference) pressure [Pa]. Defaults to 0 Pa gauge; only
            differential pressure is physically meaningful in single-phase
            incompressible flow, so the datum is arbitrary.
        injection_rate : float, optional
            Total volumetric injection rate Q [m^3/s] across the inlet face
            (CONSTANT_RATE only).
        """
        self.mode = mode
        self.p_inlet = p_inlet
        self.p_outlet = p_outlet
        self.injection_rate = injection_rate

        # ---- Validate the combination so misuse fails loudly, early --------
        if mode == BCMode.CONSTANT_PRESSURE_DROP:
            if p_inlet is None:
                raise ValueError("CONSTANT_PRESSURE_DROP requires p_inlet [Pa].")
        elif mode == BCMode.CONSTANT_RATE:
            if injection_rate is None:
                raise ValueError("CONSTANT_RATE requires injection_rate [m^3/s].")
        else:
            raise ValueError(f"Unknown BC mode: {mode}")

    # -----------------------------------------------------------------------
    def summary(self):
        if self.mode == BCMode.CONSTANT_PRESSURE_DROP:
            dp = self.p_inlet - self.p_outlet
            return ("CoreFloodBC: CONSTANT_PRESSURE_DROP\n"
                    f"  P_inlet  = {self.p_inlet:.4e} Pa\n"
                    f"  P_outlet = {self.p_outlet:.4e} Pa\n"
                    f"  dP       = {dp:.4e} Pa  (flow rate is an output)\n")
        else:
            return ("CoreFloodBC: CONSTANT_RATE\n"
                    f"  Q        = {self.injection_rate:.4e} m^3/s "
                    f"({self.injection_rate * 6e7:.3f} mL/min)\n"
                    f"  P_outlet = {self.p_outlet:.4e} Pa (datum)\n"
                    f"  dP is an output\n")
