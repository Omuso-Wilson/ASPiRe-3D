"""
ASPiRe-3D : physics/fines_migration.py
===============================================================================
Public fines-migration & reaction-kinetics interface. Re-exports the validated
kinetics models: fines detachment/deposition (critical-velocity), precipitation/
dissolution, and the composite reaction container.
===============================================================================
"""
from physics.kinetics import (FinesMigrationKinetics, PrecipitationKinetics,
                              CompositeReactionModel)

__all__ = ["FinesMigrationKinetics", "PrecipitationKinetics",
           "CompositeReactionModel"]
