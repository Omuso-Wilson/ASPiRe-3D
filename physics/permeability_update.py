"""
ASPiRe-3D : physics/permeability_update.py
===============================================================================
Public porosity-permeability damage interface. Re-exports the validated
FormationDamage coupling and the porosity-permeability laws (Kozeny-Carman,
power-law, exponential), which map porosity loss + polymer RRF to permeability.
===============================================================================
"""
from physics.formation_damage import (FormationDamage, kozeny_carman,
                                       power_law, exponential_damage)

__all__ = ["FormationDamage", "kozeny_carman", "power_law", "exponential_damage"]
