"""
ASPiRe-3D : utils/constants.py
===============================================================================
Physical constants, unit conversions, and global numerical tolerances.

DESIGN NOTE
-----------
Reservoir engineering mixes unit systems mercilessly (Darcy, mD, cP, psi, bbl).
To keep the *numerics* unambiguous and bug-free, the entire SOLVER core works
strictly in SI units. Conversions to "field" units (mD, cP, etc.) are confined
to input parsing and reporting only. Centralising the conversion factors here
makes that boundary explicit and auditable for a thesis appendix.

SI base units used throughout the core:
    length        [m]
    permeability  [m^2]
    viscosity     [Pa.s]
    pressure      [Pa]
    density       [kg/m^3]
    volume rate   [m^3/s]
    velocity      [m/s]
===============================================================================
"""

# ---------------------------------------------------------------------------
# Unit conversion factors (multiply a field-unit value by these to get SI)
# ---------------------------------------------------------------------------

# 1 Darcy = 9.869233e-13 m^2  (defined so that 1 cP fluid under 1 atm/cm
# gradient flows at 1 cm/s through 1 cm^2 of 1 Darcy rock). The milliDarcy
# (mD) is the practical unit for sandstones.
DARCY_TO_M2 = 9.869233e-13          # [m^2 / Darcy]
MILLIDARCY_TO_M2 = DARCY_TO_M2 * 1e-3   # [m^2 / mD]

# Viscosity: 1 centipoise = 1e-3 Pa.s. Water ~ 1 cP, ASP polymer slugs can be
# tens to hundreds of cP, which is exactly why mobility (k/mu) matters so much.
CENTIPOISE_TO_PAS = 1.0e-3          # [Pa.s / cP]

# Pressure: useful for reporting differential pressure across a core plug.
PSI_TO_PA = 6.894757e3              # [Pa / psi]
BAR_TO_PA = 1.0e5                   # [Pa / bar]
ATM_TO_PA = 1.01325e5              # [Pa / atm]

# Flow rate: core floods are reported in mL/min or cc/min.
ML_PER_MIN_TO_M3_PER_S = 1.0e-6 / 60.0   # [ (m^3/s) / (mL/min) ]

# ---------------------------------------------------------------------------
# Numerical tolerances
# ---------------------------------------------------------------------------
# Floor used when forming harmonic means of permeability so that a fully
# plugged (k -> 0) cell does not produce a division-by-zero. It also encodes
# the physical statement "no rock is perfectly impermeable" at solver level.
PERMEABILITY_FLOOR_M2 = 1.0e-22     # ~1e-10 mD : effectively sealed

# Tolerance used to decide whether two floating point geometry values are equal.
GEOMETRIC_TOL = 1.0e-12
