"""
ASPiRe-3D : core/data_schema.py
===============================================================================
Experimental-data SCHEMA, TEMPLATES, and unit handling for core-flood CSV
ingestion. This is the contract between a laboratory CSV and ASPiRe-3D: an
experimentalist describes THEIR file (column names, units, flood metadata) in a
small config, and the framework maps it onto the canonical signals the
simulator produces -- no code changes, no manual unit juggling.

CANONICAL SIGNALS (what the simulator outputs and the calibrator matches)
-------------------------------------------------------------------------
    dp_history            differential pressure across the core   [Pa]
    injectivity_history   injectivity ratio I/I0                  [-]
    k_ratio_history       permeability ratio k/k0                 [-]
    surfactant_effluent   produced surfactant C/C0                [-]
    polymer_effluent      produced polymer C/C0                   [-]
    salinity_effluent     produced salinity (normalised)          [-]

CANONICAL CLOCK: pore volumes injected (PV). The simulator's natural clock.
Raw data may be in seconds, minutes, or PV; the loader converts using the
flood rate and pore volume declared in the experiment metadata.

UNIT HANDLING
-------------
Each mapped column declares its raw unit; convert_to_canonical() converts it to
the canonical SI/normalised unit. Pressure: Pa/kPa/bar/psi/atm. Time:
s/min/hr/PV. Concentrations are normalised by a declared injected reference
(C0) so the canonical signal is dimensionless C/C0.
===============================================================================
"""

import numpy as np
from utils.constants import PSI_TO_PA, BAR_TO_PA, ATM_TO_PA


# ---------------------------------------------------------------------------
#  Unit conversion factors to canonical units
# ---------------------------------------------------------------------------
PRESSURE_TO_PA = {
    "pa": 1.0, "kpa": 1.0e3, "mpa": 1.0e6,
    "bar": BAR_TO_PA, "mbar": BAR_TO_PA * 1e-3,
    "psi": PSI_TO_PA, "atm": ATM_TO_PA,
}
TIME_TO_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0,
                   "min": 60.0, "minute": 60.0,
                   "hr": 3600.0, "hour": 3600.0, "h": 3600.0}

CANONICAL_SIGNALS = (
    "dp_history", "injectivity_history", "k_ratio_history",
    "surfactant_effluent", "polymer_effluent", "salinity_effluent")


# ---------------------------------------------------------------------------
class ExperimentMetadata:
    """
    Physical metadata for a core-flood experiment, used to convert raw clocks
    to pore volumes and to normalise/derive signals. All in SI unless noted.
    """

    def __init__(self, core_length_m, core_diameter_m, porosity,
                 flow_rate_ml_min, baseline_permeability_mD=None,
                 baseline_dp_pa=None, injected_concentrations=None,
                 name="core_flood"):
        self.name = name
        self.core_length = float(core_length_m)
        self.core_diameter = float(core_diameter_m)
        self.porosity = float(porosity)
        self.flow_rate_ml_min = float(flow_rate_ml_min)
        self.baseline_permeability_mD = baseline_permeability_mD
        self.baseline_dp_pa = baseline_dp_pa
        # injected reference concentrations C0 for normalising effluent signals.
        self.injected_concentrations = injected_concentrations or {}

    # ---- derived quantities ------------------------------------------------
    @property
    def bulk_volume_m3(self):
        r = 0.5 * self.core_diameter
        return np.pi * r * r * self.core_length

    @property
    def pore_volume_m3(self):
        return self.porosity * self.bulk_volume_m3

    @property
    def flow_rate_m3_s(self):
        return self.flow_rate_ml_min * 1.0e-6 / 60.0

    @property
    def seconds_per_pore_volume(self):
        return self.pore_volume_m3 / self.flow_rate_m3_s

    def summary(self):
        return (f"ExperimentMetadata '{self.name}'\n"
                f"  core           : L={self.core_length:.4f} m, "
                f"D={self.core_diameter:.4f} m, phi={self.porosity:.3f}\n"
                f"  pore volume    : {self.pore_volume_m3:.4e} m^3\n"
                f"  flow rate      : {self.flow_rate_ml_min:.3f} mL/min "
                f"({self.flow_rate_m3_s:.3e} m^3/s)\n"
                f"  1 PV           : {self.seconds_per_pore_volume:.1f} s\n"
                f"  baseline k     : {self.baseline_permeability_mD} mD\n"
                f"  baseline dP    : {self.baseline_dp_pa} Pa\n")


# ---------------------------------------------------------------------------
class SignalMapping:
    """
    Maps a raw CSV column to a canonical signal, declaring units and (for
    effluent) the normalising reference. One mapping per signal to be used.
    """

    def __init__(self, canonical, column, unit=None, clock_column=None,
                 clock_unit="PV", normalise_by=None, uncertainty=None,
                 uncertainty_column=None):
        """
        Parameters
        ----------
        canonical : one of CANONICAL_SIGNALS
        column : raw column name (or index) holding the values
        unit : raw value unit (e.g. 'psi','bar','kPa' for pressure; None if
            already canonical/dimensionless)
        clock_column : raw column name (or index) holding the clock
        clock_unit : 's','min','hr', or 'PV'
        normalise_by : float C0 to divide effluent by (-> C/C0); None if raw is
            already normalised
        uncertainty : float, constant 1-sigma measurement uncertainty in the
            canonical unit (optional)
        uncertainty_column : raw column with per-point 1-sigma uncertainty
            (optional; overrides `uncertainty`)
        """
        if canonical not in CANONICAL_SIGNALS:
            raise ValueError(f"unknown canonical signal '{canonical}'; "
                             f"choose from {CANONICAL_SIGNALS}")
        self.canonical = canonical
        self.column = column
        self.unit = unit.lower() if isinstance(unit, str) else unit
        self.clock_column = clock_column
        self.clock_unit = clock_unit
        self.normalise_by = normalise_by
        self.uncertainty = uncertainty
        self.uncertainty_column = uncertainty_column

    # -----------------------------------------------------------------------
    def convert_values_to_canonical(self, raw_values):
        """Convert raw values to the canonical unit for this signal."""
        v = np.asarray(raw_values, float)
        if self.canonical == "dp_history":
            if self.unit is None:
                return v
            if self.unit not in PRESSURE_TO_PA:
                raise ValueError(f"unknown pressure unit '{self.unit}'")
            return v * PRESSURE_TO_PA[self.unit]
        # ratio / effluent signals: dimensionless. Optional normalisation.
        if self.normalise_by:
            return v / float(self.normalise_by)
        return v

    def convert_clock_to_pv(self, raw_clock, metadata):
        """Convert a raw clock column to pore volumes injected."""
        c = np.asarray(raw_clock, float)
        unit = (self.clock_unit or "PV").lower()
        if unit == "pv":
            return c
        if unit not in TIME_TO_SECONDS:
            raise ValueError(f"unknown clock unit '{self.clock_unit}'")
        seconds = c * TIME_TO_SECONDS[unit]
        return seconds / metadata.seconds_per_pore_volume


# ---------------------------------------------------------------------------
def csv_template(signal="dp_history"):
    """
    Return a documented CSV template string for a given signal, so an
    experimentalist knows exactly what columns ASPiRe-3D expects to be told
    about. The column NAMES are free; the mapping config declares them.
    """
    templates = {
        "dp_history": (
            "# Differential-pressure history template\n"
            "# columns (names are free; declare them in the SignalMapping):\n"
            "#   <clock>, <pressure>[, <pressure_sigma>]\n"
            "time_min,dP_psi,dP_sigma_psi\n"
            "0.0,12.5,0.3\n5.0,13.1,0.3\n10.0,15.8,0.4\n"),
        "surfactant_effluent": (
            "# Surfactant effluent (produced concentration) template\n"
            "#   <clock>, <concentration>[, <sigma>]\n"
            "PV,C_surf_ppm,sigma_ppm\n"
            "0.0,0.0,5\n0.5,0.0,5\n1.0,120.0,8\n1.5,640.0,12\n"),
        "k_ratio_history": (
            "# Permeability-ratio history template (k/k0)\n"
            "#   <clock>, <k_over_k0>\n"
            "PV,k_ratio\n0.0,1.0\n1.0,0.62\n2.0,0.31\n"),
    }
    return templates.get(signal, templates["dp_history"])


def example_config():
    """
    Return an example mapping config (as a Python dict) showing how a lab CSV
    is described. This is what a user edits to integrate their own data.
    """
    return {
        "metadata": dict(
            name="NigerDelta_coreA",
            core_length_m=0.10, core_diameter_m=0.038, porosity=0.21,
            flow_rate_ml_min=2.0, baseline_permeability_mD=120.0,
            injected_concentrations=dict(surfactant=2000.0, polymer=1500.0)),
        "signals": [
            dict(canonical="dp_history", file="dp.csv",
                 column="dP_psi", unit="psi",
                 clock_column="time_min", clock_unit="min",
                 uncertainty_column="dP_sigma_psi"),
            dict(canonical="surfactant_effluent", file="surf.csv",
                 column="C_surf_ppm", clock_column="PV", clock_unit="PV",
                 normalise_by=2000.0, uncertainty=0.02),
        ],
    }
