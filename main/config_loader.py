"""
ASPiRe-3D : main/config_loader.py
===============================================================================
Robust, validated, JSON-driven configuration for ASPiRe-3D.

Removes hardcoded experimental values: every run is fully specified by a JSON
config. The loader parses the file, fills defaults for optional parameters,
validates ranges and unit consistency, converts to the SI/internal units the
solvers expect, and raises clear, actionable error messages on bad input.

CONFIG SCHEMA (sections; * = required)
--------------------------------------
  core        : *length_cm, *diameter_cm, *porosity, *permeability_mD,
                clay_fraction (0-1)
  fluid       : salinity_ppm, temperature_C, pH
  asp         : surfactant, polymer, alkali, salinity   (injected, normalised)
                initial_salinity                          (resident)
  injection   : *flow_rate_ccmin, *pore_volumes
  numerics    : courant, dispersivity_m, reflow_tolerance, grid:[nx,ny,nz]
  physics     : enable_surfactant_adsorption, enable_polymer_retention,
                enable_precipitation, enable_fines, enable_damage,
                perm_model, isotherm  (toggles for validated models only)
  parameters  : physical model parameters (q_max, K_L, rrf_max, ...)
  calibration : enabled, method, parameters:[{name,init,min,max,log}], max_nfev
  sensitivity : enabled, type ('local'|'sobol'), n_base
  uncertainty : enabled, n_samples, burn_in, sigma, step
  output      : outdir, save_figures, save_report, prefix

The loader exposes a `Config` object with both the raw (validated) dict and
convenient SI-converted accessors used by the workflow.
===============================================================================
"""

import os
import json
import copy


# ---------------------------------------------------------------------------
class ConfigError(ValueError):
    """Raised for invalid configuration with an actionable message."""


# ---------------------------------------------------------------------------
#  Defaults for optional parameters (required ones have no default).
# ---------------------------------------------------------------------------
DEFAULTS = {
    "core": {"clay_fraction": 0.05},
    "fluid": {"salinity_ppm": 10000.0, "temperature_C": 25.0, "pH": 7.0},
    "asp": {"surfactant": 1.0, "polymer": 1.0, "alkali": 1.0,
            "salinity": 0.3, "initial_salinity": 1.0},
    "injection": {},
    "numerics": {"courant": 1.0, "dispersivity_m": 5.0e-4,
                 "reflow_tolerance": 0.03, "grid": [30, 12, 12]},
    "physics": {"enable_surfactant_adsorption": True,
                "enable_polymer_retention": True,
                "enable_precipitation": False,
                "enable_fines": False,
                "enable_damage": True,
                "perm_model": "kozeny_carman",
                "isotherm": "langmuir"},
    "parameters": {},
    "calibration": {"enabled": False, "method": "least_squares",
                    "parameters": [], "max_nfev": 60},
    "sensitivity": {"enabled": False, "type": "sobol", "n_base": 16},
    "uncertainty": {"enabled": False, "n_samples": 200, "burn_in": 60,
                    "sigma": 1.0, "step": 0.08},
    "output": {"outdir": "experiments/outputs", "save_figures": True,
               "save_report": True, "prefix": "run"},
}

REQUIRED = {
    "core": ["length_cm", "diameter_cm", "porosity", "permeability_mD"],
    "injection": ["flow_rate_ccmin", "pore_volumes"],
}

# (section, key) -> (low, high, message) range checks
RANGES = {
    ("core", "length_cm"): (0.1, 100.0, "core length must be 0.1–100 cm"),
    ("core", "diameter_cm"): (0.1, 30.0, "core diameter must be 0.1–30 cm"),
    ("core", "porosity"): (0.001, 0.8, "porosity must be 0.001–0.8"),
    ("core", "permeability_mD"): (1e-3, 1e6, "permeability must be 1e-3–1e6 mD"),
    ("core", "clay_fraction"): (0.0, 1.0, "clay_fraction must be 0–1"),
    ("fluid", "salinity_ppm"): (0.0, 3.0e5, "salinity must be 0–300000 ppm"),
    ("fluid", "temperature_C"): (0.0, 200.0, "temperature must be 0–200 °C"),
    ("fluid", "pH"): (0.0, 14.0, "pH must be 0–14"),
    ("injection", "flow_rate_ccmin"): (1e-3, 100.0, "flow rate must be 1e-3–100 cc/min"),
    ("injection", "pore_volumes"): (1e-3, 1000.0, "pore_volumes must be 1e-3–1000"),
    ("numerics", "courant"): (1e-3, 10.0, "courant must be 1e-3–10"),
}

VALID_PERM_MODELS = {"kozeny_carman", "power_law", "exponential"}
VALID_ISOTHERMS = {"langmuir", "freundlich"}
VALID_CALIB_METHODS = {"least_squares", "differential_evolution"}
VALID_SENS_TYPES = {"local", "sobol"}


# ---------------------------------------------------------------------------
def _deep_merge(base, override):
    """Recursively merge override into a copy of base (override wins)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


# ---------------------------------------------------------------------------
class Config:
    """Validated configuration with SI-converted accessors for the workflow."""

    def __init__(self, raw):
        self.raw = raw

    # ---- convenience SI accessors used by the workflow ----
    @property
    def core_length_m(self):
        return self.raw["core"]["length_cm"] * 1.0e-2

    @property
    def core_diameter_m(self):
        return self.raw["core"]["diameter_cm"] * 1.0e-2

    @property
    def porosity(self):
        return self.raw["core"]["porosity"]

    @property
    def permeability_mD(self):
        return self.raw["core"]["permeability_mD"]

    @property
    def flow_rate_ml_min(self):
        # cc/min == mL/min
        return self.raw["injection"]["flow_rate_ccmin"]

    @property
    def pore_volumes(self):
        return self.raw["injection"]["pore_volumes"]

    @property
    def grid(self):
        return tuple(self.raw["numerics"]["grid"])

    def section(self, name):
        return self.raw.get(name, {})

    def get(self, section, key, default=None):
        return self.raw.get(section, {}).get(key, default)

    def __repr__(self):
        return f"Config(core={self.raw['core']}, injection={self.raw['injection']})"

    def summary(self):
        c = self.raw["core"]; inj = self.raw["injection"]; ph = self.raw["physics"]
        active = [k.replace("enable_", "") for k, v in ph.items()
                  if k.startswith("enable_") and v]
        return (f"Config '{self.raw.get('name','unnamed')}':\n"
                f"  core      : L={c['length_cm']} cm, D={c['diameter_cm']} cm, "
                f"phi={c['porosity']}, k={c['permeability_mD']} mD\n"
                f"  injection : {inj['flow_rate_ccmin']} cc/min, "
                f"{inj['pore_volumes']} PV\n"
                f"  physics   : {', '.join(active)}\n"
                f"  calibration enabled: {self.raw['calibration']['enabled']}; "
                f"sensitivity: {self.raw['sensitivity']['enabled']}; "
                f"uncertainty: {self.raw['uncertainty']['enabled']}")


# ---------------------------------------------------------------------------
def _validate(raw):
    """Validate required keys, ranges, enums, and unit consistency."""
    # required keys
    for section, keys in REQUIRED.items():
        if section not in raw:
            raise ConfigError(f"missing required section '{section}'")
        for k in keys:
            if k not in raw[section] or raw[section][k] is None:
                raise ConfigError(f"missing required parameter '{section}.{k}'")

    # range checks
    for (section, key), (lo, hi, msg) in RANGES.items():
        if section in raw and key in raw[section]:
            v = raw[section][key]
            if not isinstance(v, (int, float)):
                raise ConfigError(f"{section}.{key} must be numeric ({msg})")
            if not (lo <= v <= hi):
                raise ConfigError(f"{section}.{key}={v} out of range: {msg}")

    # enums
    pm = raw["physics"]["perm_model"]
    if pm not in VALID_PERM_MODELS:
        raise ConfigError(f"physics.perm_model '{pm}' invalid; "
                          f"choose from {sorted(VALID_PERM_MODELS)}")
    iso = raw["physics"]["isotherm"]
    if iso not in VALID_ISOTHERMS:
        raise ConfigError(f"physics.isotherm '{iso}' invalid; "
                          f"choose from {sorted(VALID_ISOTHERMS)}")
    if raw["calibration"]["enabled"]:
        m = raw["calibration"]["method"]
        if m not in VALID_CALIB_METHODS:
            raise ConfigError(f"calibration.method '{m}' invalid; "
                              f"choose from {sorted(VALID_CALIB_METHODS)}")
        if not raw["calibration"]["parameters"]:
            raise ConfigError("calibration.enabled is true but no "
                              "calibration.parameters were provided")
        for p in raw["calibration"]["parameters"]:
            for field in ("name", "init", "min", "max"):
                if field not in p:
                    raise ConfigError(f"calibration parameter missing '{field}': {p}")
            if not (p["min"] <= p["init"] <= p["max"]):
                raise ConfigError(f"calibration parameter '{p['name']}': "
                                  f"init {p['init']} not in [{p['min']},{p['max']}]")
            if p.get("log") and p["min"] <= 0:
                raise ConfigError(f"calibration parameter '{p['name']}': "
                                  f"log-scaled requires min>0")
    if raw["sensitivity"]["enabled"]:
        st = raw["sensitivity"]["type"]
        if st not in VALID_SENS_TYPES:
            raise ConfigError(f"sensitivity.type '{st}' invalid; "
                              f"choose from {sorted(VALID_SENS_TYPES)}")

    # unit-consistency / physical sanity
    grid = raw["numerics"]["grid"]
    if (not isinstance(grid, list) or len(grid) != 3
            or not all(isinstance(g, int) and g > 0 for g in grid)):
        raise ConfigError("numerics.grid must be [nx,ny,nz] positive integers")
    return raw


# ---------------------------------------------------------------------------
def load_config(path):
    """
    Load, default-fill, and validate a JSON config file. Returns a Config.
    Raises ConfigError with a clear message on any problem.
    """
    if not os.path.exists(path):
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, "r") as f:
            user = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigError(f"invalid JSON in {path}: {e}")
    if not isinstance(user, dict):
        raise ConfigError(f"{path}: top-level JSON must be an object")

    merged = _deep_merge(DEFAULTS, user)
    # preserve a name if given
    if "name" in user:
        merged["name"] = user["name"]
    validated = _validate(merged)
    return Config(validated)


def load_config_dict(user):
    """Same as load_config but from an in-memory dict (for tests/programmatic)."""
    merged = _deep_merge(DEFAULTS, user)
    if "name" in user:
        merged["name"] = user["name"]
    return Config(_validate(merged))
