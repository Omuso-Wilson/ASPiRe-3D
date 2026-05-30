"""
ASPiRe-3D : core/experiment.py
===============================================================================
EXPERIMENT CONFIGURATION and data ingestion for history matching.

Provides the bridge between (a) laboratory core-flood data on disk and (b) the
history-matching engine, plus a reusable factory that builds the forward model
(parameters -> simulated signals). Designed so that the day real Niger Delta
core-flood CSVs are available, they load with ZERO code changes.

DATA FORMAT (CSV)
-----------------
Each signal is a 2-column CSV (header optional): the first column is the clock
(pore volumes or seconds), the second the measured value. A small JSON/dict
config names the signals, their files, units, and weights. See load_observed().

FORWARD-MODEL FACTORY
---------------------
build_asp_forward_model() returns a builder(param_dict) -> {signal:(t,y)} that:
  * constructs fresh properties + species + reaction models from the parameters,
  * runs a CoupledSimulator,
  * extracts the requested signals (dp_history, surfactant_effluent,
    k_ratio_history, injectivity_history, ...).
The mapping from calibration-parameter names to physical model parameters is
explicit and centralised here, so it is auditable and reproducible.
===============================================================================
"""

import os
import numpy as np

from physics.mesh import StructuredMesh
from physics.geometry import apply_cylindrical_core
from physics.properties import FluidRockProperties
from physics.boundary_conditions import CoreFloodBC, BCMode
from physics.species import Species, SpeciesRegistry
from physics.adsorption import SurfactantAdsorption, PolymerRetention
from physics.kinetics import (PrecipitationKinetics, FinesMigrationKinetics,
                           CompositeReactionModel)
from physics.formation_damage import FormationDamage
from physics.coupled_simulator import CoupledSimulator
from calibration.calibration import ObservedData
from utils.constants import ML_PER_MIN_TO_M3_PER_S


# ===========================================================================
def load_observed_from_csv(config):
    """
    Load observed signals from CSV files described by a config dict:

        config = {
          "signals": {
            "dp_history":         {"file": "dp.csv",  "weight": 1.0},
            "surfactant_effluent":{"file": "surf.csv","weight": 1.0},
          },
          "clock": "pore_volumes"   # or "seconds"
        }

    Returns an ObservedData. CSVs are 2-column (clock, value); a header row is
    auto-detected and skipped.
    """
    obs = ObservedData()
    base = config.get("base_dir", ".")
    for name, spec in config["signals"].items():
        path = os.path.join(base, spec["file"])
        # Auto-skip a non-numeric header.
        try:
            data = np.loadtxt(path, delimiter=",")
        except ValueError:
            data = np.loadtxt(path, delimiter=",", skiprows=1)
        t, y = data[:, 0], data[:, 1]
        obs.add(name, t, y, weight=spec.get("weight", 1.0))
    return obs


def observed_from_arrays(signal_dict):
    """
    Build ObservedData directly from in-memory arrays (for synthetic studies):
        signal_dict = {name: (times, values, weight)}
    """
    obs = ObservedData()
    for name, tup in signal_dict.items():
        if len(tup) == 3:
            t, y, w = tup
        else:
            t, y = tup; w = 1.0
        obs.add(name, np.asarray(t, float), np.asarray(y, float), weight=w)
    return obs


# ===========================================================================
class ASPExperiment:
    """
    A reproducible ASP core-flood experiment configuration. Holds the fixed
    (non-calibrated) setup -- geometry, rate, grid, injected composition, run
    length, which signals to output -- so a calibration is fully specified by
    this config plus the calibration parameters.
    """

    def __init__(self,
                 core_length=0.10, core_diameter=0.038,
                 nx=30, ny=10, nz=10,
                 rate_ml_min=2.0,
                 n_pore_volumes=3.0,
                 courant=1.0,
                 perm_mD=100.0, porosity=0.20, viscosity_cP=1.0,
                 # injected ASP slug composition (defaults; overridable)
                 inlet=None,
                 initial=None,
                 dispersivity=5.0e-4,
                 reflow_tol=0.03,
                 enable_precipitation=False,
                 enable_fines=False,
                 enable_surfactant_adsorption=True,
                 enable_polymer_retention=True,
                 enable_damage=True,
                 output_signals=("dp_history", "surfactant_effluent",
                                 "k_ratio_history", "injectivity_history")):
        self.core_length = core_length
        self.core_diameter = core_diameter
        self.nx, self.ny, self.nz = nx, ny, nz
        self.rate_ml_min = rate_ml_min
        self.n_pore_volumes = n_pore_volumes
        self.courant = courant
        self.perm_mD = perm_mD
        self.porosity = porosity
        self.viscosity_cP = viscosity_cP
        self.inlet = inlet or dict(salinity=0.3, alkali=1.0, surfactant=1.0,
                                   polymer=1.0)
        self.initial = initial or dict(salinity=1.0)
        self.dispersivity = dispersivity
        self.reflow_tol = reflow_tol
        self.enable_precipitation = enable_precipitation
        self.enable_fines = enable_fines
        self.enable_surfactant_adsorption = enable_surfactant_adsorption
        self.enable_polymer_retention = enable_polymer_retention
        self.enable_damage = enable_damage
        self.output_signals = tuple(output_signals)

        # Build the (reused) mesh once.
        self.mesh = StructuredMesh(core_length, core_diameter, core_diameter,
                                   nx, ny, nz)
        apply_cylindrical_core(self.mesh, core_diameter)

    # -----------------------------------------------------------------------
    def _build_species(self):
        sp = []
        for nm in ("salinity", "alkali", "surfactant", "polymer"):
            sp.append(Species(nm, mobile=True,
                              inlet_value=self.inlet.get(nm, 0.0),
                              initial_value=self.initial.get(nm, 0.0),
                              molecular_diffusion=1.0e-9))
        if self.enable_surfactant_adsorption:
            sp.append(Species("surfactant_adsorbed", mobile=False))
        if self.enable_polymer_retention:
            sp.append(Species("polymer_retained", mobile=False))
        if self.enable_precipitation:
            sp.append(Species("precipitate", mobile=False))
        if self.enable_fines:
            sp.append(Species("fines_suspended", mobile=True, molecular_diffusion=1e-11))
            sp.append(Species("fines_deposited", mobile=False, initial_value=2.0))
        return SpeciesRegistry(sp)

    # -----------------------------------------------------------------------
    def run(self, params):
        """
        Build and run a CoupledSimulator with calibration `params` (dict).
        Returns the simulator (for signal extraction).

        Parameter names recognised (all optional; defaults used if absent):
          q_max, K_L, k_ads, ads_salinity_coeff, ads_alkali_coeff   (surfactant)
          sigma_max, k_ret, rrf_max, perm_gamma                     (polymer)
          k_precip, k_dissolve                                      (precip)
          k_detach, k_deposit, critical_velocity                   (fines)
          perm_exponent                                            (damage law)
        """
        mesh = self.mesh
        props = FluidRockProperties(mesh, self.perm_mD, self.porosity,
                                    self.viscosity_cP, 1000.0)
        reg = self._build_species()
        Q = self.rate_ml_min * ML_PER_MIN_TO_M3_PER_S
        bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)

        models = []
        fines_model = None
        poly_model = None
        if self.enable_surfactant_adsorption:
            models.append(SurfactantAdsorption(
                reg, props, isotherm="langmuir",
                q_max=params.get("q_max", 1.5e-4),
                K_L=params.get("K_L", 4.0),
                rate_constant=params.get("k_ads", 3.0e-2),
                bulk_density=2000.0,
                salinity_name="salinity",
                salinity_coeff=params.get("ads_salinity_coeff", 0.3),
                alkali_name="alkali",
                alkali_coeff=params.get("ads_alkali_coeff", 0.4)))
        if self.enable_polymer_retention:
            poly_model = PolymerRetention(
                reg, props,
                sigma_max=params.get("sigma_max", 2.0e-4),
                rate_constant=params.get("k_ret", 2.0e-2),
                perm_dependence_gamma=params.get("perm_gamma", 0.5))
            models.append(poly_model)
        if self.enable_precipitation:
            models.append(PrecipitationKinetics(
                reg, k_precip=params.get("k_precip", 0.05),
                k_dissolve=params.get("k_dissolve", 0.02),
                alkali_ref=0.5, salinity_ref=0.5, consume_per_precipitate=0.0))
        if self.enable_fines:
            fines_model = FinesMigrationKinetics(
                reg, None, props,
                critical_velocity=params.get("critical_velocity", 2e-5),
                k_detach=params.get("k_detach", 2e-3),
                k_deposit=params.get("k_deposit", 5e1))
            models.append(fines_model)

        reaction = CompositeReactionModel(models) if models else None

        damage = None
        if self.enable_damage:
            damage = FormationDamage(
                mesh, props, reg, perm_model="power_law",
                perm_kwargs={"exponent": params.get("perm_exponent", 3.0)},
                polymer_retained_name=("polymer_retained"
                                       if self.enable_polymer_retention else None),
                rrf_max=params.get("rrf_max", 4.0),
                sigma_ref_polymer=params.get("sigma_max", 2.0e-4),
                surfactant_adsorbed_name=("surfactant_adsorbed"
                                          if self.enable_surfactant_adsorption else None),
                precipitate_name=("precipitate" if self.enable_precipitation else "precipitate"),
                fines_deposited_name=("fines_deposited" if self.enable_fines else "fines_deposited"))

        sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=reaction,
                               formation_damage=damage,
                               longitudinal_dispersivity=self.dispersivity,
                               reflow_k_tolerance=self.reflow_tol,
                               fines_model=fines_model)
        if poly_model is not None:
            poly_model.update_velocity(sim.velocity, props)
        if fines_model is not None:
            fines_model.update_velocity(sim.velocity, props)

        dx = min(mesh.dx, mesh.dy, mesh.dz)
        vmax = float(np.max(sim.velocity.speed[mesh.active] /
                            np.maximum(props.phi[mesh.active], 1e-6)))
        sim.run(dt=self.courant * dx / vmax,
                n_pore_volumes=self.n_pore_volumes, verbose=False)
        return sim

    # -----------------------------------------------------------------------
    def extract_signals(self, sim):
        """Return {signal_name: (clock_pv, values)} for the requested signals."""
        pv = np.array(sim.history["pv"])
        out = {}
        if "dp_history" in self.output_signals:
            out["dp_history"] = (pv, np.array(sim.history["dP"]))
        if "injectivity_history" in self.output_signals:
            out["injectivity_history"] = (pv, np.array(sim.history["injectivity_ratio"]))
        if "k_ratio_history" in self.output_signals:
            out["k_ratio_history"] = (pv, np.array(sim.history["k_ratio_mean"]))
        if "surfactant_effluent" in self.output_signals:
            out["surfactant_effluent"] = (pv, np.array(sim.history["outlet"]["surfactant"]))
        if "polymer_effluent" in self.output_signals:
            out["polymer_effluent"] = (pv, np.array(sim.history["outlet"]["polymer"]))
        if "salinity_effluent" in self.output_signals:
            out["salinity_effluent"] = (pv, np.array(sim.history["outlet"]["salinity"]))
        return out

    def make_builder(self):
        """Return builder(param_dict) -> {signal:(t,y)} for the ForwardModel."""
        def builder(param_dict):
            sim = self.run(param_dict)
            return self.extract_signals(sim)
        return builder
