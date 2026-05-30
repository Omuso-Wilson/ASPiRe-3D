"""
ASPiRe-3D : tests/test_phase5.py
===============================================================================
Validation suite for adsorption & retention (Phase 5).

Covers every objective requirement:
  * Langmuir & Freundlich isotherm correctness and limits,
  * adsorption mass conservation (dissolved loss == adsorbed gain, mass-weighted),
  * concentration RETARDATION (adsorbing front breaks through later than 1 PV),
  * irreversible adsorption (no desorption),
  * salinity-sensitive and pH(alkali)-dependent adsorption modifiers,
  * polymer retention + Residual Resistance Factor permeability reduction,
  * permeability-dependent retention,
  * shear degradation,
  * calibration framework (parameter vector round-trip, objective, sensitivity).
===============================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from physics.mesh import StructuredMesh
from physics.geometry import apply_cylindrical_core
from physics.properties import FluidRockProperties
from physics.boundary_conditions import CoreFloodBC, BCMode
from physics.species import Species, SpeciesRegistry
from physics.adsorption import (SurfactantAdsorption, PolymerRetention,
                             langmuir_isotherm, freundlich_isotherm)
from physics.formation_damage import FormationDamage
from physics.coupled_simulator import CoupledSimulator
from calibration.calibration import (CalibrationParameters, ObservedData,
                              ObjectiveFunction, SensitivityAnalysis)
from utils.constants import ML_PER_MIN_TO_M3_PER_S


def _ctx(nx=40, ny=10, nz=10):
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, 0.038)
    props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    Q = 2.0 * ML_PER_MIN_TO_M3_PER_S
    bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    return mesh, props, bc


def _dt(mesh, props, sim, courant=0.5):
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] /
                        np.maximum(props.phi[mesh.active], 1e-6)))
    return courant * dx / vmax


def _breakthrough_pv(sim, name, level=0.5):
    pv = np.array(sim.history["pv"]); out = np.array(sim.history["outlet"][name])
    idx = np.argmax(out >= level)
    return pv[idx] if out[idx] >= level else np.nan


# ===========================================================================
#  ISOTHERMS
# ===========================================================================
def test_isotherm_limits():
    """Langmuir saturates to q_max; Freundlich is monotone increasing; both 0 at C=0."""
    C = np.linspace(0, 100, 50)
    qL = langmuir_isotherm(C, q_max=1.0, K_L=2.0)
    assert qL[0] == 0.0 and np.all(np.diff(qL) >= 0)
    assert qL[-1] < 1.0 and abs(qL[-1] - 1.0) < 0.01, "Langmuir should approach q_max"
    qF = freundlich_isotherm(C, K_F=0.5, n_F=2.0)
    assert qF[0] == 0.0 and np.all(np.diff(qF) >= 0), "Freundlich not monotone"
    print("PASS  isotherms: Langmuir saturates to q_max, Freundlich monotone, both 0 at C=0")


# ===========================================================================
#  MASS CONSERVATION
# ===========================================================================
def test_adsorption_mass_conservation():
    """
    Mass-weighted: dissolved rate (phi*V) + adsorbed rate (V) sums to zero
    everywhere -> adsorption moves mass, never creates/destroys it.
    """
    mesh, props, _ = _ctx(nx=12, ny=8, nz=8)
    reg = SpeciesRegistry([
        Species("surfactant", True, initial_value=1.0),
        Species("surfactant_adsorbed", False, initial_value=0.0)])
    ads = SurfactantAdsorption(reg, props, isotherm="langmuir",
                               q_max=2e-4, K_L=5.0, rate_constant=5e-2,
                               bulk_density=2000.0)
    state = np.zeros((2, mesh.n_active)); state[0, :] = 1.0
    r = ads.rates(state, reg, mesh, props)
    V = mesh.cell_volume
    phi = props.phi[mesh.active]
    diss_mass_rate = phi * V * r[0, :]
    ads_mass_rate = V * r[1, :]
    couple = diss_mass_rate + ads_mass_rate
    assert np.max(np.abs(couple)) < 1e-18, f"adsorption not conservative: {np.max(np.abs(couple)):.2e}"
    print("PASS  adsorption mass conservation: dissolved loss == adsorbed gain (mass-weighted)")


# ===========================================================================
#  RETARDATION
# ===========================================================================
def test_concentration_retardation():
    """Adsorbing surfactant must break through LATER than a conservative tracer."""
    mesh, props, bc = _ctx()
    # conservative
    regc = SpeciesRegistry([Species("surfactant", True, inlet_value=1.0, molecular_diffusion=5e-10)])
    simc = CoupledSimulator(mesh, props, regc, bc, reaction_model=None,
                            formation_damage=None, longitudinal_dispersivity=3e-4)
    simc.run(dt=_dt(mesh, props, simc), n_pore_volumes=3.0, verbose=False)
    pv_c = _breakthrough_pv(simc, "surfactant")

    # adsorbing (fresh properties)
    props2 = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    rega = SpeciesRegistry([
        Species("surfactant", True, inlet_value=1.0, molecular_diffusion=5e-10),
        Species("surfactant_adsorbed", False, initial_value=0.0)])
    ads = SurfactantAdsorption(rega, props2, isotherm="langmuir",
                               q_max=2e-4, K_L=5.0, rate_constant=5e-2, bulk_density=2000.0)
    sima = CoupledSimulator(mesh, props2, rega, bc, reaction_model=ads,
                            formation_damage=None, longitudinal_dispersivity=3e-4)
    sima.run(dt=_dt(mesh, props2, sima), n_pore_volumes=3.0, verbose=False)
    pv_a = _breakthrough_pv(sima, "surfactant")

    assert pv_a > pv_c + 0.2, f"no retardation: conservative {pv_c:.2f}, adsorbing {pv_a:.2f}"
    print(f"PASS  retardation: adsorbing breakthrough {pv_a:.2f} PV > conservative {pv_c:.2f} PV")


# ===========================================================================
#  IRREVERSIBILITY
# ===========================================================================
def test_irreversible_adsorption():
    """Irreversible adsorption never desorbs: rate >= 0 even when q > q_eq."""
    mesh, props, _ = _ctx(nx=10, ny=8, nz=8)
    reg = SpeciesRegistry([
        Species("surfactant", True, initial_value=0.0),     # no dissolved -> q_eq=0
        Species("surfactant_adsorbed", False, initial_value=1e-4)])  # but q present
    ads = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=2e-4,
                               K_L=5.0, rate_constant=5e-2, bulk_density=2000.0,
                               irreversible=True)
    state = np.zeros((2, mesh.n_active)); state[1, :] = 1e-4
    r = ads.rates(state, reg, mesh, props)
    assert np.all(r[1, :] >= -1e-18), "irreversible adsorption desorbed"
    # Reversible version SHOULD desorb (negative) here.
    ads_rev = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=2e-4,
                                   K_L=5.0, rate_constant=5e-2, bulk_density=2000.0,
                                   irreversible=False)
    r_rev = ads_rev.rates(state, reg, mesh, props)
    assert np.all(r_rev[1, :] <= 0), "reversible should desorb when q>q_eq"
    print("PASS  irreversibility: irreversible rate>=0; reversible desorbs when q>q_eq")


# ===========================================================================
#  SALINITY & pH MODIFIERS
# ===========================================================================
def test_salinity_sensitive_adsorption():
    """Higher salinity increases equilibrium adsorption (charge screening)."""
    mesh, props, _ = _ctx(nx=10, ny=8, nz=8)
    reg = SpeciesRegistry([
        Species("surfactant", True), Species("surfactant_adsorbed", False),
        Species("salinity", True)])
    ads = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=2e-4,
                               K_L=5.0, rate_constant=5e-2, bulk_density=2000.0,
                               salinity_name="salinity", salinity_coeff=0.5)
    C = np.full(mesh.n_active, 1.0)
    q_low = ads.equilibrium_adsorbed(C, salinity=np.full(mesh.n_active, 0.1)).mean()
    q_high = ads.equilibrium_adsorbed(C, salinity=np.full(mesh.n_active, 2.0)).mean()
    assert q_high > q_low, "salinity did not increase adsorption"
    print(f"PASS  salinity-sensitive adsorption: q rises with salinity "
          f"({q_high/q_low:.2f}x from 0.1->2.0)")


def test_ph_dependent_adsorption():
    """Higher alkali (pH) reduces surfactant adsorption."""
    mesh, props, _ = _ctx(nx=10, ny=8, nz=8)
    reg = SpeciesRegistry([
        Species("surfactant", True), Species("surfactant_adsorbed", False),
        Species("alkali", True)])
    ads = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=2e-4,
                               K_L=5.0, rate_constant=5e-2, bulk_density=2000.0,
                               alkali_name="alkali", alkali_coeff=0.4)
    C = np.full(mesh.n_active, 1.0)
    q_lowpH = ads.equilibrium_adsorbed(C, alkali=np.full(mesh.n_active, 0.0)).mean()
    q_highpH = ads.equilibrium_adsorbed(C, alkali=np.full(mesh.n_active, 1.0)).mean()
    assert q_highpH < q_lowpH, "alkali did not reduce adsorption"
    print(f"PASS  pH-dependent adsorption: q falls with alkali "
          f"({q_highpH/q_lowpH:.2f}x at alkali 0->1)")


# ===========================================================================
#  POLYMER RETENTION + RRF
# ===========================================================================
def test_polymer_retention_and_rrf():
    """
    Polymer retention reduces permeability via RRF, producing a permeability
    drop and injectivity decline even without porosity-occupancy damage.
    """
    mesh, props, bc = _ctx()
    reg = SpeciesRegistry([
        Species("polymer", True, inlet_value=1.0, molecular_diffusion=1e-10),
        Species("polymer_retained", False, initial_value=0.0)])
    ret = PolymerRetention(reg, props, dissolved_name="polymer",
                           retained_name="polymer_retained",
                           sigma_max=3e-4, rate_constant=3e-2)
    dmg = FormationDamage(mesh, props, reg, perm_model="kozeny_carman",
                          polymer_retained_name="polymer_retained",
                          rrf_max=5.0, sigma_ref_polymer=3e-4)
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=ret,
                           formation_damage=dmg, reflow_k_tolerance=0.05)
    sim.run(dt=_dt(mesh, props, sim, courant=1.0), n_pore_volumes=2.0, verbose=False)
    kmin = sim.history["k_ratio_min"][-1]
    inj = sim.history["injectivity_ratio"][-1]
    assert kmin < 0.95, f"RRF did not reduce permeability (k/k0={kmin:.3f})"
    assert inj < 0.95, f"no injectivity decline from retention (I/I0={inj:.3f})"
    # retained mass present
    assert sim.C[reg.index("polymer_retained"), :].max() > 0
    print(f"PASS  polymer retention + RRF: k/k0(min)={kmin:.3f}, I/I0={inj:.3f} "
          f"(permeability reduced by retention)")


def test_permeability_dependent_retention():
    """Lower-permeability cells retain more polymer (capacity ~ (k/k0)^-gamma)."""
    mesh, props, bc = _ctx(nx=12, ny=8, nz=8)
    # Make a heterogeneous k: halve permeability in half the cells.
    props.k[:, :, :] = props.k0
    props.k[mesh.nx // 2:, :, :] *= 0.25
    from physics.pressure_solver import PressureSolver
    from physics.velocity import VelocityField
    reg = SpeciesRegistry([
        Species("polymer", True, inlet_value=1.0),
        Species("polymer_retained", False, initial_value=0.0)])
    ret = PolymerRetention(reg, props, sigma_max=3e-4, rate_constant=3e-2,
                           perm_dependence_gamma=1.0)
    # Capacity at low-k cells must exceed capacity at high-k cells.
    state = np.zeros((2, mesh.n_active)); state[0, :] = 1.0
    r = ret.rates(state, reg, mesh, props)
    ijk = mesh.dof_to_ijk
    k = props.k[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    k0 = props.k0[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
    lowk = r[1, :][k / k0 < 0.5]
    highk = r[1, :][k / k0 > 0.9]
    assert lowk.mean() > highk.mean(), "low-k cells did not retain more"
    print(f"PASS  permeability-dependent retention: low-k retention rate "
          f"{lowk.mean():.2e} > high-k {highk.mean():.2e}")


def test_shear_degradation():
    """Shear degradation removes dissolved polymer above the shear velocity."""
    mesh, props, bc = _ctx(nx=12, ny=8, nz=8)
    from physics.pressure_solver import PressureSolver
    from physics.velocity import VelocityField
    P = PressureSolver(mesh, props).solve(bc, verbose=False)
    vel = VelocityField(mesh, props); vel.compute(P, bc)
    reg = SpeciesRegistry([
        Species("polymer", True, inlet_value=1.0),
        Species("polymer_retained", False)])
    vmean = float(np.mean(vel.speed[mesh.active] / props.phi[mesh.active]))
    ret = PolymerRetention(reg, props, velocity=vel, sigma_max=0.0,  # no retention
                           rate_constant=0.0, shear_degradation=True,
                           shear_velocity=vmean / 10, k_shear=1e-2)
    state = np.zeros((2, mesh.n_active)); state[0, :] = 1.0
    r = ret.rates(state, reg, mesh, props)
    assert np.all(r[0, :] <= 1e-18), "shear degradation should not increase polymer"
    assert r[0, :].min() < 0, "shear degradation produced no loss above v_sh"
    print(f"PASS  shear degradation: dissolved polymer lost above v_sh "
          f"(min rate {r[0,:].min():.2e})")


# ===========================================================================
#  CALIBRATION FRAMEWORK
# ===========================================================================
def test_calibration_parameter_roundtrip():
    """Parameter vector to/from round-trip (incl. log-scaled) is exact and bounded."""
    p = CalibrationParameters()
    p.add("q_max", 2e-4, 1e-5, 1e-3, log_scale=True)
    p.add("K_L", 5.0, 0.1, 50.0)
    p.add("rate", 5e-2, 1e-3, 1.0, log_scale=True)
    x = p.to_vector()
    p.from_vector(x)
    assert abs(p.get("q_max") - 2e-4) / 2e-4 < 1e-9
    assert abs(p.get("K_L") - 5.0) < 1e-9
    # clipping
    lo, hi = p.bounds_vector()
    p.from_vector(hi + 10.0)
    assert abs(p.get("K_L") - 50.0) < 1e-9, "upper bound not enforced"
    print("PASS  calibration parameters: vector round-trip exact, bounds enforced")


def test_objective_and_sensitivity():
    """Objective is zero for a perfect match; sensitivity ranks an influential
    parameter above a non-influential one."""
    # Synthetic 'observed' = a quadratic in one parameter.
    obs = ObservedData()
    obs.add("signal", times=[0, 1, 2], values=[1.0, 1.0, 1.0])

    # extractor: returns a flat signal equal to param 'a' (so match when a=1).
    class FakeSim:
        def __init__(self, a): self.a = a
    def extractor(sim):
        return np.array([0, 1, 2]), np.full(3, sim.a)
    obj = ObjectiveFunction(obs, {"signal": extractor})
    assert obj.evaluate(FakeSim(1.0))[0] < 1e-12, "objective nonzero at perfect match"
    assert obj.evaluate(FakeSim(2.0))[0] > 0, "objective zero at mismatch"

    # Sensitivity: QoI = a*influential + 0*noninfluential.
    p = CalibrationParameters()
    p.add("influential", 1.0, 0.1, 10.0)
    p.add("noninfluential", 1.0, 0.1, 10.0)
    def qoi(d): return 5.0 * d["influential"] + 0.0 * d["noninfluential"]
    S = SensitivityAnalysis(p, qoi).compute()
    keys = list(S.keys())
    assert keys[0] == "influential", "sensitivity ranking wrong"
    assert abs(S["noninfluential"]) < 1e-9
    print(f"PASS  objective & sensitivity: zero misfit at match; ranking "
          f"identifies influential parameter (S={S['influential']:.2f})")


# ===========================================================================
def _run_all():
    tests = [
        test_isotherm_limits,
        test_adsorption_mass_conservation,
        test_concentration_retardation,
        test_irreversible_adsorption,
        test_salinity_sensitive_adsorption,
        test_ph_dependent_adsorption,
        test_polymer_retention_and_rrf,
        test_permeability_dependent_retention,
        test_shear_degradation,
        test_calibration_parameter_roundtrip,
        test_objective_and_sensitivity,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 5 : adsorption, retention & calibration framework")
    print("=" * 70)
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failures += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print("=" * 70)
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    print("=" * 70)
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
