"""
ASPiRe-3D : tests/test_benchmarks.py
===============================================================================
Publication-grade BENCHMARKS for the coupled simulator. Where Phase 1-4 unit
tests assert correctness of pieces, these assert that the ASSEMBLED simulator
reproduces recognised quantitative behaviour expected in the literature:

  B1. Conservative transport baseline: with no reactions/damage, multi-species
      breakthrough matches the validated passive-tracer behaviour (0.5 crossing
      near 1 PV).
  B2. Analytical advection-dispersion: the simulated 1D breakthrough matches the
      analytical Ogata-Banks solution within the upwind numerical-diffusion
      tolerance.
  B3. Permeability impairment trend: under sustained precipitation, k/k0 vs
      porosity follows the prescribed Kozeny-Carman law (mechanistic check
      against the relationship used throughout the formation-damage literature).
  B4. ASP slug propagation: a finite ASP slug followed by chase brine produces
      a travelling concentration PULSE that arrives at ~1 PV and then clears,
      i.e. the slug is advected at the interstitial velocity.
===============================================================================
"""

import os
import sys
import math
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from physics.mesh import StructuredMesh
from physics.geometry import apply_cylindrical_core
from physics.properties import FluidRockProperties
from physics.boundary_conditions import CoreFloodBC, BCMode
from physics.species import Species, SpeciesRegistry
from physics.kinetics import PrecipitationKinetics
from physics.formation_damage import FormationDamage, kozeny_carman
from physics.coupled_simulator import CoupledSimulator
from utils.constants import ML_PER_MIN_TO_M3_PER_S


def _ctx(nx=40, ny=10, nz=10, rate=2.0):
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, 0.038)
    props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    Q = rate * ML_PER_MIN_TO_M3_PER_S
    bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    return mesh, props, bc


def _pv_axis(sim, mesh, props):
    pv0 = float(np.sum(props.phi0[mesh.active]) * mesh.cell_volume)
    Q = float(np.sum(sim._inlet_flux))
    return np.array(sim.history["time"]) * Q / pv0


# ===========================================================================
def test_B1_conservative_baseline():
    """Multi-species, no reaction/damage: 0.5 crossing of injected near 1 PV."""
    mesh, props, bc = _ctx()
    reg = SpeciesRegistry([
        Species("injected", True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9)])
    sim = CoupledSimulator(mesh, props, reg, bc,
                           reaction_model=None, formation_damage=None,
                           longitudinal_dispersivity=3e-4)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=0.5 * dx / vmax, n_pore_volumes=2.0, verbose=False)
    pv = _pv_axis(sim, mesh, props)
    out = np.array(sim.history["outlet"]["injected"])
    pv_bt = pv[np.argmax(out >= 0.5)]
    assert 0.7 <= pv_bt <= 1.3, f"baseline breakthrough at {pv_bt:.2f} PV"
    print(f"PASS  B1 conservative baseline: 0.5 crossing at {pv_bt:.2f} PV (~1 expected)")


def test_B2_analytical_advection_dispersion():
    """
    Compare simulated breakthrough to the analytical Ogata-Banks (1961) solution
    for 1D advection-dispersion with a step inlet:

        C/C0 = 0.5 * erfc( (L - v t) / (2 sqrt(D t)) )

    evaluated at the outlet (x=L), using the effective dispersion the simulator
    actually uses. We compare the breakthrough TIME at C/C0=0.5, which both
    place at t = L/v (Ogata-Banks midpoint), tolerant of upwind diffusion.
    """
    mesh, props, bc = _ctx(nx=60)         # finer axial grid -> less num. diffusion
    aL = 5e-4
    reg = SpeciesRegistry([
        Species("tracer", True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9)])
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=None,
                           formation_damage=None, longitudinal_dispersivity=aL)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=0.4 * dx / vmax, n_pore_volumes=2.0, verbose=False)

    pv = _pv_axis(sim, mesh, props)
    out = np.array(sim.history["outlet"]["tracer"])
    t = np.array(sim.history["time"])

    # Mean interstitial velocity and outlet breakthrough time.
    area = sum(1 for j in range(mesh.ny) for k in range(mesh.nz)
               if mesh.active[0, j, k]) * mesh.area_x
    Q = float(np.sum(sim._inlet_flux))
    v = Q / (props.phi[mesh.active].mean() * area)
    t_sim = t[np.argmax(out >= 0.5)]
    t_analytic = mesh.Lx / v               # Ogata-Banks midpoint
    rel = abs(t_sim - t_analytic) / t_analytic
    assert rel < 0.15, f"breakthrough time off analytic by {rel:.2f}"
    print(f"PASS  B2 Ogata-Banks: t(0.5) sim {t_sim:.1f}s vs analytic {t_analytic:.1f}s "
          f"({rel*100:.1f}%)")


def test_B3_kozeny_carman_trend():
    """
    Under precipitation, the simulator's per-cell (k/k0, phi) pairs must lie on
    the Kozeny-Carman curve it was configured with -- i.e. the damage coupling
    faithfully applies the literature relationship.
    """
    mesh, props, bc = _ctx(nx=30)
    reg = SpeciesRegistry([
        Species("salinity", True, inlet_value=1.0, initial_value=1.0, molecular_diffusion=1e-9),
        Species("alkali",   True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9),
        Species("precipitate", False, initial_value=0.0)])
    pk = PrecipitationKinetics(reg, k_precip=0.05, k_dissolve=0.0,
                               alkali_ref=0.5, salinity_ref=0.5,
                               consume_per_precipitate=0.0)
    dmg = FormationDamage(mesh, props, reg, perm_model="kozeny_carman")
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=pk,
                           formation_damage=dmg, reflow_k_tolerance=0.05)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=1.0 * dx / vmax, n_pore_volumes=1.5, verbose=False)

    # Check current k vs Kozeny-Carman(phi) per active cell.
    act = mesh.active
    phi = props.phi[act]; k = props.k[act]
    phi0 = props.phi0[act]; k0 = props.k0[act]
    k_expected = kozeny_carman(phi, phi0, k0)
    rel = np.max(np.abs(k - k_expected) / (k0))
    assert rel < 1e-6, f"k deviates from Kozeny-Carman by {rel:.2e}"
    print(f"PASS  B3 Kozeny-Carman trend: simulator k matches KC(phi) "
          f"to {rel:.1e} (relative)")


def test_B4_asp_slug_propagation():
    """
    Inject a FINITE ASP slug (alkali=1 for ~0.5 PV) then chase with brine
    (alkali=0). The outlet alkali should show a PULSE that rises after the slug
    is injected and then falls -- confirming the slug is advected, not smeared
    into a permanent plateau.
    """
    mesh, props, bc = _ctx(nx=40)
    reg = SpeciesRegistry([
        Species("alkali", True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9)])
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=None,
                           formation_damage=None, longitudinal_dispersivity=3e-4)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    dt = 0.5 * dx / vmax

    pv0 = float(np.sum(props.phi0[mesh.active]) * mesh.cell_volume)
    Q = float(np.sum(sim._inlet_flux))
    slug_pv = 0.5
    # Phase A: inject slug for slug_pv pore volumes.
    sim.run(dt=dt, n_pore_volumes=slug_pv, verbose=False)
    # Phase B: switch inlet to chase brine (alkali 0) and continue.
    reg.get("alkali").inlet_value = 0.0
    sim.run(dt=dt, n_pore_volumes=2.5, verbose=False)

    pv = _pv_axis(sim, mesh, props)
    out = np.array(sim.history["outlet"]["alkali"])
    peak_idx = int(np.argmax(out))
    peak_pv = pv[peak_idx]
    # Pulse must rise then fall: peak is interior, and tail returns toward 0.
    assert 0 < peak_idx < len(out) - 1, "no interior peak (not a pulse)"
    assert out[-1] < 0.5 * out[peak_idx], "slug did not clear (no pulse decay)"
    assert 0.5 <= peak_pv <= 1.7, f"slug peak at {peak_pv:.2f} PV (expected ~1)"
    print(f"PASS  B4 ASP slug propagation: pulse peaks at {peak_pv:.2f} PV "
          f"(peak {out[peak_idx]:.2f}) and clears to {out[-1]:.2f}")


# ===========================================================================
def _run_all():
    tests = [
        test_B1_conservative_baseline,
        test_B2_analytical_advection_dispersion,
        test_B3_kozeny_carman_trend,
        test_B4_asp_slug_propagation,
        test_B5_linear_adsorption_retardation_factor,
        test_B6_surfactant_breakthrough_delay_trend,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Benchmarks : publication-grade validation")
    print("=" * 70)
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print("=" * 70)
    print(f"{len(tests) - failures}/{len(tests)} benchmarks passed")
    print("=" * 70)
    return failures



# ===========================================================================
#  PHASE 5 ADSORPTION BENCHMARKS (appended)
# ===========================================================================
def test_B5_linear_adsorption_retardation_factor():
    """
    Analytical benchmark: for a LINEAR adsorption isotherm q = Kd * C (the
    low-concentration limit of Langmuir, q_eq = q_max*K_L*C for K_L*C<<1), the
    breakthrough is retarded by the analytical retardation factor

        R = 1 + (rho_b / phi) * Kd_volumetric

    where, with our per-bulk adsorbed units, the effective volumetric partition
    is d(q_bulk)/d(C_fluid). We verify the simulated 0.5-breakthrough PV matches
    R within numerical-diffusion tolerance.
    """
    from physics.adsorption import SurfactantAdsorption
    mesh, props, bc = _ctx(nx=60)
    phi = props.phi[mesh.active].mean()

    # Use a small q_max*K_L so Langmuir ~ linear at C up to 1.
    q_max, K_L, rho_b = 1e-4, 0.2, 2000.0
    reg = SpeciesRegistry([
        Species("surfactant", True, inlet_value=1.0, molecular_diffusion=5e-10),
        Species("surfactant_adsorbed", False, initial_value=0.0)])
    ads = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=q_max,
                               K_L=K_L, rate_constant=1.0, bulk_density=rho_b)
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=ads,
                           formation_damage=None, longitudinal_dispersivity=3e-4)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=0.4 * dx / vmax, n_pore_volumes=4.0, verbose=False)

    pv = np.array(sim.history["pv"]); out = np.array(sim.history["outlet"]["surfactant"])
    pv_bt = pv[np.argmax(out >= 0.5)]

    # Analytical R for linear regime: slope dq_bulk/dC at C->0 is q_max*K_L*rho_b
    # [kg/m3 bulk per (kg/m3 fluid)]; volumetric partition = slope/phi added to 1.
    Kd_bulk = q_max * K_L * rho_b
    R = 1.0 + Kd_bulk / phi
    rel = abs(pv_bt - R) / R
    assert rel < 0.20, f"retardation factor off: sim {pv_bt:.2f} vs analytic R {R:.2f}"
    print(f"PASS  B5 linear retardation factor: breakthrough {pv_bt:.2f} PV vs "
          f"analytic R={R:.2f} ({rel*100:.0f}%)")


def test_B6_surfactant_breakthrough_delay_trend():
    """
    ASP literature trend: stronger adsorption (higher q_max) must monotonically
    increase the breakthrough delay.
    """
    from physics.adsorption import SurfactantAdsorption
    mesh, props_base, bc = _ctx(nx=40)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    bt = []
    for q_max in (5e-5, 1e-4, 2e-4):
        props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
        reg = SpeciesRegistry([
            Species("surfactant", True, inlet_value=1.0, molecular_diffusion=5e-10),
            Species("surfactant_adsorbed", False, initial_value=0.0)])
        ads = SurfactantAdsorption(reg, props, isotherm="langmuir", q_max=q_max,
                                   K_L=5.0, rate_constant=5e-2, bulk_density=2000.0)
        sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=ads,
                               formation_damage=None, longitudinal_dispersivity=3e-4)
        vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
        sim.run(dt=0.5 * dx / vmax, n_pore_volumes=5.0, verbose=False)
        pv = np.array(sim.history["pv"]); out = np.array(sim.history["outlet"]["surfactant"])
        idx = np.argmax(out >= 0.5)
        bt.append(pv[idx] if out[idx] >= 0.5 else 5.0)
    assert bt[0] < bt[1] < bt[2], f"delay not monotone in q_max: {bt}"
    print(f"PASS  B6 surfactant breakthrough delay: increases with adsorption "
          f"capacity {[round(b,2) for b in bt]} PV")


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
