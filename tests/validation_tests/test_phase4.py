"""
ASPiRe-3D : tests/test_phase4.py
===============================================================================
Validation suite for coupled formation-damage reactive transport (Phase 4).

Covers the engineering-validation requirements:
  * reaction mass conservation (precipitation couple; fines couple),
  * porosity bounds enforcement and permeability positivity,
  * porosity-permeability laws (monotone, correct limits),
  * precipitation triggers (alkali & salinity), reversible dissolution,
  * Arrhenius temperature sensitivity,
  * fines critical-velocity threshold behaviour,
  * full coupling: damage reduces permeability and injectivity monotonically,
  * transport-core invariance (vectorized operator == validated physics, via
    the unchanged Phase 2b/3 suites run separately).
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
from physics.kinetics import (PrecipitationKinetics, FinesMigrationKinetics,
                           CompositeReactionModel)
from physics.formation_damage import (FormationDamage, kozeny_carman, power_law,
                                   exponential_damage)
from physics.coupled_simulator import CoupledSimulator
from utils.constants import ML_PER_MIN_TO_M3_PER_S


def _ctx(nx=24, ny=10, nz=10):
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, 0.038)
    props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    Q = 2.0 * ML_PER_MIN_TO_M3_PER_S
    bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    return mesh, props, bc


# ===========================================================================
#  POROSITY-PERMEABILITY LAWS
# ===========================================================================
def test_perm_laws_limits_and_monotonicity():
    """Each law: k=k0 at phi=phi0, k decreasing as phi decreases, k>0."""
    phi0, k0 = 0.20, 1e-13
    phis = np.linspace(0.20, 0.02, 10)
    for fn, kw in ((kozeny_carman, {}), (power_law, {"exponent": 4.0}),
                   (exponential_damage, {"beta": 8.0})):
        kk = fn(phis, phi0, k0, **kw)
        assert np.isclose(kk[0], k0, rtol=1e-9), f"{fn.__name__}: k(phi0)!=k0"
        assert np.all(np.diff(kk) < 0), f"{fn.__name__}: not monotone"
        assert np.all(kk > 0), f"{fn.__name__}: non-positive k"
    print("PASS  perm laws: k(phi0)=k0, monotone decreasing, positive "
          "(Kozeny-Carman, power-law, exponential)")


# ===========================================================================
#  REACTION MASS CONSERVATION
# ===========================================================================
def test_precipitation_mass_conservation():
    """
    With consume_per_precipitate=1, every unit of precipitate formed must
    remove one unit from the dissolved pool: d(dissolved+precip)/dt summed over
    cells = 0 (no transport, reaction only).
    """
    mesh, props, _ = _ctx()
    reg = SpeciesRegistry([
        Species("salinity", True, inlet_value=1.0, initial_value=1.0),
        Species("alkali",   True, inlet_value=1.0, initial_value=1.0),
        Species("precipitate", False, initial_value=0.0),
    ])
    pk = PrecipitationKinetics(reg, k_precip=1.0, k_dissolve=0.5,
                               alkali_ref=0.5, salinity_ref=0.5,
                               consume_per_precipitate=1.0)
    # Uniform supersaturated state.
    state = np.zeros((3, mesh.n_active))
    state[reg.index("salinity"), :] = 1.0
    state[reg.index("alkali"), :] = 1.0
    rates = pk.rates(state, reg, mesh, props)
    # dissolved + precip rate sum to zero everywhere.
    couple = rates[reg.index("salinity"), :] + rates[reg.index("precipitate"), :]
    assert np.max(np.abs(couple)) < 1e-12, "precip/dissolved not conservative"
    print("PASS  precipitation mass conservation: dissolved loss == precipitate gain")


def test_fines_mass_conservation():
    """Detachment + deposition move mass between suspended and deposited only:
    rate_suspended + rate_deposited = 0 everywhere."""
    mesh, props, bc = _ctx()
    # Build a velocity field.
    from physics.pressure_solver import PressureSolver
    from physics.velocity import VelocityField
    P = PressureSolver(mesh, props).solve(bc, verbose=False)
    vel = VelocityField(mesh, props); vel.compute(P, bc)
    reg = SpeciesRegistry([
        Species("fines_suspended", True, initial_value=0.5),
        Species("fines_deposited", False, initial_value=3.0),
    ])
    fk = FinesMigrationKinetics(reg, vel, props, critical_velocity=1e-6,
                                k_detach=1e-2, k_deposit=1e2)
    state = np.zeros((2, mesh.n_active))
    state[0, :] = 0.5; state[1, :] = 3.0
    rates = fk.rates(state, reg, mesh, props)
    couple = rates[0, :] + rates[1, :]
    assert np.max(np.abs(couple)) < 1e-12, "fines exchange not conservative"
    print("PASS  fines mass conservation: suspended loss == deposited gain")


# ===========================================================================
#  PRECIPITATION TRIGGERS
# ===========================================================================
def test_precipitation_triggers_alkali_and_salinity():
    """
    Precipitation rate must increase with BOTH alkali and salinity (the two
    triggers), and be zero when undersaturated (SI<1).
    """
    mesh, props, _ = _ctx()
    reg = SpeciesRegistry([
        Species("salinity", True), Species("alkali", True),
        Species("precipitate", False)])
    pk = PrecipitationKinetics(reg, k_precip=1.0, k_dissolve=0.0,
                               alkali_ref=0.5, salinity_ref=0.5,
                               consume_per_precipitate=0.0)

    def precip_rate(sal, alk):
        st = np.zeros((3, mesh.n_active))
        st[0, :] = sal; st[1, :] = alk
        return pk.rates(st, reg, mesh, props)[reg.index("precipitate"), :].mean()

    # SI = (alk/0.5)*(sal/0.5). Undersaturated -> zero.
    assert precip_rate(0.2, 0.2) == 0.0, "should not precipitate when SI<1"
    # Increasing alkali increases rate.
    assert precip_rate(1.0, 1.0) > precip_rate(1.0, 0.7) > 0, "alkali trigger weak"
    # Increasing salinity increases rate.
    assert precip_rate(1.0, 1.0) > precip_rate(0.7, 1.0) > 0, "salinity trigger weak"
    print("PASS  precipitation triggers: rate rises with alkali and salinity; "
          "zero when undersaturated")


def test_reversible_dissolution():
    """When undersaturated and precipitate exists, the net rate is negative
    (dissolution), and zero if no precipitate is present."""
    mesh, props, _ = _ctx()
    reg = SpeciesRegistry([
        Species("salinity", True), Species("alkali", True),
        Species("precipitate", False)])
    pk = PrecipitationKinetics(reg, k_precip=1.0, k_dissolve=1.0,
                               alkali_ref=0.5, salinity_ref=0.5,
                               consume_per_precipitate=0.0)
    # Undersaturated (SI<1) with precipitate present -> dissolution (<0).
    st = np.zeros((3, mesh.n_active))
    st[0, :] = 0.2; st[1, :] = 0.2; st[reg.index("precipitate"), :] = 1.0
    r_with = pk.rates(st, reg, mesh, props)[reg.index("precipitate"), :].mean()
    assert r_with < 0, "should dissolve when undersaturated with precipitate"
    # No precipitate -> no dissolution.
    st[reg.index("precipitate"), :] = 0.0
    r_without = pk.rates(st, reg, mesh, props)[reg.index("precipitate"), :].mean()
    assert abs(r_without) < 1e-9, "dissolution should vanish with no precipitate"
    print(f"PASS  reversible dissolution: rate {r_with:.3e} (<0) with precip, "
          f"~0 without")


def test_arrhenius_temperature_sensitivity():
    """Higher temperature must accelerate the reaction (Arrhenius, Ea>0)."""
    mesh, props, _ = _ctx()
    reg = SpeciesRegistry([
        Species("salinity", True), Species("alkali", True),
        Species("precipitate", False)])

    def rate_at_T(T):
        pk = PrecipitationKinetics(reg, k_precip=1.0, k_dissolve=0.0,
                                   alkali_ref=0.5, salinity_ref=0.5,
                                   consume_per_precipitate=0.0,
                                   activation_energy=50000.0,
                                   temperature=T, temperature_ref=298.15)
        st = np.zeros((3, mesh.n_active)); st[0, :] = 1.0; st[1, :] = 1.0
        return pk.rates(st, reg, mesh, props)[reg.index("precipitate"), :].mean()

    r_cold, r_hot = rate_at_T(298.15), rate_at_T(348.15)
    assert r_hot > r_cold > 0, f"Arrhenius failed: cold {r_cold}, hot {r_hot}"
    print(f"PASS  Arrhenius: rate increases with T "
          f"({r_hot/r_cold:.2f}x faster at +50 K)")


# ===========================================================================
#  FINES CRITICAL VELOCITY
# ===========================================================================
def test_fines_critical_velocity_threshold():
    """No detachment below the critical velocity; detachment turns on above it."""
    mesh, props, bc = _ctx()
    from physics.pressure_solver import PressureSolver
    from physics.velocity import VelocityField
    P = PressureSolver(mesh, props).solve(bc, verbose=False)
    vel = VelocityField(mesh, props); vel.compute(P, bc)
    reg = SpeciesRegistry([
        Species("fines_suspended", True, initial_value=0.0),
        Species("fines_deposited", False, initial_value=1.0)])

    vmean = float(np.mean(vel.speed[mesh.active] / props.phi[mesh.active]))
    # v_c well ABOVE actual velocity -> no detachment.
    fk_hi = FinesMigrationKinetics(reg, vel, props, critical_velocity=vmean * 100,
                                   k_detach=1.0, k_deposit=0.0)
    st = np.zeros((2, mesh.n_active)); st[1, :] = 1.0
    det_hi = fk_hi.rates(st, reg, mesh, props)[0, :].max()
    # v_c well BELOW actual velocity -> detachment occurs.
    fk_lo = FinesMigrationKinetics(reg, vel, props, critical_velocity=vmean / 100,
                                   k_detach=1.0, k_deposit=0.0)
    det_lo = fk_lo.rates(st, reg, mesh, props)[0, :].max()
    assert det_hi == 0.0, "detachment occurred below critical velocity"
    assert det_lo > 0.0, "no detachment above critical velocity"
    print(f"PASS  fines critical velocity: no detachment below v_c, "
          f"detachment ({det_lo:.2e}) above")


# ===========================================================================
#  FULL COUPLING + BOUNDS
# ===========================================================================
def test_coupled_damage_monotonic_and_bounded():
    """
    Full coupled run: permeability ratio and injectivity must decline
    monotonically; porosity must stay >= phi_min and k > 0 throughout.
    """
    mesh, props, bc = _ctx()
    reg = SpeciesRegistry([
        Species("salinity", True, inlet_value=1.0, initial_value=1.0, molecular_diffusion=1e-9),
        Species("alkali",   True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9),
        Species("precipitate", False, initial_value=0.0),
    ])
    pk = PrecipitationKinetics(reg, k_precip=0.05, k_dissolve=0.0,
                               alkali_ref=0.5, salinity_ref=0.5,
                               consume_per_precipitate=0.0)
    dmg = FormationDamage(mesh, props, reg, perm_model="kozeny_carman",
                          phi_min=0.01)
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=pk,
                           formation_damage=dmg, reflow_k_tolerance=0.05)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=1.0 * dx / vmax, n_pore_volumes=2.0, verbose=False)

    kmin = np.array(sim.history["k_ratio_min"])
    inj = np.array(sim.history["injectivity_ratio"])
    phim = np.array(sim.history["phi_min"])
    # Monotone non-increasing (allow tiny numerical noise).
    assert np.all(np.diff(kmin) <= 1e-9), "permeability ratio not monotone"
    assert np.all(np.diff(inj) <= 1e-9), "injectivity not monotone-declining"
    # Bounds.
    assert phim.min() >= 0.01 - 1e-9, "porosity below floor"
    assert np.all(props.k[mesh.active] > 0), "permeability non-positive"
    print(f"PASS  coupled damage: k/k0 {kmin[0]:.3f}->{kmin[-1]:.3f}, "
          f"injectivity {inj[0]:.3f}->{inj[-1]:.3f}, "
          f"phi_min>={phim.min():.3f}, k>0 (monotone & bounded)")


def test_no_damage_recovers_conservative_transport():
    """
    With NullReactionModel and no FormationDamage, the coupled simulator must
    reduce to pure conservative transport: permeability ratio stays 1, porosity
    unchanged, salinity displacement conserved (injected+resident=1).
    """
    mesh, props, bc = _ctx()
    reg = SpeciesRegistry([
        Species("injected", True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9),
        Species("resident", True, inlet_value=0.0, initial_value=1.0, molecular_diffusion=1e-9),
    ])
    sim = CoupledSimulator(mesh, props, reg, bc, reaction_model=None,
                           formation_damage=None)
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(sim.velocity.speed[mesh.active] / props.phi[mesh.active]))
    sim.run(dt=1.0 * dx / vmax, n_pore_volumes=0.6, verbose=False)
    total = sim.C[0, :] + sim.C[1, :]
    assert np.max(np.abs(total - 1.0)) < 1e-9, "displacement not conserved"
    assert np.all(np.array(sim.history["k_ratio_min"]) == 1.0), "k changed w/o damage"
    print("PASS  no-damage limit: reduces to conservative transport "
          "(k/k0=1, injected+resident=1)")


# ===========================================================================
def _run_all():
    tests = [
        test_perm_laws_limits_and_monotonicity,
        test_precipitation_mass_conservation,
        test_fines_mass_conservation,
        test_precipitation_triggers_alkali_and_salinity,
        test_reversible_dissolution,
        test_arrhenius_temperature_sensitivity,
        test_fines_critical_velocity_threshold,
        test_coupled_damage_monotonic_and_bounded,
        test_no_damage_recovers_conservative_transport,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 4 : coupled formation-damage reactive transport")
    print("=" * 70)
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL  {t.__name__}: {e}")
    print("=" * 70)
    print(f"{len(tests) - failures}/{len(tests)} tests passed")
    print("=" * 70)
    return failures


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
