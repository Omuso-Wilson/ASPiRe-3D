"""
ASPiRe-3D : tests/test_phase3.py
===============================================================================
Validation suite for the multi-species reactive transport BACKBONE.

With the NullReactionModel the backbone is PURE CONSERVATIVE multi-species
transport, so every invariant from the single-tracer solver must hold for each
species, plus new multi-species invariants:
  * independence: each species transports as if alone (no cross-talk without a
    reaction model),
  * equivalence: a single backbone species reproduces the standalone implicit
    tracer exactly,
  * immobility: immobile species never move under transport,
  * complementary displacement: a displaced resident + injected slug conserve
    total (resident + slug) mass,
  * conservation & boundedness per species,
  * reaction-hook sanity: a trivial decay reaction model changes mass in the
    expected direction (interface works) without breaking transport.
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
from physics.pressure_solver import PressureSolver
from physics.velocity import VelocityField
from physics.transport import ImplicitTracerTransport
from physics.species import (Species, SpeciesRegistry, NullReactionModel,
                          ReactionModel)
from physics.reactive_transport import ReactiveTransport
from utils.constants import ML_PER_MIN_TO_M3_PER_S


def _flow(nx=30, ny=12, nz=12):
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, 0.038)
    props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    Q = 2.0 * ML_PER_MIN_TO_M3_PER_S
    bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    P = PressureSolver(mesh, props).solve(bc, verbose=False)
    vel = VelocityField(mesh, props); vel.compute(P, bc)
    return mesh, props, vel


def _dt_for_courant(mesh, props, vel, courant):
    dx = min(mesh.dx, mesh.dy, mesh.dz)
    vmax = float(np.max(vel.speed[mesh.active] /
                        np.maximum(props.phi[mesh.active], 1e-6)))
    return courant * dx / vmax


# ===========================================================================
def test_single_species_matches_standalone():
    """
    A backbone with ONE mobile species (Null reactions) must reproduce the
    standalone ImplicitTracerTransport bit-for-bit (same operator, same step).
    This proves the multi-species generalisation did not alter the validated
    single-species physics.
    """
    mesh, props, vel = _flow()
    aL = 5e-4; Dm = 1e-9
    dt = _dt_for_courant(mesh, props, vel, 2.0)

    # Standalone.
    solo = ImplicitTracerTransport(mesh, props, vel,
                                   longitudinal_dispersivity=aL,
                                   molecular_diffusion=Dm, solver="direct")
    solo.run(dt=dt, n_pore_volumes=1.0, c_inject=1.0, verbose=False)

    # Backbone with one species.
    reg = SpeciesRegistry([Species("tracer", mobile=True, inlet_value=1.0,
                                   molecular_diffusion=Dm, initial_value=0.0)])
    multi = ReactiveTransport(mesh, props, vel, reg,
                              reaction_model=NullReactionModel(),
                              longitudinal_dispersivity=aL)
    multi.run(dt=dt, n_pore_volumes=1.0, verbose=False)

    diff = np.max(np.abs(solo.c - multi.C[0, :]))
    assert diff < 1e-10, f"backbone diverges from standalone: {diff:.2e}"
    print(f"PASS  single-species equivalence: max |Δ| {diff:.2e} "
          f"(backbone == standalone)")


def test_species_independence():
    """
    Without reactions, species must not influence each other. Running two
    species together must give the same result as running each alone.
    """
    mesh, props, vel = _flow()
    aL = 5e-4
    dt = _dt_for_courant(mesh, props, vel, 2.0)

    sA = Species("A", mobile=True, inlet_value=1.0, molecular_diffusion=1e-9)
    sB = Species("B", mobile=True, inlet_value=0.5, molecular_diffusion=1e-10,
                 initial_value=0.3)

    # Together.
    reg_both = SpeciesRegistry([Species("A", True, 1.0, 1e-9, 0.0),
                                Species("B", True, 0.5, 1e-10, 0.3)])
    both = ReactiveTransport(mesh, props, vel, reg_both,
                             longitudinal_dispersivity=aL)
    both.run(dt=dt, n_pore_volumes=0.8, verbose=False)

    # A alone.
    rA = ReactiveTransport(mesh, props, vel,
                           SpeciesRegistry([Species("A", True, 1.0, 1e-9, 0.0)]),
                           longitudinal_dispersivity=aL)
    rA.run(dt=dt, n_pore_volumes=0.8, verbose=False)
    # B alone.
    rB = ReactiveTransport(mesh, props, vel,
                           SpeciesRegistry([Species("B", True, 0.5, 1e-10, 0.3)]),
                           longitudinal_dispersivity=aL)
    rB.run(dt=dt, n_pore_volumes=0.8, verbose=False)

    dA = np.max(np.abs(both.C[0, :] - rA.C[0, :]))
    dB = np.max(np.abs(both.C[1, :] - rB.C[0, :]))
    assert dA < 1e-10 and dB < 1e-10, f"species coupled without reactions: {dA:.2e},{dB:.2e}"
    print(f"PASS  species independence: |ΔA|={dA:.2e}, |ΔB|={dB:.2e} "
          f"(no spurious cross-talk)")


def test_immobile_species_does_not_move():
    """
    An immobile species (precipitate) must be untouched by transport when no
    reactions act on it: its field stays exactly at its initial value.
    """
    mesh, props, vel = _flow()
    reg = SpeciesRegistry([
        Species("mobile_one", mobile=True, inlet_value=1.0, molecular_diffusion=1e-9),
        Species("precipitate", mobile=False, inlet_value=0.0, initial_value=0.42),
    ])
    rt = ReactiveTransport(mesh, props, vel, reg, longitudinal_dispersivity=5e-4)
    rt.run(dt=_dt_for_courant(mesh, props, vel, 3.0),
           n_pore_volumes=1.5, verbose=False)
    precip = rt.C[reg.index("precipitate"), :]
    assert np.allclose(precip, 0.42, atol=1e-12), \
        f"immobile species moved: range [{precip.min()},{precip.max()}]"
    print("PASS  immobile species: precipitate unchanged by transport (stays 0.42)")


def test_complementary_displacement_conservation():
    """
    Inject a slug (inlet=1, init=0) while a resident species (inlet=0, init=1)
    is displaced. With no reactions, at every cell the two should sum to ~1
    (complementary displacement) once dispersion is modest -- a strong
    multi-species conservation signature.
    """
    mesh, props, vel = _flow()
    reg = SpeciesRegistry([
        Species("injected", mobile=True, inlet_value=1.0, initial_value=0.0,
                molecular_diffusion=1e-9),
        Species("resident", mobile=True, inlet_value=0.0, initial_value=1.0,
                molecular_diffusion=1e-9),
    ])
    rt = ReactiveTransport(mesh, props, vel, reg, longitudinal_dispersivity=5e-4)
    rt.run(dt=_dt_for_courant(mesh, props, vel, 1.0),
           n_pore_volumes=0.6, verbose=False)
    total = rt.C[0, :] + rt.C[1, :]
    err = np.max(np.abs(total - 1.0))
    # Identical dispersion => identical operator => exact complementarity.
    assert err < 1e-9, f"injected+resident != 1 (max dev {err:.2e})"
    print(f"PASS  complementary displacement: injected+resident = 1 "
          f"everywhere (max dev {err:.2e})")


def test_per_species_boundedness():
    """Each species must stay within [min(init,inlet), max(init,inlet)]."""
    mesh, props, vel = _flow()
    reg = SpeciesRegistry([
        Species("salinity", True, inlet_value=0.1, initial_value=1.0, molecular_diffusion=1e-9),
        Species("alkali",   True, inlet_value=1.0, initial_value=0.0, molecular_diffusion=1e-9),
    ])
    rt = ReactiveTransport(mesh, props, vel, reg, longitudinal_dispersivity=5e-4)
    rt.run(dt=_dt_for_courant(mesh, props, vel, 4.0),
           n_pore_volumes=1.5, verbose=False)
    sal = rt.C[0, :]; alk = rt.C[1, :]
    assert sal.min() >= 0.1 - 1e-6 and sal.max() <= 1.0 + 1e-6, "salinity unbounded"
    assert alk.min() >= -1e-6 and alk.max() <= 1.0 + 1e-6, "alkali unbounded"
    print(f"PASS  per-species boundedness: salinity in [{sal.min():.3f},{sal.max():.3f}], "
          f"alkali in [{alk.min():.3f},{alk.max():.3f}]")


def test_reaction_hook_interface():
    """
    The reaction interface must actually be applied. Use a simple first-order
    decay reaction on one species and confirm its mass drops monotonically
    relative to the no-reaction case, WITHOUT breaking transport stability.
    """
    class DecayModel(ReactionModel):
        """dC/dt = -lambda * C for species 'decayer'; zero for others."""
        def __init__(self, registry, rate):
            self.idx = registry.index("decayer")
            self.rate = rate
        def rates(self, state, registry, mesh, properties):
            r = np.zeros_like(state)
            r[self.idx, :] = -self.rate * state[self.idx, :]
            return r

    mesh, props, vel = _flow()
    dt = _dt_for_courant(mesh, props, vel, 1.0)

    def final_mass(reaction):
        reg = SpeciesRegistry([Species("decayer", True, inlet_value=1.0,
                                       molecular_diffusion=1e-9)])
        rmodel = reaction(reg) if reaction else NullReactionModel()
        rt = ReactiveTransport(mesh, props, vel, reg, reaction_model=rmodel,
                               longitudinal_dispersivity=5e-4)
        rt.run(dt=dt, n_pore_volumes=1.0, verbose=False)
        return float(np.sum(rt.storage * rt.C[0, :])), rt

    m_null, _ = final_mass(None)
    m_decay, rt_d = final_mass(lambda reg: DecayModel(reg, rate=1e-3))

    assert m_decay < m_null, "decay reaction did not reduce mass"
    # Transport still stable/bounded under reactions.
    assert rt_d.C[0, :].min() >= -1e-9 and rt_d.C[0, :].max() <= 1.0 + 1e-6
    print(f"PASS  reaction hook: decay mass {m_decay:.3e} < null mass {m_null:.3e} "
          f"(interface applies, transport stays bounded)")


# ===========================================================================
def _run_all():
    tests = [
        test_single_species_matches_standalone,
        test_species_independence,
        test_immobile_species_does_not_move,
        test_complementary_displacement_conservation,
        test_per_species_boundedness,
        test_reaction_hook_interface,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 3 : multi-species reactive transport backbone")
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
