"""
ASPiRe-3D : tests/test_phase2.py
===============================================================================
Physics-based validation suite for Phase 2 passive tracer transport.

As in Phase 1, every test asserts a physical/mathematical invariant of the
advection-dispersion solver. Together they certify the transport core before
any reactive chemistry is layered on top.
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
from physics.transport import TracerTransport
from utils.constants import ML_PER_MIN_TO_M3_PER_S


def _build_flow(nx=30, ny=12, nz=12, alpha_L=1.0e-3, D_m=1.0e-9):
    """Build a converged Phase 1 flow field + a transport solver on top."""
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, 0.038)
    props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
    Q = 2.0 * ML_PER_MIN_TO_M3_PER_S
    bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    P = PressureSolver(mesh, props).solve(bc, verbose=False)
    vel = VelocityField(mesh, props); vel.compute(P, bc)
    tr = TracerTransport(mesh, props, vel,
                         longitudinal_dispersivity=alpha_L,
                         transverse_dispersivity=alpha_L / 10,
                         molecular_diffusion=D_m, cfl=0.5)
    return mesh, props, vel, tr


# ===========================================================================
def test_cfl_number_respected():
    """The Courant number must stay <= the configured CFL target every step."""
    mesh, props, vel, tr = _build_flow()
    tr.run(n_pore_volumes=0.5, c_inject=1.0, verbose=False)
    courants = np.array(tr.history["courant"])
    assert np.all(courants <= tr.cfl + 1e-9), \
        f"Courant exceeded CFL: max={courants.max():.3f}"
    print(f"PASS  CFL control: max Courant {courants.max():.3f} <= {tr.cfl}")


def test_diffusion_number_respected():
    """The diffusion (von Neumann) number must stay within the stable bound."""
    mesh, props, vel, tr = _build_flow(D_m=1.0e-7)   # diffusion-heavier case
    tr.run(n_pore_volumes=0.3, c_inject=1.0, verbose=False)
    dnum = np.array(tr.history["diffusion_number"])
    # Stable explicit diffusion requires the von Neumann number <= 1.
    assert np.all(dnum <= 1.0 + 1e-9), f"diffusion number exceeded 1: {dnum.max():.3f}"
    print(f"PASS  diffusion stability: max diffusion# {dnum.max():.3f} <= 1")


def test_monotonicity_boundedness():
    """
    First-order upwinding is monotone: tracer concentration must remain within
    [0, C_inj] for all cells and all times (no over/undershoot, no negatives).
    """
    mesh, props, vel, tr = _build_flow()
    C_inj = 1.0
    tr.run(n_pore_volumes=0.8, c_inject=C_inj, verbose=False)
    Cvals = tr.C[mesh.active]
    assert Cvals.min() >= -1e-9, f"negative concentration {Cvals.min():.2e}"
    assert Cvals.max() <= C_inj + 1e-9, f"overshoot {Cvals.max():.4f} > {C_inj}"
    print(f"PASS  monotonicity: C in [{Cvals.min():.2e}, {Cvals.max():.4f}] "
          f"within [0, {C_inj}]")


def test_global_mass_conservation():
    """
    Discrete tracer mass balance: the change in total tracer mass over a step
    interval must equal (injected - produced) tracer mass to round-off.

    We track cumulative inlet and outlet tracer mass by integrating their
    fluxes, and compare to the stored field mass.
    """
    mesh, props, vel, tr = _build_flow()
    C_inj = 1.0
    pore_volume = float(np.sum(props.phi[mesh.active]) * mesh.cell_volume)
    Q_total = sum(vel.ux[0, j, k] * mesh.area_x
                  for j in range(mesh.ny) for k in range(mesh.nz)
                  if mesh.active[0, j, k])

    # March, accumulating injected and produced tracer mass from fluxes.
    tr.run(n_pore_volumes=0.6, c_inject=C_inj, verbose=False)

    # Injected mass = C_inj * Q_total * elapsed_time (Dirichlet inlet).
    injected = C_inj * Q_total * tr.time
    # Produced mass = integral of outlet flux * outlet C over time.
    dt = np.array(tr.history["dt"])
    outC = np.array(tr.history["outlet_C"])
    produced = float(np.sum(Q_total * outC * dt))
    stored = float(np.sum(props.phi[mesh.active] * mesh.cell_volume
                          * tr.C[mesh.active]))

    residual = injected - produced - stored
    rel = abs(residual) / max(injected, 1e-30)
    # Tolerance is loose vs pressure (explicit time integration + outlet
    # flux-weighting introduce small O(dt) accounting error), but must be small.
    assert rel < 5.0e-3, f"tracer mass imbalance rel={rel:.3e}"
    print(f"PASS  tracer mass balance: injected={injected:.3e}, "
          f"stored+produced={stored+produced:.3e}, rel={rel:.3e}")


def test_breakthrough_near_one_pore_volume():
    """
    For a passive tracer with modest dispersion, the outlet breakthrough
    (where outlet C crosses ~0.5 C_inj) must occur near 1 pore volume injected
    -- the defining signature of correct mean transport speed.
    """
    mesh, props, vel, tr = _build_flow(alpha_L=5.0e-4, D_m=1.0e-9)
    C_inj = 1.0
    pore_volume = float(np.sum(props.phi[mesh.active]) * mesh.cell_volume)
    Q_total = sum(vel.ux[0, j, k] * mesh.area_x
                  for j in range(mesh.ny) for k in range(mesh.nz)
                  if mesh.active[0, j, k])
    tr.run(n_pore_volumes=2.0, c_inject=C_inj, verbose=False)

    t = np.array(tr.history["time"])
    outC = np.array(tr.history["outlet_C"])
    pv = t * Q_total / pore_volume
    # Find first PV where outlet C >= 0.5.
    idx = np.argmax(outC >= 0.5 * C_inj)
    pv_bt = pv[idx]
    # Upwind numerical diffusion shifts/spreads the front; accept a tolerant
    # window around 1 PV (true mean residence time).
    assert 0.6 <= pv_bt <= 1.4, f"breakthrough at {pv_bt:.2f} PV (expected ~1)"
    print(f"PASS  breakthrough: 0.5*C_inj at {pv_bt:.2f} PV (expected ~1.0)")


def test_pure_advection_front_speed():
    """
    With dispersion suppressed, the tracer front should advance at the mean
    interstitial velocity v = Q/(phi*A). Check the front midpoint position
    after a known time matches x = v*t within numerical-diffusion tolerance.
    """
    mesh, props, vel, tr = _build_flow(alpha_L=1.0e-6, D_m=1.0e-12)
    C_inj = 1.0
    # Mean interstitial velocity along the core.
    area = sum(1 for j in range(mesh.ny) for k in range(mesh.nz)
               if mesh.active[0, j, k]) * mesh.area_x
    Q_total = sum(vel.ux[0, j, k] * mesh.area_x
                  for j in range(mesh.ny) for k in range(mesh.nz)
                  if mesh.active[0, j, k])
    phi_mean = props.phi[mesh.active].mean()
    v_mean = Q_total / (phi_mean * area)

    # March to a time where the front sits mid-core.
    t_target = 0.5 * mesh.Lx / v_mean
    tr.run(total_time=t_target, c_inject=C_inj, verbose=False)

    # Front position = axial location where slice-averaged C crosses 0.5.
    p_axial = np.array([np.nanmean(tr.C[i, :, :][mesh.active[i, :, :]])
                        for i in range(mesh.nx)])
    cross = np.argmax(p_axial <= 0.5 * C_inj)
    x_front = mesh.xc[cross]
    x_expected = v_mean * t_target
    rel = abs(x_front - x_expected) / mesh.Lx
    assert rel < 0.15, f"front at {x_front:.3f} m vs expected {x_expected:.3f} m"
    print(f"PASS  advection speed: front at {x_front:.4f} m vs "
          f"v*t={x_expected:.4f} m (within {rel*100:.1f}% of L)")


# ===========================================================================
def _run_all():
    tests = [
        test_cfl_number_respected,
        test_diffusion_number_respected,
        test_monotonicity_boundedness,
        test_global_mass_conservation,
        test_breakthrough_near_one_pore_volume,
        test_pure_advection_front_speed,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 2 transport validation suite")
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
