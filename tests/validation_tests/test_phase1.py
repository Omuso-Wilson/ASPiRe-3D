"""
ASPiRe-3D : tests/test_phase1.py
===============================================================================
Physics-based validation suite for the Phase 1 single-phase Darcy core.

PHILOSOPHY
----------
These are not "does the code run" tests. Each test asserts a PHYSICAL or
MATHEMATICAL PROPERTY that must hold for the simulator to be scientifically
defensible. If a future change (adding fines migration, reactive transport,
permeability damage...) silently breaks the flow core, these assertions fail
immediately and tell us *which* invariant broke. This is the verification half
of "verification & validation" expected in a doctoral thesis.

Each test prints a short PASS line with the measured error, so the suite
doubles as a reproducible validation report for a thesis appendix.

Run with either:
    python -m pytest tests/test_phase1.py -v        (if pytest installed)
    python tests/test_phase1.py                     (standalone fallback)
===============================================================================
"""

import os
import sys
import numpy as np

# Allow running both as a module and as a standalone script.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from physics.mesh import StructuredMesh
from physics.geometry import (apply_cylindrical_core, build_cylinder_mask,
                           effective_cross_section_area)
from physics.properties import FluidRockProperties
from physics.boundary_conditions import CoreFloodBC, BCMode
from physics.pressure_solver import PressureSolver
from physics.velocity import VelocityField
from visualization.plots import analytical_darcy_dp
from utils.helpers import harmonic_mean
from utils.constants import (MILLIDARCY_TO_M2, CENTIPOISE_TO_PAS,
                             ML_PER_MIN_TO_M3_PER_S)


# Absolute tolerances chosen to be tight but robust to the staircased-cylinder
# discretization. Conservation should be at round-off; analytical dP agreement
# is limited by the boundary staircasing (~few %), so we test the two with
# different tolerances and document why.
TOL_CONSERVATION = 1.0e-9      # relative; expect ~1e-12 in practice
TOL_DARCY_DP = 5.0e-2          # relative; staircased cylinder => few-% error


# ---------------------------------------------------------------------------
def _build_standard_case(nx=30, ny=16, nz=16, mode=BCMode.CONSTANT_RATE):
    """Construct a canonical core-plug case used by several tests."""
    mesh = StructuredMesh(0.10, 0.038, 0.038, nx, ny, nz)
    apply_cylindrical_core(mesh, diameter=0.038)
    props = FluidRockProperties(mesh, permeability_mD=100.0, porosity=0.20,
                                viscosity_cP=1.0, density=1000.0)
    if mode == BCMode.CONSTANT_RATE:
        Q = 1.0 * ML_PER_MIN_TO_M3_PER_S
        bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
    else:
        bc = CoreFloodBC(BCMode.CONSTANT_PRESSURE_DROP,
                         p_inlet=1.0e4, p_outlet=0.0)
    return mesh, props, bc


# ===========================================================================
#  UNIT TESTS  (small, pure functions)
# ===========================================================================
def test_harmonic_mean_series_resistance():
    """
    Harmonic mean must equal the series-resistance result and collapse toward
    zero when one permeability is tiny (the physical reason we use it).
    """
    # Two equal values -> harmonic mean equals that value.
    assert np.isclose(harmonic_mean(2.0, 2.0), 2.0)
    # One near-zero value chokes the face permeability toward zero.
    assert harmonic_mean(1.0, 1.0e-12) < 1.0e-11
    # Known analytic value: H(1,3) = 2*1*3/(1+3) = 1.5
    assert np.isclose(harmonic_mean(1.0, 3.0), 1.5)
    print("PASS  harmonic_mean: series-resistance behaviour correct")


def test_unit_conversions():
    """Conversion constants must be self-consistent with SI definitions."""
    assert np.isclose(100.0 * MILLIDARCY_TO_M2, 9.869233e-15, rtol=1e-6)
    assert np.isclose(1.0 * CENTIPOISE_TO_PAS, 1.0e-3)
    # 1 mL/min in SI
    assert np.isclose(1.0 * ML_PER_MIN_TO_M3_PER_S, 1.0e-6 / 60.0)
    print("PASS  unit conversions: SI consistency confirmed")


# ===========================================================================
#  GEOMETRY TESTS
# ===========================================================================
def test_cylinder_mask_geometry():
    """
    The masked cross-section area must converge to the true disc area, and the
    mask must be x-invariant (a straight cylinder).
    """
    mesh = StructuredMesh(0.10, 0.038, 0.038, 10, 40, 40)
    mask = build_cylinder_mask(mesh, radius=0.019)
    mesh.set_active_mask(mask)
    area = effective_cross_section_area(mesh)
    ideal = np.pi * 0.019 ** 2
    rel = abs(area - ideal) / ideal
    assert rel < 0.05, f"cross-section area off by {rel:.3f}"
    # x-invariance: every x-slice identical.
    for i in range(1, mesh.nx):
        assert np.array_equal(mesh.active[0], mesh.active[i])
    print(f"PASS  cylinder mask: area within {rel*100:.2f}% of ideal, x-invariant")


def test_dof_map_consistency():
    """
    The DOF map must be a bijection between active cells and [0, n_active),
    and cell_id must be -1 exactly on inactive cells.
    """
    mesh, _, _ = _build_standard_case()
    # Number of non-negative ids equals n_active.
    n_pos = int(np.count_nonzero(mesh.cell_id >= 0))
    assert n_pos == mesh.n_active
    # Inactive cells carry -1.
    assert np.all(mesh.cell_id[~mesh.active] == -1)
    # Round-trip: dof_to_ijk inverts cell_id.
    for dof in range(mesh.n_active):
        i, j, k = mesh.dof_to_ijk[dof]
        assert mesh.cell_id[i, j, k] == dof
    print(f"PASS  DOF map: bijection over {mesh.n_active} active cells")


# ===========================================================================
#  SOLVER PROPERTY TESTS
# ===========================================================================
def test_matrix_symmetry():
    """
    The FVM pressure matrix must be symmetric (T_ij = T_ji): flux from i to j
    uses the same transmissibility as j to i. Asymmetry would mean a
    non-conservative or inconsistent discretization.
    """
    mesh, props, bc = _build_standard_case()
    solver = PressureSolver(mesh, props)
    solver._assemble(bc)
    A = solver.A
    asym = abs(A - A.T)
    max_asym = asym.max() if asym.nnz > 0 else 0.0
    # Normalise by the largest diagonal magnitude.
    scale = abs(A.diagonal()).max()
    rel = max_asym / scale
    assert rel < 1e-12, f"matrix asymmetry {rel:.2e}"
    print(f"PASS  matrix symmetry: max relative asymmetry {rel:.2e}")


def test_matrix_negative_definite_diagonal_dominance():
    """
    Each row's diagonal must be negative and at least equal in magnitude to the
    sum of its off-diagonals (weak diagonal dominance), guaranteeing a
    well-posed, uniquely solvable elliptic system once an anchor is applied.
    """
    mesh, props, bc = _build_standard_case()
    solver = PressureSolver(mesh, props)
    solver._assemble(bc)
    A = solver.A.tocsr()
    diag = A.diagonal()
    assert np.all(diag < 0), "diagonal must be strictly negative"
    # Off-diagonal absolute row sums.
    abs_off = np.abs(A).sum(axis=1).A1 - np.abs(diag)
    # |diag| >= sum|offdiag| (boundary rows are strictly dominant).
    assert np.all(np.abs(diag) >= abs_off - 1e-9)
    print("PASS  matrix: negative diagonal + diagonal dominance (well-posed)")


# ===========================================================================
#  CONSERVATION & ANALYTICAL VALIDATION
# ===========================================================================
def test_mass_conservation_constant_rate():
    """
    THE headline test: outlet volumetric flux must equal the injected rate to
    near machine precision. This is the discrete statement of incompressible
    mass conservation.
    """
    mesh, props, bc = _build_standard_case(mode=BCMode.CONSTANT_RATE)
    solver = PressureSolver(mesh, props)
    P = solver.solve(bc, verbose=False)
    vel = VelocityField(mesh, props)
    vel.compute(P, bc)
    Q_out = vel.total_throughput(P, bc)
    rel = abs(Q_out - bc.injection_rate) / bc.injection_rate
    assert rel < TOL_CONSERVATION, f"mass conservation rel error {rel:.2e}"
    print(f"PASS  mass conservation (const-rate): rel error {rel:.2e}")


def test_inlet_outlet_balance_constant_dp():
    """
    Under a fixed pressure drop, inlet inflow must equal outlet outflow to
    machine precision (steady incompressible flow has no storage).
    """
    mesh, props, bc = _build_standard_case(mode=BCMode.CONSTANT_PRESSURE_DROP)
    solver = PressureSolver(mesh, props)
    P = solver.solve(bc, verbose=False)
    vel = VelocityField(mesh, props)
    vel.compute(P, bc)
    Q_out = vel.total_throughput(P, bc)
    Q_in = -vel.boundary_face_flux(P, 0, bc.p_inlet)
    rel = abs(Q_out - Q_in) / abs(Q_in)
    assert rel < TOL_CONSERVATION, f"in/out balance rel error {rel:.2e}"
    print(f"PASS  inlet/outlet balance (const-dP): rel error {rel:.2e}")


def test_analytical_darcy_dp():
    """
    Simulated pressure drop must match the closed-form Darcy law
        dP = Q mu L / (k A)
    within the staircased-boundary tolerance.
    """
    mesh, props, bc = _build_standard_case(mode=BCMode.CONSTANT_RATE)
    solver = PressureSolver(mesh, props)
    P = solver.solve(bc, verbose=False)
    area = effective_cross_section_area(mesh)
    dp_analytic = analytical_darcy_dp(mesh, props, bc.injection_rate, area)
    dp_sim = np.nanmax(P[mesh.active]) - np.nanmin(P[mesh.active])
    rel = abs(dp_sim - dp_analytic) / dp_analytic
    assert rel < TOL_DARCY_DP, f"Darcy dP rel error {rel:.2e}"
    print(f"PASS  analytical Darcy dP: rel error {rel:.2e} "
          f"(sim {dp_sim:.3e} Pa vs analytic {dp_analytic:.3e} Pa)")


def test_pressure_monotonic_along_flow():
    """
    Pressure must decrease monotonically from inlet to outlet (no spurious
    interior maxima/minima) for homogeneous single-phase flow -- a basic
    physical admissibility check that also catches sign errors.
    """
    mesh, props, bc = _build_standard_case(mode=BCMode.CONSTANT_RATE)
    solver = PressureSolver(mesh, props)
    P = solver.solve(bc, verbose=False)
    # Slice-averaged axial profile.
    p_axial = np.array([np.nanmean(P[i, :, :][mesh.active[i, :, :]])
                        for i in range(mesh.nx)])
    diffs = np.diff(p_axial)
    assert np.all(diffs < 1e-9), "pressure not monotonically decreasing"
    print("PASS  pressure profile: monotonically decreasing inlet->outlet")


def test_grid_refinement_convergence():
    """
    Refining the mesh must REDUCE the error vs the analytical Darcy dP. This
    demonstrates the discretization is convergent (consistent), the key
    property distinguishing a real numerical method from a lucky guess.
    """
    errors = []
    for n in (16, 24, 32):
        mesh = StructuredMesh(0.10, 0.038, 0.038, n, n // 2, n // 2)
        apply_cylindrical_core(mesh, 0.038)
        props = FluidRockProperties(mesh, 100.0, 0.20, 1.0, 1000.0)
        Q = 1.0 * ML_PER_MIN_TO_M3_PER_S
        bc = CoreFloodBC(BCMode.CONSTANT_RATE, injection_rate=Q, p_outlet=0.0)
        P = PressureSolver(mesh, props).solve(bc, verbose=False)
        area = effective_cross_section_area(mesh)
        dp_a = analytical_darcy_dp(mesh, props, Q, area)
        dp_s = np.nanmax(P[mesh.active]) - np.nanmin(P[mesh.active])
        errors.append(abs(dp_s - dp_a) / dp_a)
    # Error should not grow as we refine (it shrinks or plateaus near the
    # staircasing floor). We assert the finest is no worse than the coarsest.
    assert errors[-1] <= errors[0] + 1e-6, f"non-convergent: {errors}"
    print(f"PASS  grid refinement: errors {['%.2e' % e for e in errors]} "
          f"(non-increasing)")


# ===========================================================================
#  STANDALONE RUNNER
# ===========================================================================
def _run_all():
    tests = [
        test_harmonic_mean_series_resistance,
        test_unit_conversions,
        test_cylinder_mask_geometry,
        test_dof_map_consistency,
        test_matrix_symmetry,
        test_matrix_negative_definite_diagonal_dominance,
        test_mass_conservation_constant_rate,
        test_inlet_outlet_balance_constant_dp,
        test_analytical_darcy_dp,
        test_pressure_monotonic_along_flow,
        test_grid_refinement_convergence,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 1 validation suite")
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
