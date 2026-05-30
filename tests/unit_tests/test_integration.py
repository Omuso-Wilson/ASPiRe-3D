"""
ASPiRe-3D : tests/unit_tests/test_integration.py
===============================================================================
Integration tests for the software-framework layer (Phases 1-3):
  * module imports resolve under the new package layout,
  * config loading: defaults, validation, error messages, unit conversion,
  * the three bundled example configs load and validate,
  * workflow construction and a fast end-to-end run (baseline-style),
  * end-to-end calibration run via the workflow (synthetic-truth).

These complement the physics/calibration validation suites (tests/validation_tests).
===============================================================================
"""

import os
import sys
import json
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main.config_loader import load_config, load_config_dict, ConfigError


# ===========================================================================
#  MODULE IMPORTS
# ===========================================================================
def test_package_imports():
    """All public package entry points import cleanly under the new layout."""
    import physics.darcy_solver, physics.transport_solver
    import physics.reactive_transport, physics.fines_migration
    import physics.permeability_update, physics.coupled_simulator
    import calibration.history_matching, calibration.uncertainty
    import calibration.sensitivity, calibration.optimization
    import data.ingestion, data.preprocessing, data.schemas
    import visualization.plots, visualization.validation_plots
    import visualization.sensitivity_plots
    import main.config_loader, main.workflow, main.run_aspire3d
    # spot-check key symbols
    from physics.darcy_solver import PressureSolver, VelocityField
    from physics.permeability_update import FormationDamage, kozeny_carman
    from calibration.optimization import HistoryMatcher
    from data.schemas import ExperimentMetadata
    assert PressureSolver and VelocityField and FormationDamage
    assert kozeny_carman and HistoryMatcher and ExperimentMetadata
    print("PASS  package imports: physics/calibration/data/visualization/main all resolve")


# ===========================================================================
#  CONFIG LOADING
# ===========================================================================
def test_config_defaults_filled():
    """Optional sections/keys are filled from defaults; required SI accessors work."""
    cfg = load_config_dict({
        "core": {"length_cm": 10, "diameter_cm": 3.8,
                 "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0}})
    assert cfg.core_length_m == 0.10            # cm -> m
    assert cfg.core_diameter_m == 0.038
    assert cfg.flow_rate_ml_min == 2.0
    # defaults present
    assert cfg.section("physics")["perm_model"] == "kozeny_carman"
    assert cfg.section("numerics")["grid"] == [30, 12, 12]
    assert cfg.section("calibration")["enabled"] is False
    print("PASS  config defaults: optional keys filled, cm->m conversion correct")


def test_config_missing_required_raises():
    """Missing a required key produces a clear ConfigError."""
    try:
        load_config_dict({"core": {"length_cm": 10}})   # missing most
        assert False, "should have raised"
    except ConfigError as e:
        assert "required" in str(e).lower()
    print("PASS  config validation: missing required key raises clear ConfigError")


def test_config_range_validation():
    """Out-of-range values are rejected with an actionable message."""
    for bad, key in (({"porosity": 1.5}, "porosity"),
                     ({"permeability_mD": -5}, "permeability")):
        core = {"length_cm": 10, "diameter_cm": 3.8,
                "porosity": 0.2, "permeability_mD": 100}
        core.update(bad)
        try:
            load_config_dict({"core": core,
                              "injection": {"flow_rate_ccmin": 2, "pore_volumes": 3}})
            assert False, f"should reject {key}"
        except ConfigError as e:
            assert key in str(e)
    print("PASS  config validation: out-of-range porosity/permeability rejected")


def test_config_enum_validation():
    """Invalid enum (perm_model) is rejected."""
    try:
        load_config_dict({
            "core": {"length_cm": 10, "diameter_cm": 3.8,
                     "porosity": 0.2, "permeability_mD": 100},
            "injection": {"flow_rate_ccmin": 2, "pore_volumes": 3},
            "physics": {"perm_model": "nonsense"}})
        assert False, "should reject bad perm_model"
    except ConfigError as e:
        assert "perm_model" in str(e)
    print("PASS  config validation: invalid perm_model enum rejected")


def test_calibration_config_validation():
    """Calibration enabled without parameters, or with bad bounds, is rejected."""
    base = {"core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2,
                     "permeability_mD": 100},
            "injection": {"flow_rate_ccmin": 2, "pore_volumes": 3}}
    # enabled, no params
    try:
        load_config_dict({**base, "calibration": {"enabled": True, "parameters": []}})
        assert False
    except ConfigError as e:
        assert "parameter" in str(e).lower()
    # init outside bounds
    try:
        load_config_dict({**base, "calibration": {"enabled": True, "parameters": [
            {"name": "rrf_max", "init": 50.0, "min": 1.0, "max": 10.0}]}})
        assert False
    except ConfigError as e:
        assert "rrf_max" in str(e)
    print("PASS  calibration config: empty params and out-of-bounds init rejected")


def test_example_configs_load():
    """All three bundled example configs load and validate."""
    cfgdir = os.path.join(_ROOT, "experiments", "configs")
    for name in ("baseline_case", "sensitivity_case", "validation_case"):
        cfg = load_config(os.path.join(cfgdir, f"{name}.json"))
        assert cfg.raw["name"] == name
    print("PASS  example configs: baseline/sensitivity/validation all valid")


def test_bad_json_raises():
    """A malformed JSON file raises a clear ConfigError."""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        f.write("{ not valid json ]")
        path = f.name
    try:
        load_config(path)
        assert False
    except ConfigError as e:
        assert "JSON" in str(e) or "json" in str(e)
    finally:
        os.unlink(path)
    print("PASS  config loading: malformed JSON raises clear ConfigError")


# ===========================================================================
#  WORKFLOW EXECUTION (end-to-end)
# ===========================================================================
def test_workflow_end_to_end_baseline():
    """A fast baseline-style workflow runs end to end and produces outputs."""
    from main.workflow import Workflow
    tmp = tempfile.mkdtemp()
    cfg = load_config_dict({
        "name": "itest_baseline",
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2,
                 "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 1.5},
        "numerics": {"courant": 2.0, "grid": [14, 6, 6]},
        "physics": {"enable_polymer_retention": True, "enable_damage": True,
                    "perm_model": "power_law"},
        "parameters": {"rrf_max": 4.0, "perm_exponent": 3.0},
        "output": {"outdir": tmp, "prefix": "itest", "save_figures": True,
                   "save_report": True}})
    results = Workflow(cfg, verbose=False).run()
    assert results["simulation"] is not None, "no simulation result"
    assert os.path.exists(results["report"]), "no report written"
    assert os.path.exists(results["logfile"]), "no log written"
    assert len(results["figures"]) >= 1, "no figures written"
    inj = results["simulation"].history["injectivity_ratio"][-1]
    assert 0.0 < inj <= 1.0, f"implausible injectivity {inj}"
    print(f"PASS  workflow end-to-end baseline: ran, I/I0={inj:.3f}, "
          f"report+log+{len(results['figures'])} figures written")


def test_workflow_end_to_end_calibration():
    """A fast calibration workflow (synthetic-truth) recovers the truth."""
    from main.workflow import Workflow
    tmp = tempfile.mkdtemp()
    cfg = load_config_dict({
        "name": "itest_calib",
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.21,
                 "permeability_mD": 120},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 2.0},
        "numerics": {"courant": 2.0, "grid": [14, 6, 6]},
        "physics": {"enable_polymer_retention": True, "enable_damage": True,
                    "perm_model": "power_law"},
        "parameters": {"rrf_max": 4.0, "perm_exponent": 3.0},
        "calibration": {"enabled": True, "method": "least_squares",
                        "max_nfev": 25,
                        "synthetic_truth": {"rrf_max": 5.0},
                        "parameters": [
                            {"name": "rrf_max", "init": 2.5, "min": 1.0, "max": 10.0}]},
        "output": {"outdir": tmp, "prefix": "itestcal", "save_figures": False,
                   "save_report": True}})
    results = Workflow(cfg, verbose=False).run()
    r = results["calibration"]
    assert r is not None, "no calibration result"
    err = abs(r.best_values["rrf_max"] - 5.0) / 5.0
    assert r.r2_total > 0.99 and err < 0.05, \
        f"calibration failed: R2={r.r2_total:.4f}, rrf={r.best_values['rrf_max']:.3f}"
    print(f"PASS  workflow end-to-end calibration: R2={r.r2_total:.4f}, "
          f"RRF {r.best_values['rrf_max']:.3f} recovered (truth 5.0, {err:.1%} err)")


# ===========================================================================
def _run_all():
    tests = [
        test_package_imports,
        test_config_defaults_filled,
        test_config_missing_required_raises,
        test_config_range_validation,
        test_config_enum_validation,
        test_calibration_config_validation,
        test_example_configs_load,
        test_bad_json_raises,
        test_workflow_end_to_end_baseline,
        test_workflow_end_to_end_calibration,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Integration tests : framework (config, workflow, imports)")
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
