"""
ASPiRe-3D : tests/unit_tests/test_gui_integration.py
===============================================================================
Integration tests for the Streamlit GUI components.

Tests that the GUI components correctly interface with the existing
run_aspire3d.py workflow without modifying physics modules.
===============================================================================
"""

import os
import sys
import json
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main.config_loader import load_config_dict, ConfigError


def test_gui_session_state():
    """GUI session state initialization works."""
    # We can't directly test Streamlit state without running the app,
    # but we can test the supporting functions
    from gui.session import get_bundled_config
    
    baseline = get_bundled_config("baseline_case")
    assert baseline is not None
    assert "core" in baseline
    assert baseline["core"]["length_cm"] == 10.0
    print("PASS  GUI session: bundled config loads correctly")


def test_gui_config_modification():
    """GUI can load and modify a config without breaking it."""
    config = {
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
    }
    
    # Modify config (as the GUI sidebar would)
    config["core"]["length_cm"] = 15.0
    config["core"]["porosity"] = 0.25
    config["injection"]["flow_rate_ccmin"] = 3.0
    
    # Validate modified config
    try:
        cfg = load_config_dict(config)
        assert cfg.core_length_m == 0.15
        assert cfg.porosity == 0.25
        assert cfg.flow_rate_ml_min == 3.0
        print("PASS  GUI config modification: edited config validates and converts correctly")
    except ConfigError as e:
        assert False, f"Modified config should validate: {e}"


def test_gui_physics_toggle():
    """GUI physics toggles are preserved through config load."""
    config = {
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
        "physics": {
            "enable_surfactant_adsorption": False,
            "enable_polymer_retention": True,
            "enable_precipitation": True,
            "enable_fines": False,
            "enable_damage": True,
            "perm_model": "power_law",
        }
    }
    
    cfg = load_config_dict(config)
    phys = cfg.section("physics")
    assert phys["enable_surfactant_adsorption"] is False
    assert phys["enable_polymer_retention"] is True
    assert phys["enable_precipitation"] is True
    assert phys["perm_model"] == "power_law"
    print("PASS  GUI physics toggles: correctly preserved through config load")


def test_gui_output_prefix():
    """GUI output prefix is correctly set."""
    config = {
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
        "output": {"prefix": "my_custom_run"},
    }
    
    cfg = load_config_dict(config)
    assert cfg.get("output", "prefix") == "my_custom_run"
    print("PASS  GUI output control: custom prefix preserved")


def test_gui_metrics_extraction():
    """Test metrics extraction from a simple simulation."""
    from gui.session import get_metrics_from_result
    from main.workflow import Workflow
    
    config = {
        "name": "gui_test",
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 1.0},
        "numerics": {"courant": 2.0, "grid": [14, 6, 6]},
        "physics": {"enable_damage": True, "perm_model": "kozeny_carman"},
        "parameters": {"rrf_max": 4.0},
        "output": {"outdir": tempfile.mkdtemp(), "save_figures": False, "save_report": False},
    }
    
    cfg = load_config_dict(config)
    results = Workflow(cfg, verbose=False).run()
    
    metrics = get_metrics_from_result(results)
    assert "final_injectivity" in metrics
    assert "min_permeability_ratio" in metrics
    assert metrics["final_injectivity"] is not None
    assert 0.0 < metrics["final_injectivity"] <= 1.0
    print(f"PASS  GUI metrics extraction: final I/I0={metrics['final_injectivity']:.3f}")


def test_gui_all_bundled_configs():
    """All bundled configs load and validate through GUI logic."""
    from gui.session import get_bundled_config
    
    for name in ["baseline_case", "sensitivity_case", "validation_case"]:
        config = get_bundled_config(name)
        assert config is not None, f"Could not load {name}"
        try:
            cfg = load_config_dict(config)
            assert cfg is not None
        except ConfigError as e:
            assert False, f"{name} failed validation: {e}"
    
    print("PASS  GUI bundled configs: all three load and validate correctly")


def _run_all():
    tests = [
        test_gui_session_state,
        test_gui_config_modification,
        test_gui_physics_toggle,
        test_gui_output_prefix,
        test_gui_all_bundled_configs,
        test_gui_metrics_extraction,
    ]
    print("=" * 70)
    print("ASPiRe-3D  GUI Integration tests")
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
