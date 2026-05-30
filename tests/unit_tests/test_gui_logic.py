"""
ASPiRe-3D : tests/unit_tests/test_gui_logic.py
===============================================================================
Tests for GUI logic (without streamlit dependency).

Tests config modification, metrics extraction, and bundled config loading.
===============================================================================
"""

import os
import sys
import tempfile

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from main.config_loader import load_config_dict, ConfigError
from main.workflow import Workflow
from pathlib import Path


def test_gui_config_modification():
    """GUI can load and modify a config without breaking it."""
    config = {
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
    }
    
    config["core"]["length_cm"] = 15.0
    config["core"]["porosity"] = 0.25
    config["injection"]["flow_rate_ccmin"] = 3.0
    
    cfg = load_config_dict(config)
    assert cfg.core_length_m == 0.15
    assert cfg.porosity == 0.25
    assert cfg.flow_rate_ml_min == 3.0
    print("PASS  GUI config modification: edited config validates and converts correctly")


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


def test_gui_all_bundled_configs_validate():
    """All bundled configs load and validate."""
    config_dir = Path(__file__).parent.parent.parent / "experiments" / "configs"
    for name in ["baseline_case", "sensitivity_case", "validation_case"]:
        config_path = config_dir / f"{name}.json"
        assert config_path.exists(), f"{name}.json not found"
        
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        
        try:
            cfg = load_config_dict(config)
            assert cfg is not None
        except ConfigError as e:
            assert False, f"{name} failed validation: {e}"
    
    print("PASS  GUI bundled configs: all three load and validate correctly")


def test_gui_metrics_extraction_fast():
    """Test metrics extraction from a fast simulation."""
    config = {
        "name": "gui_test_metrics",
        "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
        "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 1.0},
        "numerics": {"courant": 2.0, "grid": [14, 6, 6]},
        "physics": {"enable_damage": True, "perm_model": "kozeny_carman"},
        "parameters": {"rrf_max": 4.0},
        "output": {"outdir": tempfile.mkdtemp(), "save_figures": False, "save_report": False},
    }
    
    cfg = load_config_dict(config)
    results = Workflow(cfg, verbose=False).run()
    
    # Simulate what the GUI does
    metrics = {}
    if "simulation" in results and results["simulation"] is not None:
        sim = results["simulation"]
        if hasattr(sim, "history"):
            h = sim.history
            metrics["final_injectivity"] = float(h["injectivity_ratio"][-1]) if "injectivity_ratio" in h else None
            metrics["min_permeability_ratio"] = float(h["k_ratio_min"][-1]) if "k_ratio_min" in h else None
    
    assert "final_injectivity" in metrics
    assert "min_permeability_ratio" in metrics
    assert metrics["final_injectivity"] is not None
    assert 0.0 < metrics["final_injectivity"] <= 1.0
    print(f"PASS  GUI metrics extraction: final I/I0={metrics['final_injectivity']:.3f}, "
          f"min k/k0={metrics['min_permeability_ratio']:.3f}")


def test_gui_parameter_ranges():
    """GUI parameter sliders respect valid ranges."""
    test_cases = [
        ({"core": {"length_cm": 0.5, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100}}, True),  # valid
        ({"core": {"length_cm": 150, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100}}, True),  # edge valid
        ({"core": {"length_cm": 0, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100}}, False),  # invalid
        ({"core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.9, "permeability_mD": 100}}, False),  # invalid porosity
    ]
    
    for config_delta, should_pass in test_cases:
        config = {
            "core": {"length_cm": 10, "diameter_cm": 3.8, "porosity": 0.2, "permeability_mD": 100},
            "injection": {"flow_rate_ccmin": 2.0, "pore_volumes": 3.0},
        }
        config.update(config_delta)
        config["core"].update(config_delta.get("core", {}))
        
        try:
            cfg = load_config_dict(config)
            if not should_pass:
                assert False, f"Config should have failed: {config}"
        except ConfigError:
            if should_pass:
                assert False, f"Config should have passed: {config}"
    
    print("PASS  GUI parameter ranges: validation works correctly")


def _run_all():
    tests = [
        test_gui_config_modification,
        test_gui_physics_toggle,
        test_gui_output_prefix,
        test_gui_all_bundled_configs_validate,
        test_gui_parameter_ranges,
        test_gui_metrics_extraction_fast,
    ]
    print("=" * 70)
    print("ASPiRe-3D  GUI Logic tests (streamlit-independent)")
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
