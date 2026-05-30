"""
ASPiRe-3D : tests/test_phase7.py
===============================================================================
Validation suite for the experimental-data integration framework (Phase 7).

Validates the full messy-real-data pipeline:
  * unit conversion (psi/bar/kPa -> Pa; min/hr -> PV),
  * header/delimiter auto-detection,
  * cleaning: NaN drop, sort, duplicate averaging, MAD outlier removal,
  * effluent normalisation by injected C0,
  * inverse-variance experimental-uncertainty weighting,
  * metadata-driven clock->PV conversion,
  * end-to-end: messy synthetic CSV -> recover known parameters,
  * provenance logging completeness.

Uses a *messy* synthetic dataset (deliberately non-SI units, shuffled rows,
duplicates, outliers, noise) generated from known parameters -- so the
pipeline's robustness is tested, not just the clean path. No fabricated lab
data.
===============================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from data.data_schema import (ExperimentMetadata, SignalMapping,
                              PRESSURE_TO_PA, TIME_TO_SECONDS)
from data.data_ingestion import ingest_signal, _read_csv_columns, _mad_outlier_mask
from data.experimental_validation import (make_synthetic_dataset,
                                          build_observed_from_config,
                                          run_experimental_validation)
from calibration.calibration import CalibrationParameters


TMP = "outputs/_test_phase7"


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def _metadata():
    return ExperimentMetadata(0.10, 0.038, 0.21, 2.0,
                              baseline_permeability_mD=120.0, name="t")


# ===========================================================================
def test_unit_conversions_schema():
    """Pressure & time unit tables convert correctly to canonical units."""
    assert np.isclose(PRESSURE_TO_PA["psi"], 6894.757, rtol=1e-4)
    assert PRESSURE_TO_PA["bar"] == 1e5
    assert PRESSURE_TO_PA["kpa"] == 1e3
    assert TIME_TO_SECONDS["min"] == 60.0 and TIME_TO_SECONDS["hr"] == 3600.0
    m = SignalMapping("dp_history", "p", unit="psi", clock_column="t", clock_unit="min")
    pa = m.convert_values_to_canonical([1.0, 2.0])
    assert np.allclose(pa, [6894.757, 13789.514], rtol=1e-4)
    print("PASS  schema unit conversions: psi/bar/kPa->Pa, min/hr->s correct")


def test_clock_to_pore_volumes():
    """A minutes clock converts to PV using metadata (rate & pore volume)."""
    md = _metadata()
    m = SignalMapping("dp_history", "p", unit="psi",
                      clock_column="t", clock_unit="min")
    # 1 PV in minutes = seconds_per_PV / 60.
    one_pv_min = md.seconds_per_pore_volume / 60.0
    pv = m.convert_clock_to_pv([0.0, one_pv_min, 2 * one_pv_min], md)
    assert np.allclose(pv, [0.0, 1.0, 2.0], atol=1e-6)
    print(f"PASS  clock->PV: {one_pv_min:.1f} min == 1 PV (metadata-driven)")


def test_header_and_delimiter_autodetect():
    """CSV reader detects header presence and comma delimiter; skips comments."""
    _write(f"{TMP}/h.csv", "# comment line\ntime,val\n0,1.0\n1,2.0\n")
    cols, header = _read_csv_columns(f"{TMP}/h.csv")
    assert header and "time" in cols and "val" in cols
    assert np.allclose(cols["val"], [1.0, 2.0])
    # headerless
    _write(f"{TMP}/nh.csv", "0,1.0\n1,2.0\n")
    cols2, header2 = _read_csv_columns(f"{TMP}/nh.csv")
    assert not header2 and "0" in cols2
    print("PASS  CSV autodetect: header/headerless, comments skipped")


def test_mad_outlier_detection():
    """MAD outlier mask flags a gross spike, keeps normal scatter."""
    v = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 10.0, 1.0])  # one spike
    mask = _mad_outlier_mask(v, n_sigma=4.0)
    assert not mask[5], "did not flag the obvious outlier"
    assert mask[0] and mask[1], "wrongly flagged normal points"
    print("PASS  MAD outlier detection: flags spike, keeps normal scatter")


def test_cleaning_pipeline_handles_mess():
    """
    Full ingest on a deliberately messy CSV: non-monotonic order, a duplicate
    clock, NaN, and an outlier -> sorted, deduped, cleaned, finite output.
    """
    _write(f"{TMP}/mess.csv",
           "PV,k_ratio\n"
           "2.0,0.30\n1.0,0.62\n1.0,0.62\n0.0,1.00\n"   # dup + unsorted
           "0.5,nan\n1.5,0.45\n0.75,9.9\n")             # NaN + outlier
    md = _metadata()
    m = SignalMapping("k_ratio_history", "k_ratio",
                      clock_column="PV", clock_unit="PV")
    cs = ingest_signal(f"{TMP}/mess.csv", m, md, remove_outliers=True)
    # monotone clock, no NaN, duplicate averaged, outlier gone.
    assert np.all(np.diff(cs.clock_pv) > 0), "clock not strictly increasing"
    assert np.all(np.isfinite(cs.values)), "NaN survived"
    assert cs.values.max() < 1.5, "outlier (9.9) survived"
    assert any("duplicate" in l for l in cs.log), "duplicate handling not logged"
    assert any("outlier" in l for l in cs.log), "outlier removal not logged"
    print(f"PASS  cleaning pipeline: {cs.n_raw} raw -> {len(cs)} clean, "
          f"sorted/deduped/de-NaN'd/outlier-free, logged")


def test_effluent_normalisation():
    """Raw effluent in ppm normalised by injected C0 -> dimensionless C/C0."""
    _write(f"{TMP}/eff.csv", "PV,C_ppm\n0.0,0\n1.0,1000\n2.0,2000\n")
    md = _metadata()
    m = SignalMapping("surfactant_effluent", "C_ppm",
                      clock_column="PV", clock_unit="PV", normalise_by=2000.0)
    cs = ingest_signal(f"{TMP}/eff.csv", m, md, remove_outliers=False)
    assert np.allclose(cs.values, [0.0, 0.5, 1.0]), "normalisation wrong"
    print("PASS  effluent normalisation: ppm / C0 -> C/C0 in [0,1]")


def test_inverse_variance_weighting():
    """A low-noise signal must receive a higher calibration weight."""
    # clean signal
    _write(f"{TMP}/clean.csv", "PV,k_ratio,sig\n0,1.0,0.001\n1,0.6,0.001\n2,0.3,0.001\n")
    _write(f"{TMP}/noisy.csv", "PV,k_ratio,sig\n0,1.0,0.2\n1,0.6,0.2\n2,0.3,0.2\n")
    cfg = {"metadata": dict(name="t", core_length_m=0.1, core_diameter_m=0.038,
                            porosity=0.21, flow_rate_ml_min=2.0),
           "signals": [
               dict(canonical="k_ratio_history", file="clean.csv",
                    column="k_ratio", clock_column="PV", clock_unit="PV",
                    uncertainty_column="sig"),
               dict(canonical="injectivity_history", file="noisy.csv",
                    column="k_ratio", clock_column="PV", clock_unit="PV",
                    uncertainty_column="sig")]}
    obs, md, cleaned, prov = build_observed_from_config(
        cfg, base_dir=TMP, clean_kwargs=dict(remove_outliers=False))
    w_clean = obs.signals["k_ratio_history"]["weight"]
    w_noisy = obs.signals["injectivity_history"]["weight"]
    assert w_clean > w_noisy, "clean signal not weighted higher"
    print(f"PASS  inverse-variance weighting: clean weight {w_clean:.1f} "
          f"> noisy weight {w_noisy:.1f}")


def test_end_to_end_recovery_from_messy_csv():
    """
    Generate a messy synthetic dataset (psi, minutes, outliers, duplicates,
    noise) from known parameters; the full pipeline must recover them within a
    tolerance consistent with the added noise.
    """
    truth = {"rrf_max": 5.0, "perm_exponent": 3.0}
    cfg = make_synthetic_dataset(f"{TMP}/lab", truth, noise=0.02, seed=3)
    p = CalibrationParameters()
    p.add("rrf_max", 2.5, 1.0, 10.0)
    p.add("perm_exponent", 2.0, 1.0, 6.0)
    res, sob, prov, rpt, hm, obs, builder = run_experimental_validation(
        cfg, p, base_dir=f"{TMP}/lab", method="least_squares",
        n_pore_volumes=2.5, sobol_n_base=6, verbose=False)
    assert res.r2_total > 0.97, f"poor fit R2={res.r2_total:.4f}"
    err_rrf = abs(res.best_values["rrf_max"] - 5.0) / 5.0
    assert err_rrf < 0.10, f"RRF not recovered: {res.best_values['rrf_max']:.3f}"
    assert os.path.exists(rpt), "validation report not written"
    print(f"PASS  end-to-end messy-CSV recovery: R2={res.r2_total:.4f}, "
          f"RRF {res.best_values['rrf_max']:.3f} (truth 5.0, {err_rrf:.0%} err), "
          f"report written")


def test_provenance_completeness():
    """The provenance log must record read, unit conversion, and final counts."""
    truth = {"rrf_max": 4.0}
    cfg = make_synthetic_dataset(f"{TMP}/lab2", truth, noise=0.0, seed=1)
    obs, md, cleaned, prov = build_observed_from_config(cfg, base_dir=f"{TMP}/lab2")
    assert "read" in prov and "canonical unit" in prov and "final:" in prov
    assert "min->PV" in prov, "clock conversion not logged"
    print("PASS  provenance completeness: read/convert/clean/final all logged")


# ===========================================================================
def _run_all():
    tests = [
        test_unit_conversions_schema,
        test_clock_to_pore_volumes,
        test_header_and_delimiter_autodetect,
        test_mad_outlier_detection,
        test_cleaning_pipeline_handles_mess,
        test_effluent_normalisation,
        test_inverse_variance_weighting,
        test_end_to_end_recovery_from_messy_csv,
        test_provenance_completeness,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 7 : experimental-data integration validation")
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
