"""
ASPiRe-3D : tests/test_phase6.py
===============================================================================
Validation suite for the HISTORY-MATCHING / CALIBRATION engine (Phase 6).

The engine operates on whatever observed data is supplied; no experimental data
is bundled or fabricated. Correctness is therefore validated by SYNTHETIC-TRUTH
RECOVERY -- the standard way to qualify an inversion workflow:

  * generate data from KNOWN parameters,
  * confirm the optimizer recovers identifiable parameters within tolerance,
  * confirm the statistics (RMSE->0, R^2->1 at the truth) are correct,
  * confirm identifiability diagnostics correctly FLAG non-identifiable
    parameters (a real, honest capability -- not every parameter is
    recoverable from every dataset),
  * confirm both least_squares and differential_evolution work,
  * confirm global (Sobol) sensitivity ranks an influential parameter above a
    non-influential one.

These tests use a small, fast experiment so the suite runs in reasonable time.
===============================================================================
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from calibration.experiment import ASPExperiment, observed_from_arrays
from calibration.calibration import CalibrationParameters
from calibration.history_matching import ForwardModel, HistoryMatcher
from calibration.uncertainty import SobolSensitivity, BayesianCalibration


def _fast_experiment():
    """A small ASP experiment that runs quickly (for repeated calibration)."""
    return ASPExperiment(nx=16, ny=8, nz=8, n_pore_volumes=2.0, courant=2.0,
                         enable_surfactant_adsorption=True,
                         enable_polymer_retention=True,
                         enable_damage=True,
                         output_signals=("dp_history", "k_ratio_history"))


def _synthetic_observed(exp, truth, signals=("dp_history", "k_ratio_history")):
    builder = exp.make_builder()
    sim = builder(truth)
    return observed_from_arrays({s: (*sim[s], 1.0) for s in signals}), builder


# ===========================================================================
def test_zero_residual_at_truth():
    """
    The residual (and RMSE) must be ~0 and R^2 ~1 when the model is evaluated at
    the TRUE parameters -- the basic sanity check that the objective machinery
    and signal extraction are consistent.
    """
    exp = _fast_experiment()
    truth = {"rrf_max": 5.0, "perm_exponent": 3.0}
    obs, builder = _synthetic_observed(exp, truth)

    p = CalibrationParameters()
    p.add("rrf_max", 5.0, 1.0, 10.0)
    p.add("perm_exponent", 3.0, 1.0, 6.0)
    fm = ForwardModel(builder)
    hm = HistoryMatcher(fm, p, obs, extractors={})
    r = hm._residual_vector(truth)
    ssr = float(np.sum(r ** 2))
    assert ssr < 1e-12, f"residual at truth not zero: {ssr:.2e}"
    print(f"PASS  zero residual at truth: SSR={ssr:.2e} (objective machinery consistent)")


def test_recover_identifiable_parameter():
    """
    Recover an IDENTIFIABLE parameter (RRF, which strongly and uniquely controls
    the permeability/dP plateau) from a perturbed initial guess, via
    least_squares. Recovery should be within a few percent and R^2 ~ 1.
    """
    exp = _fast_experiment()
    truth = {"rrf_max": 5.0}
    obs, builder = _synthetic_observed(exp, truth)

    p = CalibrationParameters()
    p.add("rrf_max", 2.5, 1.0, 10.0)        # perturbed from truth 5.0
    fm = ForwardModel(builder)
    hm = HistoryMatcher(fm, p, obs, extractors={})
    res = hm.run_least_squares(max_nfev=30)

    err = abs(res.best_values["rrf_max"] - 5.0) / 5.0
    assert res.r2_total > 0.99, f"poor fit R2={res.r2_total:.4f}"
    assert err < 0.05, f"RRF not recovered: {res.best_values['rrf_max']:.3f} (err {err:.2%})"
    print(f"PASS  identifiable recovery: RRF {res.best_values['rrf_max']:.3f} "
          f"(truth 5.0, err {err:.1%}), R2={res.r2_total:.4f}")


def test_statistics_and_confidence_intervals():
    """
    After a least_squares match, statistics must be populated: RMSE>=0, R^2<=1,
    a confidence interval for each parameter, and a finite condition number.
    """
    exp = _fast_experiment()
    truth = {"rrf_max": 4.0}
    obs, builder = _synthetic_observed(exp, truth)
    p = CalibrationParameters(); p.add("rrf_max", 3.0, 1.0, 10.0)
    hm = HistoryMatcher(ForwardModel(builder), p, obs, extractors={})
    res = hm.run_least_squares(max_nfev=25)
    assert res.rmse_total >= 0 and res.r2_total <= 1.0 + 1e-9
    assert "rrf_max" in res.conf_int, "no confidence interval produced"
    lo, hi = res.conf_int["rrf_max"]
    assert lo <= res.best_values["rrf_max"] <= hi, "CI does not bracket estimate"
    assert np.isfinite(res.condition_number), "no condition number"
    print(f"PASS  statistics: R2={res.r2_total:.4f}, RMSE={res.rmse_total:.3e}, "
          f"RRF 95% CI=[{lo:.2f},{hi:.2f}], cond={res.condition_number:.1e}")


def test_identifiability_flagged_for_correlated_pair():
    """
    HONEST diagnostic test: q_max and K_L are individually poorly identifiable
    from a near-saturated effluent (only their product/effect is constrained).
    The engine must FLAG this -- via a large identifiability metric (relative
    std) and/or high parameter correlation -- rather than silently returning a
    confident-but-wrong estimate.
    """
    exp = _fast_experiment()
    truth = {"q_max": 2.0e-4, "K_L": 5.0}
    obs, builder = _synthetic_observed(exp, truth,
                                       signals=("dp_history", "k_ratio_history"))
    p = CalibrationParameters()
    p.add("q_max", 1.5e-4, 5e-5, 5e-4, log_scale=True)
    p.add("K_L", 3.0, 0.5, 20.0)
    hm = HistoryMatcher(ForwardModel(builder), p, obs, extractors={})
    res = hm.run_least_squares(max_nfev=30)
    # At least one of the pair should be flagged weak/poor (rel std large) OR
    # they should be strongly correlated -- the signature of non-identifiability.
    ident = res.identifiability
    weak = any(v > 0.5 for v in ident.values())
    high_corr = (res.correlation is not None
                 and abs(res.correlation[0, 1]) > 0.8)
    assert weak or high_corr, (f"non-identifiability not flagged: "
                               f"ident={ident}, corr={res.correlation}")
    print(f"PASS  identifiability flagged: ident={ {k: round(v,2) for k,v in ident.items()} }, "
          f"corr={res.correlation[0,1]:.2f} (engine honestly reports weak constraint)")


def test_differential_evolution_global():
    """
    differential_evolution recovers an identifiable parameter from a POOR guess
    without gradients -- the global-optimizer path works.
    """
    exp = _fast_experiment()
    truth = {"rrf_max": 6.0}
    obs, builder = _synthetic_observed(exp, truth)
    p = CalibrationParameters(); p.add("rrf_max", 2.0, 1.0, 10.0)
    hm = HistoryMatcher(ForwardModel(builder), p, obs, extractors={})
    res = hm.run_differential_evolution(maxiter=8, popsize=6, seed=1, polish=True)
    err = abs(res.best_values["rrf_max"] - 6.0) / 6.0
    assert err < 0.08, f"DE failed to recover RRF: {res.best_values['rrf_max']:.3f}"
    print(f"PASS  differential_evolution: RRF {res.best_values['rrf_max']:.3f} "
          f"(truth 6.0, err {err:.1%}), R2={res.r2_total:.4f}")


def test_convergence_history_decreases():
    """The best-so-far objective must be monotonically non-increasing."""
    exp = _fast_experiment()
    truth = {"rrf_max": 5.0}
    obs, builder = _synthetic_observed(exp, truth)
    p = CalibrationParameters(); p.add("rrf_max", 2.0, 1.0, 10.0)
    hm = HistoryMatcher(ForwardModel(builder), p, obs, extractors={})
    hm.run_least_squares(max_nfev=20)
    conv = hm.convergence_history
    assert np.all(np.diff(conv) <= 1e-12), "convergence not monotone"
    assert conv[-1] < conv[0], "no improvement over the run"
    print(f"PASS  convergence: best objective {conv[0]:.3e} -> {conv[-1]:.3e} "
          f"(monotone decreasing)")


def test_sobol_global_sensitivity_ranking():
    """
    Sobol total-effect indices must rank an influential parameter above a
    non-influential one for a known analytic QoI.
    """
    p = CalibrationParameters()
    p.add("influential", 1.0, 0.0, 1.0)
    p.add("noninfluential", 1.0, 0.0, 1.0)
    # QoI depends strongly on 'influential', not at all on the other.
    def qoi(d):
        return 3.0 * d["influential"] ** 2 + 0.0 * d["noninfluential"]
    sob = SobolSensitivity(p, qoi, n_base=128, seed=0).compute()
    assert sob["order"][0] == "influential", f"Sobol ranking wrong: {sob['order']}"
    assert sob["ST"]["influential"] > sob["ST"]["noninfluential"] + 0.3
    assert abs(sob["ST"]["noninfluential"]) < 0.1
    print(f"PASS  Sobol sensitivity: ST(influential)={sob['ST']['influential']:.2f} "
          f">> ST(noninfluential)={sob['ST']['noninfluential']:.2f}")


def test_bayesian_posterior_brackets_truth():
    """
    A short MCMC chain on an identifiable parameter must produce a posterior
    whose 95% credible interval brackets the true value.
    """
    exp = _fast_experiment()
    truth = {"rrf_max": 5.0}
    obs, builder = _synthetic_observed(exp, truth)
    p = CalibrationParameters(); p.add("rrf_max", 4.0, 1.0, 10.0)
    hm = HistoryMatcher(ForwardModel(builder), p, obs, extractors={})
    # Small noise level so the likelihood is informative.
    bayes = BayesianCalibration(hm, sigma=50.0, step=0.08, seed=2)
    out = bayes.sample(n_samples=120, burn_in=40)
    lo, hi = out["posterior"]["rrf_max"]["ci95"]
    assert lo <= 5.0 <= hi, f"posterior CI [{lo:.2f},{hi:.2f}] excludes truth 5.0"
    print(f"PASS  Bayesian posterior: RRF 95% CI [{lo:.2f},{hi:.2f}] brackets truth 5.0 "
          f"(acceptance {out['acceptance']:.2f})")


# ===========================================================================
def _run_all():
    tests = [
        test_zero_residual_at_truth,
        test_recover_identifiable_parameter,
        test_statistics_and_confidence_intervals,
        test_identifiability_flagged_for_correlated_pair,
        test_differential_evolution_global,
        test_convergence_history_decreases,
        test_sobol_global_sensitivity_ranking,
        test_bayesian_posterior_brackets_truth,
    ]
    print("=" * 70)
    print("ASPiRe-3D  Phase 6 : history matching & calibration validation")
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
