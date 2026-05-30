"""
ASPiRe-3D : core/history_matching.py
===============================================================================
Optimizer-driven HISTORY MATCHING / CALIBRATION engine.

This turns ASPiRe-3D into an experimentally-calibratable simulator: given
observed core-flood signals (differential-pressure history, produced/effluent
concentration curves, permeability-impairment data) it adjusts physical
parameters (adsorption, retention, RRF, kinetics, damage coefficients) to
reproduce them quantitatively.

ARCHITECTURE
------------
  ForwardModel       : parameters (dict) -> simulated signals, via a user-
                       supplied builder that constructs+runs a CoupledSimulator.
                       This is the only simulator-specific glue; everything else
                       is generic.
  HistoryMatcher     : wraps scipy.optimize.least_squares (local, gradient,
                       gives a Jacobian -> covariance/confidence intervals) and
                       differential_evolution (global, gradient-free, robust to
                       multimodal misfit). Returns a CalibrationResult.
  CalibrationResult  : best parameters, fit statistics (RMSE, R^2 per signal and
                       aggregate), linearized parameter covariance, confidence
                       intervals, correlation matrix, identifiability metrics.

The misfit is the WEIGHTED, NORMALISED residual vector (concatenated across
signals) so least_squares can exploit its structure; differential_evolution
uses the scalar sum-of-squares of the same vector.

SCIENTIFIC INTEGRITY NOTE
-------------------------
No experimental data is bundled or fabricated. The engine operates on whatever
ObservedData the user supplies. Its correctness is validated by SYNTHETIC-TRUTH
RECOVERY (tests/test_phase6.py): generate data from known parameters and confirm
the optimizer recovers them within confidence intervals -- the standard way to
qualify an inversion workflow before applying it to real laboratory data.
===============================================================================
"""

import numpy as np
from scipy.optimize import least_squares, differential_evolution


# ===========================================================================
class ForwardModel:
    """
    Wraps a user 'builder' that maps a parameter dict to simulated signals.

    builder(param_dict) -> dict signal_name -> (times, values)

    The builder constructs a fresh CoupledSimulator with the given parameters,
    runs it, and extracts the signals to be matched. Keeping it as a callable
    makes the engine independent of which physics/parameters are calibrated.
    """

    def __init__(self, builder):
        self.builder = builder
        self.n_evaluations = 0

    def simulate(self, param_dict):
        self.n_evaluations += 1
        return self.builder(param_dict)


# ===========================================================================
class CalibrationResult:
    """Holds the outcome of a history match and derived statistics."""

    def __init__(self, parameters, best_values, observed, simulated_signals,
                 residual_vector, success, message, method,
                 jacobian=None, n_eval=0):
        self.parameters = parameters          # CalibrationParameters (at optimum)
        self.best_values = best_values        # dict name->value
        self.observed = observed
        self.simulated = simulated_signals     # dict name->(t,y) at optimum
        self.residual = residual_vector
        self.success = success
        self.message = message
        self.method = method
        self.jacobian = jacobian
        self.n_eval = n_eval
        self._compute_statistics()

    # -----------------------------------------------------------------------
    def _compute_statistics(self):
        """Per-signal RMSE and R^2, plus aggregate; covariance & CIs if Jacobian."""
        self.rmse = {}
        self.r2 = {}
        all_obs, all_sim = [], []
        for name, obs in self.observed.signals.items():
            if name not in self.simulated:
                continue
            t_sim, y_sim = self.simulated[name]
            y_interp = np.interp(obs["t"], t_sim, y_sim)
            resid = y_interp - obs["y"]
            self.rmse[name] = float(np.sqrt(np.mean(resid ** 2)))
            ss_res = float(np.sum(resid ** 2))
            ss_tot = float(np.sum((obs["y"] - np.mean(obs["y"])) ** 2)) + 1e-30
            self.r2[name] = 1.0 - ss_res / ss_tot
            all_obs.append(obs["y"]); all_sim.append(y_interp)
        if all_obs:
            o = np.concatenate(all_obs); s = np.concatenate(all_sim)
            self.rmse_total = float(np.sqrt(np.mean((s - o) ** 2)))
            ss_res = float(np.sum((s - o) ** 2))
            ss_tot = float(np.sum((o - np.mean(o)) ** 2)) + 1e-30
            self.r2_total = 1.0 - ss_res / ss_tot
        else:
            self.rmse_total = np.nan; self.r2_total = np.nan

        # Linearized covariance from the Jacobian (least_squares only):
        #   cov = sigma^2 (J^T J)^-1 ,  sigma^2 = SS_res / (m - p)
        self.covariance = None
        self.conf_int = {}
        self.correlation = None
        self.identifiability = {}
        if self.jacobian is not None:
            J = self.jacobian
            m, p = J.shape
            dof = max(1, m - p)
            ss_res = float(np.sum(self.residual ** 2))
            sigma2 = ss_res / dof
            JTJ = J.T @ J
            try:
                cov = sigma2 * np.linalg.inv(JTJ)
                self.covariance = cov
                names = self.parameters.names
                std = np.sqrt(np.clip(np.diag(cov), 0, None))
                for i, n in enumerate(names):
                    val = self.best_values[n]
                    self.conf_int[n] = (val - 1.96 * std[i],
                                        val + 1.96 * std[i])
                # Correlation matrix.
                d = np.sqrt(np.clip(np.diag(cov), 1e-30, None))
                self.correlation = cov / np.outer(d, d)
                # Identifiability: relative std (CV) and condition number.
                self.identifiability = {
                    n: float(std[i] / (abs(self.best_values[n]) + 1e-30))
                    for i, n in enumerate(names)}
                self.condition_number = float(np.linalg.cond(JTJ))
            except np.linalg.LinAlgError:
                self.condition_number = np.inf

    # -----------------------------------------------------------------------
    def summary(self):
        lines = ["=" * 62, "ASPiRe-3D  CALIBRATION RESULT", "=" * 62,
                 f"method        : {self.method}",
                 f"success       : {self.success}  ({self.message})",
                 f"evaluations   : {self.n_eval}",
                 f"aggregate RMSE: {self.rmse_total:.4e}",
                 f"aggregate R^2 : {self.r2_total:.4f}", "",
                 "per-signal fit:"]
        for name in self.rmse:
            lines.append(f"  {name:22s} RMSE={self.rmse[name]:.4e}  "
                         f"R^2={self.r2[name]:.4f}")
        lines.append("")
        lines.append("calibrated parameters:")
        for n in self.parameters.names:
            v = self.best_values[n]
            if n in self.conf_int:
                lo, hi = self.conf_int[n]
                ident = self.identifiability.get(n, np.nan)
                flag = "well-id" if ident < 0.5 else ("weak" if ident < 2 else "POOR")
                lines.append(f"  {n:22s} = {v:.4e}  95% CI [{lo:.3e}, {hi:.3e}]  ({flag})")
            else:
                lines.append(f"  {n:22s} = {v:.4e}")
        if hasattr(self, "condition_number"):
            lines.append("")
            lines.append(f"Jacobian J^T J condition number: {self.condition_number:.2e} "
                         f"({'identifiable' if self.condition_number < 1e8 else 'ILL-CONDITIONED'})")
        lines.append("=" * 62)
        return "\n".join(lines)


# ===========================================================================
class HistoryMatcher:
    """
    Drives a ForwardModel against ObservedData by adjusting CalibrationParameters
    to minimise the weighted, normalised residual.
    """

    def __init__(self, forward_model, parameters, observed, extractors,
                 normalise=True, verbose=False):
        """
        Parameters
        ----------
        forward_model : ForwardModel
        parameters : CalibrationParameters  (defines what is adjusted & bounds)
        observed : ObservedData
        extractors : dict signal_name -> callable(simulated_signals) -> (t, y)
            Usually identity (the builder already returns (t,y) per signal), but
            kept for flexibility (e.g. derive injectivity from dP).
        """
        self.fm = forward_model
        self.params = parameters
        self.observed = observed
        self.extractors = extractors
        self.normalise = normalise
        self.verbose = verbose
        self._history = []        # objective value per evaluation
        self._scales = self._signal_scales()

    def _signal_scales(self):
        scales = {}
        for name, obs in self.observed.signals.items():
            scales[name] = (np.sqrt(np.mean(obs["y"] ** 2)) + 1e-30
                            if self.normalise else 1.0)
        return scales

    # -----------------------------------------------------------------------
    def _residual_vector(self, param_dict):
        """Concatenated weighted/normalised residuals across all signals."""
        sim = self.fm.simulate(param_dict)
        parts = []
        for name, obs in self.observed.signals.items():
            extractor = self.extractors.get(name)
            t_sim, y_sim = extractor(sim) if extractor else sim[name]
            y_interp = np.interp(obs["t"], t_sim, y_sim)
            w = np.sqrt(obs["weight"]) / self._scales[name]
            parts.append(w * (y_interp - obs["y"]))
        r = np.concatenate(parts) if parts else np.array([0.0])
        self._history.append(float(np.sum(r ** 2)))
        if self.verbose:
            print(f"  eval {len(self._history):4d}  SSR={self._history[-1]:.4e}")
        return r

    def _vector_to_dict(self, x):
        self.params.from_vector(x)
        return self.params.as_dict()

    # -----------------------------------------------------------------------
    def run_least_squares(self, max_nfev=200):
        """
        Local gradient-based match (Trust Region Reflective). Returns a
        CalibrationResult including the Jacobian -> covariance/confidence
        intervals. Best when a reasonable initial guess is available.
        """
        x0 = self.params.to_vector()
        lo, hi = self.params.bounds_vector()

        def resid(x):
            return self._residual_vector(self._vector_to_dict(x))

        sol = least_squares(resid, x0, bounds=(lo, hi),
                            max_nfev=max_nfev, method="trf")
        self.params.from_vector(sol.x)
        best = self.params.as_dict()
        sim = self.fm.simulate(best)
        return CalibrationResult(self.params, best, self.observed, sim,
                                 sol.fun, sol.success, sol.message,
                                 "least_squares", jacobian=sol.jac,
                                 n_eval=self.fm.n_evaluations)

    # -----------------------------------------------------------------------
    def run_differential_evolution(self, maxiter=30, popsize=12, seed=0,
                                   tol=1e-4, polish=True):
        """
        Global gradient-free match. Robust to multimodal misfit surfaces and
        poor initial guesses, at higher cost. No Jacobian -> CIs come from a
        follow-up local refine if desired.
        """
        lo, hi = self.params.bounds_vector()
        bounds = list(zip(lo, hi))

        def scalar_obj(x):
            r = self._residual_vector(self._vector_to_dict(x))
            return float(np.sum(r ** 2))

        sol = differential_evolution(scalar_obj, bounds, maxiter=maxiter,
                                     popsize=popsize, seed=seed, tol=tol,
                                     polish=polish, init="sobol")
        self.params.from_vector(sol.x)
        best = self.params.as_dict()
        sim = self.fm.simulate(best)
        return CalibrationResult(self.params, best, self.observed, sim,
                                 self._residual_vector(best), sol.success,
                                 str(sol.message), "differential_evolution",
                                 jacobian=None, n_eval=self.fm.n_evaluations)

    # -----------------------------------------------------------------------
    @property
    def convergence_history(self):
        """Best-so-far objective value per evaluation (for convergence plots)."""
        h = np.array(self._history)
        return np.minimum.accumulate(h) if len(h) else h
