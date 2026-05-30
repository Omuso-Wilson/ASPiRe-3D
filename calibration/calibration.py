"""
ASPiRe-3D : core/calibration.py
===============================================================================
Calibration / history-matching framework for ASPiRe-3D.

This prepares the simulator for quantitative matching against Niger Delta
sandstone core-flood data WITHOUT yet committing to a specific optimizer. It
provides three composable pieces:

  1. CalibrationParameters -- a typed, bounded container of the physical
     parameters a history match would adjust (adsorption capacity/affinity,
     retention, RRF, reaction rates, dispersivity, ...), with get/set as a flat
     vector so any optimizer (scipy.optimize, CMA-ES, Bayesian, ML surrogate)
     can drive it.
  2. ObjectiveFunction -- weighted least-squares misfit between simulated and
     observed core-flood signals (pressure-drop history, produced/effluent
     concentration curves, permeability-impairment data), with optional
     normalisation so disparate signals are comparably weighted.
  3. SensitivityAnalysis -- one-at-a-time (local) finite-difference sensitivity
     of the objective (or any scalar QoI) to each parameter, to rank which
     parameters matter for the match (essential before inversion).

Design intent: the OBJECTS are provided now; the optimizer and the actual field
dataset plug in later. This keeps the calibration architecture in the codebase
(so the thesis can describe a complete inversion workflow) while honestly not
fabricating data or an inversion result.
===============================================================================
"""

import numpy as np


# ===========================================================================
class CalibrationParameters:
    """
    Bounded container of calibratable parameters. Each entry: name -> (value,
    lower, upper). Exposes a flat vector interface for optimizers, with
    log-scaling for parameters that span orders of magnitude.
    """

    def __init__(self):
        self._names = []
        self._val = {}
        self._lo = {}
        self._hi = {}
        self._log = {}

    def add(self, name, value, lower, upper, log_scale=False):
        if name in self._val:
            raise ValueError(f"duplicate parameter '{name}'")
        if not (lower <= value <= upper):
            raise ValueError(f"{name}: value {value} outside [{lower},{upper}]")
        if log_scale and lower <= 0:
            raise ValueError(f"{name}: log-scaled parameter needs lower>0")
        self._names.append(name)
        self._val[name] = float(value)
        self._lo[name] = float(lower)
        self._hi[name] = float(upper)
        self._log[name] = bool(log_scale)
        return self

    @property
    def names(self):
        return list(self._names)

    def get(self, name):
        return self._val[name]

    def set(self, name, value):
        v = np.clip(value, self._lo[name], self._hi[name])
        self._val[name] = float(v)

    # ---- flat-vector interface for optimizers --------------------------
    def to_vector(self):
        """Return parameters as a flat array (log10 for log-scaled entries)."""
        return np.array([
            np.log10(self._val[n]) if self._log[n] else self._val[n]
            for n in self._names])

    def from_vector(self, x):
        """Set parameters from a flat array (inverse of to_vector), clipped."""
        for n, xi in zip(self._names, x):
            v = (10.0 ** xi) if self._log[n] else xi
            self.set(n, v)

    def bounds_vector(self):
        """Return (lower, upper) arrays in the same (possibly log) space."""
        lo = np.array([np.log10(self._lo[n]) if self._log[n] else self._lo[n]
                       for n in self._names])
        hi = np.array([np.log10(self._hi[n]) if self._log[n] else self._hi[n]
                       for n in self._names])
        return lo, hi

    def as_dict(self):
        return dict(self._val)

    def summary(self):
        lines = ["CalibrationParameters"]
        for n in self._names:
            scale = " (log)" if self._log[n] else ""
            lines.append(f"  {n:24s} = {self._val[n]:.4g}  "
                         f"in [{self._lo[n]:.3g}, {self._hi[n]:.3g}]{scale}")
        return "\n".join(lines)


# ===========================================================================
class ObservedData:
    """
    Container for observed core-flood signals to match. Each signal is a
    (times, values) pair plus a weight. Signals may have different lengths and
    units; normalisation handles the scaling.
    """

    def __init__(self):
        self.signals = {}     # name -> dict(t, y, weight)

    def add(self, name, times, values, weight=1.0):
        t = np.asarray(times, float); y = np.asarray(values, float)
        if t.shape != y.shape:
            raise ValueError(f"{name}: times/values shape mismatch")
        self.signals[name] = dict(t=t, y=y, weight=float(weight))
        return self


# ===========================================================================
class ObjectiveFunction:
    """
    Weighted, normalised least-squares misfit between simulated and observed
    signals. The simulator-output extractor is supplied as a callable so the
    objective is agnostic to which signal (dP history, effluent C, k/k0) is
    matched.

    misfit = sum_signals  weight * mean( ((sim - obs)/scale)^2 )
    """

    def __init__(self, observed, extractors, normalise=True):
        """
        Parameters
        ----------
        observed : ObservedData
        extractors : dict signal_name -> callable(sim) returning (times, values)
            interpolated/sampled onto the observed times.
        normalise : divide residuals by the observed signal's RMS scale so
            disparate signals contribute comparably.
        """
        self.observed = observed
        self.extractors = extractors
        self.normalise = normalise

    def evaluate(self, sim):
        total = 0.0
        breakdown = {}
        for name, obs in self.observed.signals.items():
            if name not in self.extractors:
                continue
            t_sim, y_sim = self.extractors[name](sim)
            # Interpolate simulated signal onto observed times.
            y_interp = np.interp(obs["t"], t_sim, y_sim)
            resid = y_interp - obs["y"]
            scale = (np.sqrt(np.mean(obs["y"] ** 2)) + 1e-30
                     if self.normalise else 1.0)
            term = obs["weight"] * float(np.mean((resid / scale) ** 2))
            breakdown[name] = term
            total += term
        return total, breakdown


# ===========================================================================
class SensitivityAnalysis:
    """
    Local one-at-a-time finite-difference sensitivity of a scalar quantity of
    interest (QoI) to each calibration parameter. Ranks parameters by influence
    -- the standard pre-inversion screening step.
    """

    def __init__(self, parameters, run_and_measure, rel_perturbation=0.05):
        """
        Parameters
        ----------
        parameters : CalibrationParameters
        run_and_measure : callable(param_dict) -> scalar QoI
            Builds and runs a simulation with the given parameters and returns
            the scalar of interest (e.g. final injectivity ratio, or objective).
        rel_perturbation : fractional bump applied to each parameter.
        """
        self.params = parameters
        self.run = run_and_measure
        self.rel = float(rel_perturbation)

    def compute(self):
        """
        Returns a dict name -> normalised sensitivity
            S = (dQoI/QoI) / (dp/p)        (dimensionless elasticity)
        computed by central differences where bounds allow.
        """
        base = self.run(self.params.as_dict())
        results = {}
        for name in self.params.names:
            p0 = self.params.get(name)
            dp = self.rel * abs(p0) if p0 != 0 else self.rel
            # forward and backward (clipped to bounds inside set()).
            d = dict(self.params.as_dict())
            d[name] = p0 + dp
            q_plus = self.run(d)
            d[name] = p0 - dp
            q_minus = self.run(d)
            dQoI = (q_plus - q_minus) / 2.0
            # normalised elasticity.
            S = (dQoI / (abs(base) + 1e-30)) / (dp / (abs(p0) + 1e-30))
            results[name] = float(S)
        # sort by |S| descending.
        return dict(sorted(results.items(), key=lambda kv: -abs(kv[1])))
