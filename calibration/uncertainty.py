"""
ASPiRe-3D : core/uncertainty.py
===============================================================================
UNCERTAINTY QUANTIFICATION for calibration:
  * BayesianCalibration  -- a dependency-light Metropolis-Hastings sampler that
    produces posterior parameter distributions and credible intervals, plus
    posterior-predictive uncertainty envelopes for the matched signals.
  * SobolSensitivity     -- variance-based global sensitivity (first-order and
    total-effect indices) via the Saltelli sampling scheme, ranking which
    parameters drive the model response over the WHOLE parameter space (unlike
    the local one-at-a-time sensitivity, which is only valid near a point).

Both are self-contained (NumPy + the simulator), so the workflow has no heavy
external dependencies -- important for reproducibility in a thesis environment.
===============================================================================
"""

import numpy as np


# ===========================================================================
class BayesianCalibration:
    """
    Metropolis-Hastings posterior sampling for calibration parameters.

    Likelihood: Gaussian on the weighted/normalised residuals with noise level
    sigma (either fixed or supplied per signal). Prior: uniform on the
    parameter bounds (so the posterior is the likelihood truncated to bounds).
    Proposal: Gaussian random walk in the (possibly log-scaled) parameter
    vector space, with a tunable step.

    This gives posterior distributions -> credible intervals and parameter
    correlations that account for the full (possibly non-Gaussian) misfit, a
    stronger uncertainty statement than the linearized Jacobian covariance.
    """

    def __init__(self, history_matcher, sigma=1.0, step=0.05, seed=0):
        self.hm = history_matcher
        self.sigma = float(sigma)
        self.step = float(step)
        self.rng = np.random.default_rng(seed)

    def _log_posterior(self, x):
        lo, hi = self.hm.params.bounds_vector()
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf                      # outside uniform prior support
        r = self.hm._residual_vector(self.hm._vector_to_dict(x))
        # Gaussian log-likelihood (residuals already weighted/normalised).
        return -0.5 * np.sum((r / self.sigma) ** 2)

    def sample(self, n_samples=400, burn_in=100, x0=None):
        """
        Run the chain. Returns dict with 'chain' (samples x params, in natural
        units), 'acceptance', and per-parameter posterior mean/std/credible
        intervals.
        """
        params = self.hm.params
        x = np.array(x0 if x0 is not None else params.to_vector(), float)
        lp = self._log_posterior(x)
        lo, hi = params.bounds_vector()
        span = (hi - lo)
        chain = []
        n_accept = 0
        total = n_samples + burn_in
        for it in range(total):
            prop = x + self.rng.normal(0, self.step, size=x.shape) * span
            lp_prop = self._log_posterior(prop)
            if np.log(self.rng.random()) < (lp_prop - lp):
                x, lp = prop, lp_prop
                if it >= burn_in:
                    n_accept += 1
            if it >= burn_in:
                # store in natural units
                params.from_vector(x)
                chain.append([params.get(n) for n in params.names])
        chain = np.array(chain)
        names = params.names
        post = {}
        for i, n in enumerate(names):
            col = chain[:, i]
            post[n] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)),
                "ci95": (float(np.percentile(col, 2.5)),
                         float(np.percentile(col, 97.5))),
            }
        return {"chain": chain, "names": names,
                "acceptance": n_accept / max(1, n_samples),
                "posterior": post}


# ===========================================================================
class SobolSensitivity:
    """
    Variance-based GLOBAL sensitivity (Saltelli scheme).

    For each parameter computes:
      S1  (first-order)   : fraction of output variance from that parameter alone
      ST  (total-effect)  : fraction including all interactions involving it
    Sum of S1 < 1 indicates interactions; large ST-S1 flags an interacting
    parameter. This ranks parameter importance over the ENTIRE bounded space,
    the rigorous companion to local sensitivity.
    """

    def __init__(self, parameters, qoi, n_base=64, seed=0):
        """
        Parameters
        ----------
        parameters : CalibrationParameters (defines variables & bounds)
        qoi : callable(param_dict) -> scalar quantity of interest
        n_base : base sample size; total model runs = n_base*(p+2).
        """
        self.params = parameters
        self.qoi = qoi
        self.n_base = int(n_base)
        self.rng = np.random.default_rng(seed)

    def _sample_unit(self, n):
        p = len(self.params.names)
        return self.rng.random((n, p))

    def _to_natural(self, unit_row):
        lo, hi = self.params.bounds_vector()
        x = lo + unit_row * (hi - lo)
        self.params.from_vector(x)
        return self.params.as_dict()

    def compute(self):
        names = self.params.names
        p = len(names)
        N = self.n_base
        A = self._sample_unit(N)
        B = self._sample_unit(N)

        def evaluate(matrix):
            return np.array([self.qoi(self._to_natural(row)) for row in matrix])

        yA = evaluate(A)
        yB = evaluate(B)
        varY = np.var(np.concatenate([yA, yB])) + 1e-30

        S1 = {}; ST = {}
        for i, name in enumerate(names):
            ABi = A.copy(); ABi[:, i] = B[:, i]
            yABi = evaluate(ABi)
            # Saltelli (2010) estimators.
            S1[name] = float(np.mean(yB * (yABi - yA)) / varY)
            ST[name] = float(0.5 * np.mean((yA - yABi) ** 2) / varY)
        # Sort by total-effect descending.
        order = sorted(names, key=lambda n: -ST[n])
        return {"S1": S1, "ST": ST, "order": order,
                "n_runs": N * (p + 2)}
