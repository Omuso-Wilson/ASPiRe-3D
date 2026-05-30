"""
ASPiRe-3D : postprocessing/calibration_plots.py
===============================================================================
Visualization for history matching / calibration (Phase 6):
  * observed vs simulated overlays (per signal),
  * residual plots,
  * optimizer convergence history,
  * sensitivity tornado charts (local or Sobol),
  * posterior / uncertainty bands (from a Bayesian chain or CI envelope).
All Matplotlib (headless), saved to disk for thesis figures.
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_observed_vs_simulated(result, outfile, signal_labels=None):
    """One panel per matched signal: observed points vs simulated curve."""
    names = [n for n in result.observed.signals if n in result.simulated]
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.0 * n + 0.5), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        obs = result.observed.signals[name]
        t_sim, y_sim = result.simulated[name]
        ax.plot(t_sim, y_sim, "-", lw=2, color="firebrick", label="simulated")
        ax.plot(obs["t"], obs["y"], "o", ms=4, color="steelblue",
                label="observed", alpha=0.8)
        lab = (signal_labels or {}).get(name, name)
        ax.set_ylabel(lab)
        ax.set_title(f"{lab}  (R²={result.r2.get(name, float('nan')):.4f})",
                     fontsize=10, loc="left")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Pore volumes injected  [-]")
    fig.suptitle("Observed vs simulated (history match)", y=0.995)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_residuals(result, outfile, signal_labels=None):
    """Residual (simulated - observed) per signal vs clock."""
    names = [n for n in result.observed.signals if n in result.simulated]
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 2.6 * n + 0.5), squeeze=False)
    for ax, name in zip(axes[:, 0], names):
        obs = result.observed.signals[name]
        t_sim, y_sim = result.simulated[name]
        y_interp = np.interp(obs["t"], t_sim, y_sim)
        resid = y_interp - obs["y"]
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(obs["t"], resid, "o-", ms=3, color="darkorange")
        lab = (signal_labels or {}).get(name, name)
        ax.set_ylabel(f"resid: {lab}")
        ax.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel("Pore volumes injected  [-]")
    fig.suptitle("Residuals (simulated − observed)", y=0.995)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_convergence(history_matcher, outfile):
    """Best-so-far objective vs evaluation (log-y)."""
    conv = history_matcher.convergence_history
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.semilogy(np.arange(1, len(conv) + 1), np.maximum(conv, 1e-30),
                "-", lw=2, color="purple")
    ax.set_xlabel("Forward-model evaluation")
    ax.set_ylabel("Best objective (sum of sq. residuals)")
    ax.set_title("Calibration convergence history")
    ax.grid(True, alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_tornado(sensitivity, outfile, title="Parameter sensitivity (tornado)",
                use_total_effect=True):
    """
    Tornado chart of sensitivity magnitudes. `sensitivity` may be:
      * a Sobol result dict with 'ST'/'S1', or
      * a plain dict name->value (local sensitivities).
    """
    if isinstance(sensitivity, dict) and "ST" in sensitivity:
        data = sensitivity["ST"] if use_total_effect else sensitivity["S1"]
        xlabel = "Sobol total-effect index ST" if use_total_effect else "S1"
    else:
        data = sensitivity
        xlabel = "Sensitivity (|elasticity|)"
    items = sorted(data.items(), key=lambda kv: abs(kv[1]))
    names = [k for k, _ in items]
    vals = [abs(v) for _, v in items]
    fig, ax = plt.subplots(figsize=(7.0, 0.5 * len(names) + 1.5))
    ax.barh(names, vals, color="teal", alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile


def plot_uncertainty_band(builder, signal_name, bayes_result, observed, outfile,
                          n_draws=40, clock_label="Pore volumes injected  [-]"):
    """
    Posterior-predictive uncertainty band: run the forward model for a sample of
    posterior parameter draws and shade the 5-95% envelope of the signal, with
    the observed data overlaid.
    """
    chain = bayes_result["chain"]; names = bayes_result["names"]
    idx = np.random.default_rng(0).choice(len(chain),
                                          size=min(n_draws, len(chain)),
                                          replace=False)
    curves = []
    t_ref = None
    for i in idx:
        pd = {n: chain[i, j] for j, n in enumerate(names)}
        sig = builder(pd)
        t, y = sig[signal_name]
        if t_ref is None:
            t_ref = t
        curves.append(np.interp(t_ref, t, y))
    curves = np.array(curves)
    lo = np.percentile(curves, 5, axis=0)
    hi = np.percentile(curves, 95, axis=0)
    med = np.percentile(curves, 50, axis=0)

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.fill_between(t_ref, lo, hi, alpha=0.3, color="steelblue",
                    label="5–95% posterior")
    ax.plot(t_ref, med, "-", lw=2, color="navy", label="posterior median")
    if signal_name in observed.signals:
        obs = observed.signals[signal_name]
        ax.plot(obs["t"], obs["y"], "o", ms=4, color="firebrick",
                label="observed")
    ax.set_xlabel(clock_label); ax.set_ylabel(signal_name)
    ax.set_title(f"Posterior-predictive uncertainty: {signal_name}")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(outfile, dpi=150); plt.close(fig)
    return outfile
