"""
ASPiRe-3D : main/workflow.py
===============================================================================
Workflow controller: the reproducible, config-driven execution pipeline.

Orchestrates the validated physics, calibration, sensitivity, and visualization
modules into a single ordered flow, with structured logging, robust exception
handling, and concise console output. Adds NO new physics -- it wires together
the existing, validated components under the control of a Config object.

EXECUTION FLOW
--------------
  1. load configuration            (done by caller; Config passed in)
  2. validate inputs               (Config already validated; re-checked)
  3. load experimental/core data   (optional; synthetic-truth if requested)
  4. initialize physics models     (build the ASPExperiment forward model)
  5. run simulation                (baseline forward run)
  6. run calibration               (if calibration.enabled)
  7. sensitivity analysis          (if sensitivity.enabled)
  8. generate plots                (if output.save_figures)
  9. export outputs / reports      (if output.save_report)
 10. save logs

Each step is a method; run() executes them in order, logging start/finish and
trapping exceptions so a failure in one optional stage (e.g. plotting) does not
discard completed results.
===============================================================================
"""

import os
import time
import logging
import numpy as np

from calibration.experiment import ASPExperiment, observed_from_arrays
from calibration.calibration import CalibrationParameters
from calibration.history_matching import ForwardModel, HistoryMatcher
from calibration.uncertainty import SobolSensitivity
from calibration.calibration import SensitivityAnalysis


# ---------------------------------------------------------------------------
def _make_logger(name, logfile, verbose=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(logfile, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(fh)
    if verbose:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("  %(message)s"))
        logger.addHandler(ch)
    return logger


# ===========================================================================
class Workflow:
    """Config-driven execution controller for ASPiRe-3D."""

    def __init__(self, config, verbose=True):
        self.cfg = config
        self.verbose = verbose
        self.outdir = config.get("output", "outdir", "experiments/outputs")
        self.prefix = config.get("output", "prefix", "run")
        os.makedirs(self.outdir, exist_ok=True)
        self.logfile = os.path.join(self.outdir, f"{self.prefix}_log.txt")
        self.log = _make_logger(f"aspire.{self.prefix}", self.logfile, verbose)

        # result holders
        self.experiment = None
        self.builder = None
        self.sim = None
        self.observed = None
        self.calibration_result = None
        self.sensitivity_result = None
        self.history_matcher = None
        self.figures = []
        self.report_path = None
        self._t0 = None

    # -----------------------------------------------------------------------
    def run(self):
        """Execute the full ordered pipeline. Returns a result dict."""
        self._t0 = time.time()
        self.log.info("=" * 60)
        self.log.info(f"ASPiRe-3D workflow: {self.cfg.raw.get('name', 'unnamed')}")
        self.log.info("=" * 60)
        steps = [
            ("validate inputs", self.step_validate),
            ("load data", self.step_load_data),
            ("initialize physics", self.step_init_physics),
            ("run simulation", self.step_run_simulation),
            ("calibration", self.step_calibration),
            ("sensitivity", self.step_sensitivity),
            ("generate plots", self.step_plots),
            ("export report", self.step_report),
        ]
        for name, fn in steps:
            try:
                self.log.info(f"[step] {name} ...")
                fn()
            except Exception as e:           # robust: log and continue/abort
                self.log.exception(f"[step] {name} FAILED: {e}")
                # Physics/init failures are fatal; optional stages are not.
                if name in ("validate inputs", "initialize physics",
                            "run simulation"):
                    self.log.error("fatal step failed; aborting workflow")
                    raise
        dt = time.time() - self._t0
        self.log.info(f"workflow complete in {dt:.1f}s; log -> {self.logfile}")
        return self.results()

    # -----------------------------------------------------------------------
    def step_validate(self):
        """Re-affirm the config is internally consistent (already validated)."""
        c = self.cfg
        self.log.debug(c.summary())
        # Light sanity: grid not absurdly large for a desktop run.
        nx, ny, nz = c.grid
        if nx * ny * nz > 400000:
            self.log.warning(f"large grid {nx}x{ny}x{nz}; run may be slow")

    # -----------------------------------------------------------------------
    def step_load_data(self):
        """
        Load observed data for calibration. If calibration is enabled with a
        'synthetic_truth' block, generate synthetic observations from the
        forward model at those parameters (validated synthetic-truth mode). Real
        data loading is delegated to the data package (config 'data' block).
        """
        cal = self.cfg.section("calibration")
        if not cal.get("enabled"):
            self.log.info("  no calibration; skipping data load")
            return
        # Build the experiment early so we can synthesize if requested.
        self._ensure_experiment()
        truth = cal.get("synthetic_truth")
        if truth:
            self.log.info(f"  synthetic-truth observations from {truth}")
            sig = self.builder(truth)
            self.observed = observed_from_arrays(
                {s: (*sig[s], 1.0) for s in sig})
        else:
            # Real-data hook: expect a 'data' config block (see data package).
            data_cfg = self.cfg.section("data")
            if not data_cfg:
                raise ValueError("calibration enabled without 'synthetic_truth' "
                                 "or a 'data' block to load observations")
            from data.experimental_validation import build_observed_from_config
            base = data_cfg.get("base_dir", ".")
            self.observed, *_ = build_observed_from_config(data_cfg, base_dir=base)
            self.log.info(f"  loaded {len(self.observed.signals)} observed signals")

    # -----------------------------------------------------------------------
    def _ensure_experiment(self):
        if self.experiment is not None:
            return
        c = self.cfg
        phys = c.section("physics")
        params = c.section("parameters")
        # Decide which signals to output (those a calibration/plots would need).
        out_signals = ("dp_history", "k_ratio_history", "surfactant_effluent",
                       "injectivity_history")
        self.experiment = ASPExperiment(
            core_length=c.core_length_m, core_diameter=c.core_diameter_m,
            porosity=c.porosity, perm_mD=c.permeability_mD,
            rate_ml_min=c.flow_rate_ml_min, n_pore_volumes=c.pore_volumes,
            courant=c.get("numerics", "courant", 1.0),
            dispersivity=c.get("numerics", "dispersivity_m", 5e-4),
            reflow_tol=c.get("numerics", "reflow_tolerance", 0.03),
            nx=c.grid[0], ny=c.grid[1], nz=c.grid[2],
            inlet=dict(salinity=c.get("asp", "salinity", 0.3),
                       alkali=c.get("asp", "alkali", 1.0),
                       surfactant=c.get("asp", "surfactant", 1.0),
                       polymer=c.get("asp", "polymer", 1.0)),
            initial=dict(salinity=c.get("asp", "initial_salinity", 1.0)),
            enable_surfactant_adsorption=phys.get("enable_surfactant_adsorption", True),
            enable_polymer_retention=phys.get("enable_polymer_retention", True),
            enable_precipitation=phys.get("enable_precipitation", False),
            enable_fines=phys.get("enable_fines", False),
            enable_damage=phys.get("enable_damage", True),
            output_signals=out_signals)
        self._base_params = dict(params)
        self.builder = lambda p: self.experiment.extract_signals(
            self.experiment.run({**self._base_params, **p}))

    def step_init_physics(self):
        """Initialize the forward model (geometry/rate/physics from config)."""
        self._ensure_experiment()
        self.log.info(f"  experiment grid {self.cfg.grid}, "
                      f"{self.cfg.pore_volumes} PV; physics toggles applied")

    # -----------------------------------------------------------------------
    def step_run_simulation(self):
        """Baseline forward run at the config's nominal parameters."""
        self.sim = self.experiment.run(self._base_params)
        inj = self.sim.history["injectivity_ratio"]
        kr = self.sim.history["k_ratio_min"]
        self.log.info(f"  simulation done: {len(self.sim.history['pv'])} steps, "
                      f"final I/I0={inj[-1]:.3f}, k/k0(min)={kr[-1]:.3f}")

    # -----------------------------------------------------------------------
    def _build_calib_parameters(self, spec_list):
        p = CalibrationParameters()
        for s in spec_list:
            p.add(s["name"], s["init"], s["min"], s["max"],
                  log_scale=bool(s.get("log", False)))
        return p

    def step_calibration(self):
        """Run history matching if enabled."""
        cal = self.cfg.section("calibration")
        if not cal.get("enabled"):
            self.log.info("  calibration disabled; skipping")
            return
        if self.observed is None:
            raise ValueError("calibration enabled but no observed data loaded")
        params = self._build_calib_parameters(cal["parameters"])
        fm = ForwardModel(self.builder)
        self.history_matcher = HistoryMatcher(fm, params, self.observed,
                                              extractors={})
        method = cal.get("method", "least_squares")
        self.log.info(f"  calibrating ({method}) on "
                      f"{len(self.observed.signals)} signals ...")
        if method == "differential_evolution":
            self.calibration_result = self.history_matcher.run_differential_evolution(
                maxiter=cal.get("maxiter", 20), popsize=cal.get("popsize", 8))
        else:
            self.calibration_result = self.history_matcher.run_least_squares(
                max_nfev=cal.get("max_nfev", 60))
        r = self.calibration_result
        self.log.info(f"  calibration: R2={r.r2_total:.4f}, "
                      f"RMSE={r.rmse_total:.3e}, {r.n_eval} runs")

    # -----------------------------------------------------------------------
    def step_sensitivity(self):
        """Run sensitivity analysis if enabled."""
        sen = self.cfg.section("sensitivity")
        if not sen.get("enabled"):
            self.log.info("  sensitivity disabled; skipping")
            return
        spec = sen.get("parameters") or self.cfg.section("calibration").get("parameters")
        if not spec:
            raise ValueError("sensitivity enabled but no parameters specified")
        params = self._build_calib_parameters(spec)

        def qoi(pd):
            sim = self.experiment.run({**self._base_params, **pd})
            h = sim.history["injectivity_ratio"]
            return h[-1] if h else 1.0

        if sen.get("type", "sobol") == "sobol":
            self.sensitivity_result = SobolSensitivity(
                params, qoi, n_base=sen.get("n_base", 16), seed=0).compute()
            self.log.info(f"  Sobol sensitivity order: "
                          f"{self.sensitivity_result['order']}")
        else:
            self.sensitivity_result = SensitivityAnalysis(params, qoi).compute()
            self.log.info(f"  local sensitivity: {self.sensitivity_result}")

    # -----------------------------------------------------------------------
    def step_plots(self):
        """Generate figures if enabled."""
        if not self.cfg.get("output", "save_figures", True):
            self.log.info("  figures disabled; skipping")
            return
        from visualization import damage_plots as dmg
        from visualization import calibration_plots as cpl
        figdir = self.outdir
        # baseline injectivity/permeability figures
        if self.sim is not None:
            try:
                self.figures.append(dmg.plot_injectivity_decline(
                    self.sim, os.path.join(figdir, f"{self.prefix}_injectivity.png")))
                self.figures.append(dmg.plot_permeability_porosity_evolution(
                    self.sim, os.path.join(figdir, f"{self.prefix}_perm_poro.png")))
            except Exception as e:
                self.log.warning(f"  baseline plots skipped: {e}")
        # calibration figures
        if self.calibration_result is not None:
            try:
                self.figures.append(cpl.plot_observed_vs_simulated(
                    self.calibration_result,
                    os.path.join(figdir, f"{self.prefix}_obs_vs_sim.png")))
                self.figures.append(cpl.plot_convergence(
                    self.history_matcher,
                    os.path.join(figdir, f"{self.prefix}_convergence.png")))
            except Exception as e:
                self.log.warning(f"  calibration plots skipped: {e}")
        # sensitivity tornado
        if self.sensitivity_result is not None:
            try:
                self.figures.append(cpl.plot_tornado(
                    self.sensitivity_result,
                    os.path.join(figdir, f"{self.prefix}_tornado.png")))
            except Exception as e:
                self.log.warning(f"  tornado plot skipped: {e}")
        self.log.info(f"  {len(self.figures)} figures written")

    # -----------------------------------------------------------------------
    def step_report(self):
        """Export a Markdown run report if enabled."""
        if not self.cfg.get("output", "save_report", True):
            self.log.info("  report disabled; skipping")
            return
        lines = [f"# ASPiRe-3D Run Report: {self.cfg.raw.get('name','unnamed')}\n",
                 "_Auto-generated by the workflow controller._\n",
                 "## Configuration\n", "```", self.cfg.summary(), "```\n"]
        if self.sim is not None:
            inj = self.sim.history["injectivity_ratio"][-1]
            kr = self.sim.history["k_ratio_min"][-1]
            lines += ["## Baseline simulation\n",
                      f"- final injectivity ratio I/I0 = **{inj:.3f}**",
                      f"- minimum permeability ratio k/k0 = **{kr:.3f}**\n"]
        if self.calibration_result is not None:
            r = self.calibration_result
            lines += ["## Calibration\n",
                      f"- method: {r.method}, {r.n_eval} forward runs",
                      f"- aggregate R² = **{r.r2_total:.4f}**, "
                      f"RMSE = **{r.rmse_total:.4e}**\n",
                      "| Parameter | Estimate | 95% CI | Identifiability |",
                      "|---|---|---|---|"]
            for n in r.parameters.names:
                v = r.best_values[n]; ci = r.conf_int.get(n)
                ci_s = f"[{ci[0]:.3e}, {ci[1]:.3e}]" if ci else "—"
                ident = r.identifiability.get(n, float("nan"))
                flag = ("well" if ident < 0.5 else "weak" if ident < 2 else "POOR")
                lines.append(f"| {n} | {v:.4e} | {ci_s} | {flag} |")
            lines.append("")
        if self.sensitivity_result is not None and "ST" in self.sensitivity_result:
            s = self.sensitivity_result
            lines += ["## Global sensitivity (Sobol)\n",
                      "| Parameter | ST | S1 |", "|---|---|---|"]
            for n in s["order"]:
                lines.append(f"| {n} | {s['ST'][n]:.3f} | {s['S1'][n]:.3f} |")
            lines.append("")
        if self.figures:
            lines += ["## Figures\n"] + [f"- `{os.path.basename(f)}`"
                                         for f in self.figures]
        report = "\n".join(lines)
        self.report_path = os.path.join(self.outdir, f"{self.prefix}_report.md")
        with open(self.report_path, "w") as f:
            f.write(report)
        self.log.info(f"  report -> {self.report_path}")

    # -----------------------------------------------------------------------
    def results(self):
        return {
            "config": self.cfg,
            "simulation": self.sim,
            "calibration": self.calibration_result,
            "sensitivity": self.sensitivity_result,
            "figures": self.figures,
            "report": self.report_path,
            "logfile": self.logfile,
        }
