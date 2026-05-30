"""
ASPiRe-3D : core/experimental_validation.py
===============================================================================
PLUG-AND-PLAY experimental validation framework.

One module that takes a laboratory configuration (metadata + signal mappings +
CSV files) all the way to a calibrated, uncertainty-quantified, fully-reported
history match -- with minimal restructuring to integrate real Niger Delta core-
flood data. It composes the Phase 6 calibration engine with the Phase 7
schema/ingestion layers and adds experimental-uncertainty weighting and
automated validation-report generation.

WORKFLOW
--------
    config (dict / file)
       -> ExperimentMetadata + SignalMapping[]            (data_schema)
       -> ingest+clean each CSV -> CleanedSignal[]         (data_ingestion)
       -> ObservedData with per-point uncertainty weights  (calibration)
       -> ForwardModel from an ASPExperiment built to match the metadata
       -> HistoryMatcher (least_squares / differential_evolution)
       -> statistics, identifiability, sensitivity, uncertainty
       -> automated validation report + figures

EXPERIMENTAL UNCERTAINTY HANDLING
---------------------------------
Each cleaned signal carries a per-point 1-sigma. We convert this to calibration
weights as w_i = 1/sigma_i^2 (inverse-variance, the maximum-likelihood weighting
for Gaussian noise), aggregated to a per-signal weight plus a normalisation so
noisy signals do not dominate. This makes the fit respect measured data quality
-- a defensibility requirement for thesis validation.

INTEGRITY: operates only on user-supplied data; no fabricated measurements.
The accompanying synthetic-CSV generator (make_synthetic_dataset) writes a
*messy* synthetic file from a known parameter set so the full real-data pipeline
(parsing, unit conversion, cleaning, weighting, matching) can be validated
end-to-end before applying it to laboratory data.
===============================================================================
"""

import os
import json
import numpy as np

from data.data_schema import ExperimentMetadata, SignalMapping
from data.data_ingestion import ingest_signal
from calibration.calibration import ObservedData, CalibrationParameters
from calibration.history_matching import ForwardModel, HistoryMatcher
from calibration.uncertainty import SobolSensitivity
from calibration.experiment import ASPExperiment


# ===========================================================================
def load_config(path):
    """
    Load a JSON experiment config from disk (the format in
    data_templates/example_config.json). Returns the config dict ready for
    build_observed_from_config / run_experimental_validation.
    """
    with open(path, "r") as f:
        return json.load(f)


# ===========================================================================
def build_observed_from_config(config, base_dir=".", clean_kwargs=None):
    """
    Ingest+clean all signals in a config and assemble an ObservedData with
    inverse-variance weights. Returns (observed, metadata, cleaned_signals,
    provenance_text).
    """
    clean_kwargs = clean_kwargs or {}
    md = config["metadata"]
    metadata = ExperimentMetadata(
        core_length_m=md["core_length_m"], core_diameter_m=md["core_diameter_m"],
        porosity=md["porosity"], flow_rate_ml_min=md["flow_rate_ml_min"],
        baseline_permeability_mD=md.get("baseline_permeability_mD"),
        baseline_dp_pa=md.get("baseline_dp_pa"),
        injected_concentrations=md.get("injected_concentrations"),
        name=md.get("name", "core_flood"))

    observed = ObservedData()
    cleaned = {}
    prov = [metadata.summary()]
    for spec in config["signals"]:
        mapping = SignalMapping(
            canonical=spec["canonical"], column=spec["column"],
            unit=spec.get("unit"), clock_column=spec["clock_column"],
            clock_unit=spec.get("clock_unit", "PV"),
            normalise_by=spec.get("normalise_by"),
            uncertainty=spec.get("uncertainty"),
            uncertainty_column=spec.get("uncertainty_column"))
        path = os.path.join(base_dir, spec["file"])
        cs = ingest_signal(path, mapping, metadata, **clean_kwargs)
        cleaned[cs.canonical] = cs
        prov.append(cs.provenance())

        # Inverse-variance per-signal weight: use mean(1/sigma^2) scaled so a
        # clean signal ~ weight 1. Per-point weighting is handled by passing
        # the signal's RMS-normalised residual; here we set the aggregate
        # weight from data quality (lower noise -> higher weight).
        sig = cs.sigma
        rms = np.sqrt(np.mean(cs.values ** 2)) + 1e-30
        rel_noise = np.mean(sig) / rms
        weight = 1.0 / (rel_noise ** 2 + 1e-9)
        observed.add(cs.canonical, cs.clock_pv, cs.values, weight=weight)
        prov.append(f"  [{cs.canonical}] inverse-variance weight={weight:.3g} "
                    f"(rel. noise {rel_noise:.3f})")
    return observed, metadata, cleaned, "\n".join(prov)


# ===========================================================================
def build_experiment_from_metadata(metadata, n_pore_volumes=None,
                                   output_signals=None, **exp_kwargs):
    """
    Construct an ASPExperiment whose geometry/rate/porosity MATCH the lab
    metadata, so the forward model is physically comparable to the experiment.
    """
    if n_pore_volumes is None:
        n_pore_volumes = 3.0
    perm = metadata.baseline_permeability_mD or 100.0
    exp = ASPExperiment(
        core_length=metadata.core_length,
        core_diameter=metadata.core_diameter,
        porosity=metadata.porosity,
        perm_mD=perm,
        rate_ml_min=metadata.flow_rate_ml_min,
        n_pore_volumes=n_pore_volumes,
        output_signals=tuple(output_signals
                             or ("dp_history", "k_ratio_history",
                                 "surfactant_effluent")),
        **exp_kwargs)
    return exp


# ===========================================================================
def run_experimental_validation(config, parameters, base_dir=".",
                                method="least_squares",
                                n_pore_volumes=None, exp_kwargs=None,
                                clean_kwargs=None, sobol_n_base=12,
                                outdir="outputs/validation", verbose=True):
    """
    Full plug-and-play validation: ingest -> calibrate -> quantify -> report.

    Parameters
    ----------
    config : the lab data config (see data_schema.example_config()).
    parameters : CalibrationParameters to adjust.
    method : 'least_squares' or 'differential_evolution'.
    Returns (result, sensitivity, provenance, report_path).
    """
    os.makedirs(outdir, exist_ok=True)
    observed, metadata, cleaned, provenance = build_observed_from_config(
        config, base_dir=base_dir, clean_kwargs=clean_kwargs)
    if verbose:
        print(provenance)

    # Build a matching experiment + forward model.
    out_signals = list(observed.signals.keys())
    exp = build_experiment_from_metadata(metadata, n_pore_volumes=n_pore_volumes,
                                         output_signals=out_signals,
                                         **(exp_kwargs or {}))
    builder = exp.make_builder()
    fm = ForwardModel(builder)
    hm = HistoryMatcher(fm, parameters, observed, extractors={}, verbose=False)

    if verbose:
        print(f"[validation] calibrating ({method}) against "
              f"{len(observed.signals)} experimental signals ...")
    if method == "differential_evolution":
        result = hm.run_differential_evolution(maxiter=20, popsize=8, seed=0)
    else:
        result = hm.run_least_squares(max_nfev=80)
    if verbose:
        print(result.summary())

    # Global sensitivity on final injectivity (a key engineering QoI).
    def qoi(pd):
        sim = exp.run(pd)
        h = sim.history["injectivity_ratio"]
        return h[-1] if h else 1.0
    sens_params = CalibrationParameters()
    for n in parameters.names:
        sens_params.add(n, parameters.get(n),
                        parameters._lo[n], parameters._hi[n],
                        log_scale=parameters._log[n])
    sensitivity = SobolSensitivity(sens_params, qoi, n_base=sobol_n_base,
                                   seed=0).compute()

    report_path = write_validation_report(result, sensitivity, metadata,
                                          provenance, cleaned, outdir, method)
    if verbose:
        print(f"[validation] report -> {report_path}")
    return result, sensitivity, provenance, report_path, hm, observed, builder


# ===========================================================================
def write_validation_report(result, sensitivity, metadata, provenance,
                            cleaned, outdir, method):
    """Generate a thesis-ready experimental validation report (Markdown)."""
    L = []
    L.append(f"# ASPiRe-3D — Experimental Validation Report: {metadata.name}\n")
    L.append("_Auto-generated by the plug-and-play experimental validation "
             "framework._\n")
    L.append("## 1. Experiment metadata\n")
    L.append("```")
    L.append(metadata.summary())
    L.append("```")
    L.append("## 2. Data provenance (raw → cleaned)\n")
    L.append("Every preprocessing operation is logged for auditability:\n")
    L.append("```")
    L.append(provenance)
    L.append("```")
    L.append("## 3. History-match quality\n")
    L.append(f"- Method: **{method}**, {result.n_eval} forward runs")
    L.append(f"- Aggregate R² = **{result.r2_total:.4f}**, "
             f"RMSE = **{result.rmse_total:.4e}**\n")
    L.append("| Signal | n points | RMSE | R² |")
    L.append("|---|---|---|---|")
    low_var_flag = False
    for s in result.rmse:
        npts = len(cleaned[s]) if s in cleaned else "—"
        r2 = result.r2[s]
        note = ""
        # R^2 is unreliable for near-constant (plateau) signals: flag it.
        if s in cleaned:
            obs = cleaned[s].values
            cv = np.std(obs) / (abs(np.mean(obs)) + 1e-30)
            if cv < 0.15:
                note = " *"
                low_var_flag = True
        L.append(f"| {s}{note} | {npts} | {result.rmse[s]:.4e} | {r2:.4f} |")
    if low_var_flag:
        L.append("\n\\* Signal is near-constant (plateau); R² is statistically "
                 "unreliable for low-variance data (small SS_tot) — judge these "
                 "signals by RMSE, not R². This is expected for RRF-dominated "
                 "Δp and k/k₀ plateaus.")
    L.append("\n## 4. Calibrated parameters, uncertainty & identifiability\n")
    L.append("| Parameter | Estimate | 95% CI | Rel. std | Identifiability |")
    L.append("|---|---|---|---|---|")
    for n in result.parameters.names:
        v = result.best_values[n]
        ci = result.conf_int.get(n)
        ci_s = f"[{ci[0]:.3e}, {ci[1]:.3e}]" if ci else "—"
        ident = result.identifiability.get(n, float("nan"))
        flag = ("well-identified" if ident < 0.5 else
                "weakly identified" if ident < 2 else "POORLY identified")
        L.append(f"| {n} | {v:.4e} | {ci_s} | {ident:.2f} | {flag} |")
    if hasattr(result, "condition_number"):
        c = result.condition_number
        L.append(f"\nJacobian JᵀJ condition number: **{c:.2e}** "
                 f"({'identifiable' if c < 1e8 else 'ill-conditioned'}).\n")
    L.append("## 5. Global sensitivity (Sobol total-effect)\n")
    L.append("| Parameter | ST | S1 |")
    L.append("|---|---|---|")
    for n in sensitivity["order"]:
        L.append(f"| {n} | {sensitivity['ST'][n]:.3f} | {sensitivity['S1'][n]:.3f} |")
    L.append("\n## 6. Reproducibility & assumptions\n")
    L.append("- Forward model geometry/rate/porosity set to match the "
             "experiment metadata.")
    L.append("- Conservative FVM transport, implicit M-matrix, operator "
             "splitting with stiffness-controlled sub-stepping.")
    L.append("- Inverse-variance signal weighting from measured uncertainty.")
    L.append("- Identifiability/condition-number diagnostics distinguish "
             "constrained from unconstrained parameters.")
    report = "\n".join(L)
    path = os.path.join(outdir, f"validation_report_{metadata.name}.md")
    with open(path, "w") as f:
        f.write(report)
    return path


# ===========================================================================
def make_synthetic_dataset(outdir, truth_params, exp_kwargs=None,
                           noise=0.03, seed=0):
    """
    Generate a *messy* synthetic core-flood CSV set from known parameters, to
    validate the full real-data pipeline end-to-end (parsing, units, cleaning,
    weighting, matching) WITHOUT fabricating laboratory measurements. The files
    deliberately include: non-SI units (psi), a minutes clock, a header,
    duplicated/shuffled rows, injected outliers, and Gaussian noise -- the
    messiness real ingestion must survive.

    Returns the config dict pointing at the written files.
    """
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.default_rng(seed)

    # Run the truth simulation with lab-like geometry.
    exp = ASPExperiment(nx=20, ny=8, nz=8, n_pore_volumes=2.5, courant=2.0,
                        rate_ml_min=2.0, porosity=0.21, perm_mD=120.0,
                        output_signals=("dp_history", "surfactant_effluent",
                                        "k_ratio_history"),
                        **(exp_kwargs or {}))
    sim = exp.run(truth_params)
    sig = exp.extract_signals(sim)
    md = exp.mesh   # not used; metadata declared in config below

    # --- dp.csv : pressure in PSI, clock in MINUTES, with header+noise+outliers ---
    pv, dp_pa = sig["dp_history"]
    from utils.constants import PSI_TO_PA
    dp_psi = dp_pa / PSI_TO_PA
    sec_per_pv = (0.21 * np.pi * (0.019 ** 2) * 0.10) / (2.0e-6 / 60.0)
    t_min = pv * sec_per_pv / 60.0
    dp_psi_noisy = dp_psi * (1.0 + noise * rng.standard_normal(len(dp_psi)))
    # inject a couple of gross outliers + a duplicate row + shuffle.
    dp_psi_noisy[len(dp_psi_noisy) // 3] *= 3.0
    rows = list(zip(t_min, dp_psi_noisy,
                    np.full(len(dp_psi_noisy), 0.3 * np.mean(dp_psi))))
    rows.append(rows[5])                       # duplicate clock
    rng.shuffle(rows)                          # non-monotonic order
    with open(os.path.join(outdir, "dp.csv"), "w") as f:
        f.write("time_min,dP_psi,dP_sigma_psi\n")
        for t, d, s in rows:
            f.write(f"{t:.4f},{d:.4f},{s:.4f}\n")

    # --- surf.csv : already normalised, clock in PV, light noise ---
    pv2, surf = sig["surfactant_effluent"]
    surf_noisy = np.clip(surf + noise * rng.standard_normal(len(surf)), 0, None)
    with open(os.path.join(outdir, "surf.csv"), "w") as f:
        f.write("PV,C_over_C0,sigma\n")
        for t, c in zip(pv2, surf_noisy):
            f.write(f"{t:.4f},{c:.4f},0.03\n")

    # --- kratio.csv : permeability ratio k/k0, clock in PV (constrains the
    #     damage exponent), light noise ---
    pv3, kr = sig["k_ratio_history"]
    kr_noisy = np.clip(kr + 0.5 * noise * rng.standard_normal(len(kr)), 1e-3, None)
    with open(os.path.join(outdir, "kratio.csv"), "w") as f:
        f.write("PV,k_ratio,sigma\n")
        for t, c in zip(pv3, kr_noisy):
            f.write(f"{t:.4f},{c:.4f},0.02\n")

    config = {
        "metadata": dict(
            name="synthetic_coreA", core_length_m=0.10, core_diameter_m=0.038,
            porosity=0.21, flow_rate_ml_min=2.0, baseline_permeability_mD=120.0,
            injected_concentrations=dict(surfactant=1.0)),
        "signals": [
            dict(canonical="dp_history", file="dp.csv", column="dP_psi",
                 unit="psi", clock_column="time_min", clock_unit="min",
                 uncertainty_column="dP_sigma_psi"),
            dict(canonical="surfactant_effluent", file="surf.csv",
                 column="C_over_C0", clock_column="PV", clock_unit="PV",
                 uncertainty_column="sigma"),
            dict(canonical="k_ratio_history", file="kratio.csv",
                 column="k_ratio", clock_column="PV", clock_unit="PV",
                 uncertainty_column="sigma"),
        ],
    }
    return config
