"""
ASPiRe-3D : core/interpretation.py
===============================================================================
Automated ENGINEERING INTERPRETATION of a completed coupled simulation.

Turns raw simulator output into the qualitative/quantitative assessments a
reservoir engineer would write in a core-flood report:
  * retention severity (how much chemical was lost to the rock),
  * chemical efficiency (fraction of injected chemical produced vs retained),
  * formation-damage interpretation (permeability/porosity impairment class),
  * injectivity-impairment assessment (operational severity).

These are transparent, threshold-based interpretations -- deliberately simple
and auditable, with the thresholds documented so a thesis can defend them. They
are decision SUPPORT, not black-box verdicts.
===============================================================================
"""

import numpy as np


def _classify(value, thresholds, labels):
    """Map a value to a label using ascending thresholds."""
    for t, lab in zip(thresholds, labels):
        if value <= t:
            return lab
    return labels[-1]


def chemical_efficiency(sim, dissolved_name, adsorbed_name=None):
    """
    Fraction of injected chemical that was (a) produced, (b) retained on rock,
    (c) still in solution. A low produced fraction => high chemical loss.
    """
    mesh = sim.mesh
    Q = float(np.sum(sim._inlet_flux))
    inlet_value = sim.registry.get(dissolved_name).inlet_value
    injected = inlet_value * Q * sim.time if sim.time > 0 else 0.0

    dt = np.array(sim.history["dt"])
    outC = np.array(sim.history["outlet"][dissolved_name])
    produced = float(np.sum(Q * outC * dt))

    idx = sim.registry.index(dissolved_name)
    in_solution = float(np.sum(sim.storage * sim.C[idx, :]))

    retained = 0.0
    if adsorbed_name and adsorbed_name in sim.registry.names:
        ia = sim.registry.index(adsorbed_name)
        retained = float(np.sum(mesh.cell_volume * sim.C[ia, :]))

    inj = max(injected, 1e-30)
    return {
        "injected": injected,
        "produced_fraction": produced / inj,
        "retained_fraction": retained / inj,
        "in_solution_fraction": in_solution / inj,
        "loss_to_rock_fraction": retained / inj,
    }


def retention_severity(efficiency):
    """Classify retention severity from the retained fraction."""
    f = efficiency["retained_fraction"]
    label = _classify(f, [0.05, 0.15, 0.35],
                      ["negligible", "mild", "moderate", "severe"])
    return {
        "retained_fraction": f,
        "severity": label,
        "note": ("Surfactant/polymer loss to rock reduces the chemical slug "
                 "reaching the target zone; severe retention demands higher "
                 "injected concentration or sacrificial agents."),
    }


def formation_damage_assessment(sim):
    """Classify permeability impairment from the final k/k0 (min and mean)."""
    kmin = sim.history["k_ratio_min"][-1] if sim.history["k_ratio_min"] else 1.0
    kmean = sim.history["k_ratio_mean"][-1] if sim.history["k_ratio_mean"] else 1.0
    # Damage class on (1 - k/k0).
    damage = 1.0 - kmin
    label = _classify(damage, [0.1, 0.3, 0.6],
                      ["minimal", "moderate", "significant", "severe"])
    return {
        "k_ratio_min": kmin,
        "k_ratio_mean": kmean,
        "max_local_damage": damage,
        "class": label,
        "note": ("Permeability impairment localised where k/k0 is lowest "
                 "(typically the inlet mixing/retention zone)."),
    }


def injectivity_assessment(sim):
    """Operational injectivity-impairment assessment from I/I0 history."""
    if not sim.history["injectivity_ratio"]:
        return {"final_injectivity_ratio": 1.0, "class": "none"}
    final = sim.history["injectivity_ratio"][-1]
    loss = 1.0 - final
    label = _classify(loss, [0.1, 0.3, 0.6],
                      ["acceptable", "noticeable", "problematic", "critical"])
    # Rate of decline (per PV) over the run.
    pv = np.array(sim.history["pv"]); inj = np.array(sim.history["injectivity_ratio"])
    rate = (inj[0] - inj[-1]) / max(pv[-1] - pv[0], 1e-9) if len(pv) > 1 else 0.0
    return {
        "final_injectivity_ratio": final,
        "injectivity_loss": loss,
        "decline_rate_per_PV": float(rate),
        "class": label,
        "note": ("Injectivity decline raises required injection pressure; "
                 "'critical' implies the design rate may be unsustainable."),
    }


def generate_report(sim, dissolved_name="surfactant",
                    adsorbed_name="surfactant_adsorbed"):
    """
    Assemble a full text interpretation report. Returns a formatted string
    suitable for logging or thesis appendix.
    """
    eff = chemical_efficiency(sim, dissolved_name, adsorbed_name)
    ret = retention_severity(eff)
    dmg = formation_damage_assessment(sim)
    inj = injectivity_assessment(sim)

    lines = []
    lines.append("=" * 66)
    lines.append("ASPiRe-3D  ENGINEERING INTERPRETATION REPORT")
    lines.append("=" * 66)
    lines.append(f"Simulated time     : {sim.time:.1f} s "
                 f"({sim.history['pv'][-1]:.2f} pore volumes)")
    lines.append("")
    lines.append("-- Chemical efficiency (%s) --" % dissolved_name)
    lines.append(f"  produced     : {100*eff['produced_fraction']:.1f}% of injected")
    lines.append(f"  retained     : {100*eff['retained_fraction']:.1f}% (loss to rock)")
    lines.append(f"  in solution  : {100*eff['in_solution_fraction']:.1f}%")
    lines.append("")
    lines.append("-- Retention severity --")
    lines.append(f"  classification : {ret['severity'].upper()}")
    lines.append("")
    lines.append("-- Formation damage --")
    lines.append(f"  k/k0 (min)     : {dmg['k_ratio_min']:.3f}")
    lines.append(f"  k/k0 (mean)    : {dmg['k_ratio_mean']:.3f}")
    lines.append(f"  classification : {dmg['class'].upper()}")
    lines.append("")
    lines.append("-- Injectivity impairment --")
    lines.append(f"  I/I0 (final)   : {inj['final_injectivity_ratio']:.3f}")
    lines.append(f"  decline rate   : {inj['decline_rate_per_PV']:.3f} per PV")
    lines.append(f"  classification : {inj['class'].upper()}")
    lines.append("=" * 66)
    return "\n".join(lines), {
        "efficiency": eff, "retention": ret,
        "damage": dmg, "injectivity": inj}
