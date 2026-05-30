"""
ASPiRe-3D : core/kinetics.py
===============================================================================
Reaction KINETICS for formation damage: precipitation/dissolution and fines
migration. Both subclass the ReactionModel interface from physics.species, so the
transport backbone (core.reactive_transport) consumes them WITHOUT modification.

This is deliberately KINETIC (rate-based), not equilibrium-based: per the
project plan, full geochemical equilibrium is a later phase. Kinetic rate laws
are the right first step because (a) core floods are dynamic, finite-residence-
time experiments where rates matter, and (b) kinetics slot cleanly into the
operator-splitting time integration already in place.

-------------------------------------------------------------------------------
1. PRECIPITATION / DISSOLUTION  (PrecipitationKinetics)
-------------------------------------------------------------------------------
Physical picture: ASP flooding mixes injected alkali with resident formation
brine. Where alkalinity is high and/or the brine is supersaturated (often
correlated with salinity / divalent-ion content), a mineral scale precipitates
onto the rock; where conditions reverse, it can re-dissolve. We model a lumped
"precipitate" species fed by a lumped dissolved "scaling" tendency.

We define a dimensionless SATURATION INDEX SI from the local state:

    SI = (alkali / alkali_ref) * (salinity / salinity_ref)

with SI > 1 => supersaturated (precipitation), SI < 1 => undersaturated
(dissolution). This is a transparent, monotone surrogate for the true ion-
activity product; it captures the two triggers the thesis cares about
(alkali/pH-driven and salinity-driven scaling) and is trivially replaceable by a
real IAP/Ksp ratio when equilibrium chemistry is added.

KINETIC RATE (precipitate concentration c_p, [kg_precipitate / m^3_bulk]):

    precipitation (SI>1):  R = k_p * f_T * (SI - 1)            (>=0)
    dissolution   (SI<1):  R = -k_d * f_T * (1 - SI) * H(c_p)   (<=0, reversible)

    H(c_p) = c_p/(c_p + eps)  -> dissolution only where precipitate exists,
                                 and smoothly -> 0 as c_p -> 0 (no negatives).

f_T is an optional ARRHENIUS temperature factor:

    f_T = exp( -Ea/R * (1/T - 1/T_ref) )

so higher temperature accelerates the rate (Ea>0). Set Ea=0 to disable.

MASS COUPLING (conservative): precipitate gained is removed from the dissolved
pool. We debit the lumped scaling potential from a chosen dissolved species
(default 'salinity', representing consumed scaling ions) with a stoichiometric
factor `consume_per_precipitate`. Dissolution credits it back. This keeps total
(dissolved + precipitated) scaling mass conserved -- checked in validation.

-------------------------------------------------------------------------------
2. FINES MIGRATION  (FinesMigrationKinetics)
-------------------------------------------------------------------------------
Physical picture: colloidal fines (clays) attached to pore walls detach when
the interstitial velocity exceeds a CRITICAL VELOCITY v_c (hydrodynamic /
salinity shock effect), travel in suspension, and re-deposit by straining /
attachment. Suspended fines are mobile; attached/deposited fines are immobile.

States: c_s = suspended fines [kg/m^3 fluid], sigma = deposited fines
[kg/m^3 bulk].

DETACHMENT (sigma -> c_s) when |v| > v_c:

    R_det = k_det * (|v|/v_c - 1)_+ * sigma

REMOBILIZATION threshold is the same v_c (we use the (.)_+ ramp).

DEPOSITION / ATTACHMENT (c_s -> sigma), classic filtration kinetics:

    R_dep = k_dep * |v| * c_s

Net rate on suspended:  d c_s/dt |_rxn = R_det - R_dep
Net rate on deposited:  d sigma/dt      = R_dep - R_det
(equal and opposite -> fines mass conserved; checked in validation.)

Both models return per-species dC/dt arrays of shape (n_species, n_active),
with the sign convention "positive increases concentration".
===============================================================================
"""

import numpy as np
from physics.species import ReactionModel

# Universal gas constant [J/(mol.K)] for the Arrhenius factor.
R_GAS = 8.314462618


# ===========================================================================
class PrecipitationKinetics(ReactionModel):
    """
    Salinity- and alkali-triggered precipitation with reversible dissolution
    and optional Arrhenius temperature dependence. Subclasses ReactionModel.
    """

    def __init__(self, registry,
                 dissolved_name="salinity",
                 precipitate_name="precipitate",
                 alkali_name="alkali",
                 k_precip=1.0e-3, k_dissolve=5.0e-4,
                 alkali_ref=0.5, salinity_ref=0.5,
                 consume_per_precipitate=1.0,
                 # Arrhenius (optional): set activation_energy=0 to disable.
                 activation_energy=0.0, temperature=298.15, temperature_ref=298.15,
                 eps=1.0e-6):
        self.i_diss = registry.index(dissolved_name)
        self.i_pre = registry.index(precipitate_name)
        self.i_alk = registry.index(alkali_name)
        self.k_p = float(k_precip)
        self.k_d = float(k_dissolve)
        self.alk_ref = float(alkali_ref)
        self.sal_ref = float(salinity_ref)
        self.stoich = float(consume_per_precipitate)
        self.Ea = float(activation_energy)
        self.T = float(temperature)
        self.T_ref = float(temperature_ref)
        self.eps = float(eps)

        # Precompute the constant temperature factor (T is uniform in Phase 4).
        if self.Ea > 0.0:
            self.f_T = np.exp(-self.Ea / R_GAS * (1.0 / self.T - 1.0 / self.T_ref))
        else:
            self.f_T = 1.0

    def saturation_index(self, state):
        """SI = (alkali/alk_ref)*(salinity/sal_ref), elementwise per cell."""
        alk = state[self.i_alk, :]
        sal = state[self.i_diss, :]
        return (alk / self.alk_ref) * (sal / self.sal_ref)

    def rates(self, state, registry, mesh, properties):
        r = np.zeros_like(state)
        SI = self.saturation_index(state)
        c_p = state[self.i_pre, :]

        # Precipitation where supersaturated; dissolution where undersaturated.
        precip = self.k_p * self.f_T * np.maximum(SI - 1.0, 0.0)            # >=0
        # Reversible dissolution, gated by available precipitate (smooth H).
        H = c_p / (c_p + self.eps)
        dissolve = self.k_d * self.f_T * np.maximum(1.0 - SI, 0.0) * H      # >=0

        net_precip = precip - dissolve            # d c_p/dt
        r[self.i_pre, :] = net_precip
        # Conservative coupling: dissolved scaling pool debited/credited.
        r[self.i_diss, :] = -self.stoich * net_precip
        return r


# ===========================================================================
class FinesMigrationKinetics(ReactionModel):
    """
    Critical-velocity fines detachment + filtration deposition, with a mobile
    suspended-fines species and an immobile deposited-fines species.
    Subclasses ReactionModel.
    """

    def __init__(self, registry, velocity, properties,
                 suspended_name="fines_suspended",
                 deposited_name="fines_deposited",
                 critical_velocity=1.0e-4,
                 k_detach=1.0e-2, k_deposit=5.0e1):
        self.i_sus = registry.index(suspended_name)
        self.i_dep = registry.index(deposited_name)
        self.v_c = float(critical_velocity)
        self.k_det = float(k_detach)
        self.k_dep = float(k_deposit)

        # Interstitial speed field per DOF (constant within a flow update;
        # refreshed via update_velocity when the Darcy field changes). Velocity
        # may be None at construction and supplied later via update_velocity
        # (useful when the simulator solves flow after building the model).
        if velocity is not None:
            self._set_velocity(velocity, properties)
        else:
            self.v_dof = np.zeros(properties.mesh.n_active)

    def _set_velocity(self, velocity, properties):
        mesh = properties.mesh
        v_mag = velocity.speed / np.maximum(properties.phi, 1e-6)
        ijk = mesh.dof_to_ijk
        self.v_dof = v_mag[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def update_velocity(self, velocity, properties):
        """Refresh interstitial speed after a formation-damage velocity update."""
        self._set_velocity(velocity, properties)

    def rates(self, state, registry, mesh, properties):
        r = np.zeros_like(state)
        c_s = state[self.i_sus, :]
        sigma = state[self.i_dep, :]
        v = self.v_dof

        # Detachment ramps up once |v| exceeds the critical velocity.
        excess = np.maximum(v / self.v_c - 1.0, 0.0)
        R_det = self.k_det * excess * sigma            # >=0  (sigma -> c_s)
        # Filtration deposition proportional to speed and suspended load.
        R_dep = self.k_dep * v * c_s                   # >=0  (c_s -> sigma)

        r[self.i_sus, :] = R_det - R_dep
        r[self.i_dep, :] = R_dep - R_det               # equal & opposite
        return r


# ===========================================================================
class CompositeReactionModel(ReactionModel):
    """
    Sum the rates of several reaction models so multiple processes
    (precipitation AND fines migration) act simultaneously, while each remains
    an independent, separately-testable module. The transport backbone sees a
    single ReactionModel, preserving the clean interface.
    """

    def __init__(self, models):
        self.models = list(models)

    def rates(self, state, registry, mesh, properties):
        total = np.zeros_like(state)
        for m in self.models:
            total += m.rates(state, registry, mesh, properties)
        return total

    def name(self):
        return "Composite(" + "+".join(m.name() for m in self.models) + ")"
