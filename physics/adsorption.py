"""
ASPiRe-3D : core/adsorption.py
===============================================================================
Surfactant ADSORPTION and polymer RETENTION reaction models for ASP flooding.
Both subclass ReactionModel, so the validated transport core is untouched. This
is the phase that makes ASPiRe-3D realistically ASP: adsorption/retention are
the dominant chemical-loss and permeability-damage mechanisms that core-flood
history matching actually fits.

-------------------------------------------------------------------------------
WHY ADSORPTION NEEDS NO NEW TRANSPORT MACHINERY
-------------------------------------------------------------------------------
Adsorbed surfactant / retained polymer are IMMOBILE species (already supported).
Adsorption is a REACTION that moves mass from the dissolved (mobile) species to
the adsorbed (immobile) one. The classic RETARDATION of the breakthrough front
(R = 1 + (rho_b/phi) dq/dC) then emerges automatically from the operator-split
mass exchange -- we do NOT hand-code a retardation factor into the transport
matrix. This is the payoff of the modular ReactionModel architecture.

EQUILIBRIUM ISOTHERMS VIA LINEAR-DRIVING-FORCE KINETICS
-------------------------------------------------------
Isotherms q = q_eq(C) are equilibrium statements; our solver is kinetic. We
drive the adsorbed amount toward the isotherm with first-order (linear driving
force) kinetics:

        dq/dt = k_a ( q_eq(C) - q )

Large k_a  -> local equilibrium (instantaneous isotherm).
Finite k_a -> rate-limited adsorption (kinetic hysteresis).
This is the standard way to embed isotherms in a kinetic reactive-transport
code and stays entirely within the existing splitting framework.

Concentrations here use C in [kg/m^3 fluid]; adsorbed amount q is carried as an
immobile species in [kg adsorbate / m^3 BULK] so it plugs directly into the
formation-damage porosity/permeability coupling. The isotherm's natural units
(mass adsorbate / mass rock) are converted with the bulk density rho_b.
===============================================================================
"""

import numpy as np
from physics.species import ReactionModel


# ===========================================================================
#  ISOTHERM FUNCTIONS  (equilibrium adsorbed amount q_eq given dissolved C)
# ===========================================================================
def langmuir_isotherm(C, q_max, K_L):
    """
    Langmuir: q_eq = q_max * K_L * C / (1 + K_L * C)   [mass adsorbate/mass rock]

    Monolayer adsorption with saturation: q_eq -> q_max as C -> inf. K_L is the
    affinity (1/concentration). The canonical surfactant adsorption model.
    """
    C = np.maximum(C, 0.0)
    return q_max * K_L * C / (1.0 + K_L * C)


def freundlich_isotherm(C, K_F, n_F):
    """
    Freundlich: q_eq = K_F * C^(1/n_F)   (no saturation; empirical).

    Captures heterogeneous-surface adsorption that keeps rising with C. n_F>1 is
    typical (favourable adsorption). Used when Langmuir saturation is not seen.
    """
    C = np.maximum(C, 0.0)
    return K_F * np.power(C, 1.0 / n_F)


# ===========================================================================
class SurfactantAdsorption(ReactionModel):
    """
    Surfactant adsorption with Langmuir or Freundlich isotherm, optional
    irreversibility, salinity sensitivity, and a pH/alkali-dependent modifier.

    States used:
        dissolved species  (mobile)   : 'surfactant'
        adsorbed species   (immobile) : 'surfactant_adsorbed'  [kg/m^3 bulk]
    Optional context species for modifiers: 'salinity', 'alkali'.
    """

    def __init__(self, registry, properties,
                 dissolved_name="surfactant",
                 adsorbed_name="surfactant_adsorbed",
                 isotherm="langmuir",
                 q_max=5.0e-4, K_L=2.0,         # Langmuir params (mass/mass, 1/conc)
                 K_F=3.0e-4, n_F=1.5,           # Freundlich params
                 rate_constant=1.0e-2,          # k_a linear-driving-force [1/s]
                 bulk_density=2000.0,           # rho_b [kg/m^3] grain*(1-phi)
                 irreversible=False,
                 # salinity sensitivity: adsorption scales with (1 + beta_s * salinity)
                 salinity_name=None, salinity_coeff=0.0,
                 # pH/alkali modifier: adsorption scales with (1 - beta_pH * alkali),
                 # i.e. high alkalinity reduces surfactant adsorption (well documented)
                 alkali_name=None, alkali_coeff=0.0):
        self.props = properties
        self.i_diss = registry.index(dissolved_name)
        self.i_ads = registry.index(adsorbed_name)
        self.isotherm = isotherm
        self.q_max = float(q_max); self.K_L = float(K_L)
        self.K_F = float(K_F); self.n_F = float(n_F)
        self.k_a = float(rate_constant)
        self.rho_b = float(bulk_density)
        self.irreversible = bool(irreversible)
        self.i_sal = registry.index(salinity_name) if salinity_name else None
        self.beta_s = float(salinity_coeff)
        self.i_alk = registry.index(alkali_name) if alkali_name else None
        self.beta_pH = float(alkali_coeff)

    # -----------------------------------------------------------------------
    def equilibrium_adsorbed(self, C, salinity=None, alkali=None):
        """
        Equilibrium adsorbed amount in [kg/m^3 BULK], including salinity and
        pH modifiers. Isotherm gives mass/mass; multiply by rho_b for per-bulk.
        """
        if self.isotherm == "langmuir":
            q_mm = langmuir_isotherm(C, self.q_max, self.K_L)
        elif self.isotherm == "freundlich":
            q_mm = freundlich_isotherm(C, self.K_F, self.n_F)
        else:
            raise ValueError(f"unknown isotherm '{self.isotherm}'")

        modifier = 1.0
        if self.i_sal is not None:
            # Higher salinity screens charge -> MORE surfactant adsorption.
            modifier = modifier * (1.0 + self.beta_s * np.maximum(salinity, 0.0))
        if self.i_alk is not None:
            # Higher alkalinity (pH) -> LESS surfactant adsorption.
            modifier = modifier * np.maximum(1.0 - self.beta_pH * np.maximum(alkali, 0.0), 0.0)
        return q_mm * self.rho_b * modifier

    # -----------------------------------------------------------------------
    def rates(self, state, registry, mesh, properties):
        r = np.zeros_like(state)
        C = state[self.i_diss, :]
        q = state[self.i_ads, :]
        sal = state[self.i_sal, :] if self.i_sal is not None else None
        alk = state[self.i_alk, :] if self.i_alk is not None else None

        q_eq = self.equilibrium_adsorbed(C, sal, alk)
        # Linear driving force toward equilibrium.
        dq_dt = self.k_a * (q_eq - q)
        if self.irreversible:
            # Irreversible: adsorption only (no desorption); clamp negative LDF.
            dq_dt = np.maximum(dq_dt, 0.0)

        r[self.i_ads, :] = dq_dt
        # Conservative coupling: dissolved loses what adsorbs (per bulk volume,
        # but dissolved is per fluid volume; convert with porosity so total mass
        # is conserved). d C_fluid = -(1/phi) dq_bulk.
        ijk = mesh.dof_to_ijk
        phi = properties.phi[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        r[self.i_diss, :] = -dq_dt / np.maximum(phi, 1e-6)
        return r


# ===========================================================================
class PolymerRetention(ReactionModel):
    """
    Polymer retention with residual resistance factor (RRF), inaccessible pore
    volume (IAPV), optional shear degradation, and permeability-dependent
    retention. Subclasses ReactionModel.

    States:
        dissolved species (mobile)   : 'polymer'
        retained species  (immobile) : 'polymer_retained'  [kg/m^3 bulk]

    PHYSICS
    -------
    1. RETENTION (adsorption + mechanical entrapment) toward a capacity, by
       linear driving force:
           d(sigma_p)/dt = k_r ( sigma_max_eff - sigma_p )   if C>0 (irreversible)
       Polymer retention is largely IRREVERSIBLE, so by default desorption is
       disabled.

    2. PERMEABILITY-DEPENDENT RETENTION: retention capacity scales with a power
       of the local permeability ratio -- lower-k zones retain more polymer
       (smaller pores trap more), an experimentally observed trend:
           sigma_max_eff = sigma_max * (k/k0)^gamma

    3. SHEAR DEGRADATION (optional): at high interstitial velocity the polymer
       chains break, lowering the effective dissolved concentration's viscosity
       contribution. We model degradation as a first-order loss of dissolved
       polymer above a critical shear velocity (a simple, documented surrogate):
           d C/dt |_shear = -k_sh (|v|/v_sh - 1)_+ C

    RRF and inaccessible pore volume are PROPERTIES of the retained state and are
    applied to flow/transport by the coupling layer (see apply_to_properties),
    not as reaction rates -- they modify permeability and effective porosity.
    """

    def __init__(self, registry, properties, velocity=None,
                 dissolved_name="polymer",
                 retained_name="polymer_retained",
                 sigma_max=3.0e-4,              # retention capacity [kg/m^3 bulk]
                 rate_constant=2.0e-2,          # k_r [1/s]
                 perm_dependence_gamma=0.0,     # capacity ~ (k/k0)^(-gamma)
                 reversible=False,
                 # shear degradation
                 shear_degradation=False, shear_velocity=1.0e-3, k_shear=1.0e-2):
        self.props = properties
        self.i_diss = registry.index(dissolved_name)
        self.i_ret = registry.index(retained_name)
        self.sigma_max = float(sigma_max)
        self.k_r = float(rate_constant)
        self.gamma = float(perm_dependence_gamma)
        self.reversible = bool(reversible)
        self.shear = bool(shear_degradation)
        self.v_sh = float(shear_velocity)
        self.k_sh = float(k_shear)
        self._v_dof = None
        if velocity is not None:
            self.update_velocity(velocity, properties)

    def update_velocity(self, velocity, properties):
        mesh = properties.mesh
        v_mag = velocity.speed / np.maximum(properties.phi, 1e-6)
        ijk = mesh.dof_to_ijk
        self._v_dof = v_mag[ijk[:, 0], ijk[:, 1], ijk[:, 2]]

    def rates(self, state, registry, mesh, properties):
        r = np.zeros_like(state)
        C = state[self.i_diss, :]
        sigma = state[self.i_ret, :]
        ijk = mesh.dof_to_ijk

        # Permeability-dependent retention capacity (lower k -> more retention).
        k = properties.k[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        k0 = properties.k0[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        cap = self.sigma_max * np.power(np.maximum(k / k0, 1e-9), -self.gamma)

        # Retention only proceeds where dissolved polymer is present.
        present = (C > 0.0).astype(float)
        d_sigma = self.k_r * (cap - sigma) * present
        if not self.reversible:
            d_sigma = np.maximum(d_sigma, 0.0)
        r[self.i_ret, :] = d_sigma
        phi = properties.phi[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        r[self.i_diss, :] = -d_sigma / np.maximum(phi, 1e-6)

        # Optional shear degradation of dissolved polymer.
        if self.shear and self._v_dof is not None:
            excess = np.maximum(self._v_dof / self.v_sh - 1.0, 0.0)
            r[self.i_diss, :] += -self.k_sh * excess * C
        return r
