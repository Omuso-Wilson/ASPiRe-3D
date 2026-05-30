"""
ASPiRe-3D : core/formation_damage.py
===============================================================================
Formation-damage COUPLING: convert reaction products (precipitate, deposited
fines) into porosity reduction, then update permeability via a chosen
porosity-permeability relationship. This is the module that turns "reactive
transport" into "formation-damage reactive transport".

PHYSICAL CHAIN
--------------
    precipitate volume + deposited fines volume
        -> porosity reduction   phi = phi0 - V_precip - V_fines
        -> permeability update  k = f(phi)        [Kozeny-Carman / power / exp]
        -> (caller re-solves pressure & velocity with the new k)

POROSITY REDUCTION
------------------
Each immobile damage species carries a concentration in [mass / bulk volume].
Dividing by its solid density gives the volume fraction it occupies, which is
subtracted from the initial porosity:

    phi = phi0 - (c_precip / rho_precip) - (sigma_fines / rho_fines)

We clamp phi to a small positive floor (phi_min) so the rock never becomes a
true zero-porosity solid (numerically and physically defensible: some residual
pore space always remains, and k stays positive).

POROSITY-PERMEABILITY RELATIONSHIPS (selectable)
------------------------------------------------
1. KOZENY-CARMAN (mechanistic, the classic):
       k/k0 = (phi/phi0)^3 * ((1-phi0)/(1-phi))^2
   Derived from capillary-bundle theory; strong, physically grounded
   sensitivity of k to phi. Default.

2. POWER-LAW (empirical, tunable exponent n):
       k/k0 = (phi/phi0)^n
   Widely fitted to core data; n ~ 3 recovers near-Kozeny behaviour, larger n
   models more severe damage.

3. EXPONENTIAL DAMAGE (formation-damage literature, uses damage amount):
       k/k0 = exp( -beta * (phi0 - phi)/phi0 )
   A compact one-parameter impairment model common in injectivity studies.

All three are monotone (k decreases as phi decreases) and return k>0.

DESIGN: the damage update reads the immobile species fields from the reactive
solver state and writes the updated phi and k back into the shared
FluidRockProperties arrays -- the SAME arrays the pressure solver and transport
operators read. So once damage updates them, the next pressure solve and
operator rebuild automatically reflect the impairment. This is the coupling
seam designed back in Phase 1.
===============================================================================
"""

import numpy as np


# ---------------------------------------------------------------------------
#  POROSITY-PERMEABILITY RELATIONSHIPS
# ---------------------------------------------------------------------------
def kozeny_carman(phi, phi0, k0):
    """k/k0 = (phi/phi0)^3 * ((1-phi0)/(1-phi))^2 ; elementwise, k>0."""
    phi = np.clip(phi, 1e-6, 0.999999)
    ratio = (phi / phi0) ** 3 * ((1.0 - phi0) / (1.0 - phi)) ** 2
    return k0 * ratio


def power_law(phi, phi0, k0, exponent=3.0):
    """k/k0 = (phi/phi0)^n."""
    phi = np.clip(phi, 1e-6, 0.999999)
    return k0 * (phi / phi0) ** exponent


def exponential_damage(phi, phi0, k0, beta=10.0):
    """k/k0 = exp(-beta * (phi0 - phi)/phi0)."""
    damage_fraction = np.clip((phi0 - phi) / phi0, 0.0, 1.0)
    return k0 * np.exp(-beta * damage_fraction)


_PERM_MODELS = {
    "kozeny_carman": kozeny_carman,
    "power_law": power_law,
    "exponential": exponential_damage,
}


# ---------------------------------------------------------------------------
class FormationDamage:
    """
    Applies porosity reduction and permeability update from immobile damage
    species, enforcing porosity bounds and permeability positivity.
    """

    def __init__(self, mesh, properties, registry,
                 perm_model="kozeny_carman",
                 perm_kwargs=None,
                 precipitate_name="precipitate",
                 fines_deposited_name="fines_deposited",
                 rho_precipitate=2600.0,   # kg/m^3 (mineral scale density)
                 rho_fines=2650.0,         # kg/m^3 (clay/quartz fines density)
                 phi_min=0.01,
                 k_min=1.0e-20,
                 # --- Phase 5: polymer retention & surfactant adsorption ---
                 polymer_retained_name=None,   # immobile retained polymer species
                 rrf_max=1.0,                  # max residual resistance factor
                 sigma_ref_polymer=3.0e-4,     # retention at which RRF ~ rrf_max
                 surfactant_adsorbed_name=None,
                 rho_surfactant=1100.0):       # adsorbate density [kg/m^3]
        if perm_model not in _PERM_MODELS:
            raise ValueError(f"unknown perm_model '{perm_model}'; "
                             f"choose from {list(_PERM_MODELS)}")
        self.mesh = mesh
        self.props = properties
        self.registry = registry
        self.perm_model = perm_model
        self.perm_fn = _PERM_MODELS[perm_model]
        self.perm_kwargs = perm_kwargs or {}
        self.rho_p = float(rho_precipitate)
        self.rho_f = float(rho_fines)
        self.phi_min = float(phi_min)
        self.k_min = float(k_min)

        # Optional immobile species (may be absent in a given run).
        self._has_precip = precipitate_name in registry.names
        self._has_fines = fines_deposited_name in registry.names
        self._i_precip = (registry.index(precipitate_name)
                          if self._has_precip else None)
        self._i_fines = (registry.index(fines_deposited_name)
                         if self._has_fines else None)

        # Phase 5 retention/adsorption coupling.
        self._has_polymer = (polymer_retained_name is not None
                             and polymer_retained_name in registry.names)
        self._i_polymer = (registry.index(polymer_retained_name)
                           if self._has_polymer else None)
        self.rrf_max = float(rrf_max)
        self.sigma_ref_polymer = float(sigma_ref_polymer)
        self._has_surf_ads = (surfactant_adsorbed_name is not None
                              and surfactant_adsorbed_name in registry.names)
        self._i_surf_ads = (registry.index(surfactant_adsorbed_name)
                            if self._has_surf_ads else None)
        self.rho_s = float(rho_surfactant)

        # Snapshots of pristine fields for ratio-based laws.
        self.phi0 = properties.phi0.copy()
        self.k0 = properties.k0.copy()

    # -----------------------------------------------------------------------
    def apply(self, reactive_state):
        """
        Update properties.phi and properties.k in place from the current
        immobile-species concentrations in `reactive_state` (n_species,n_active).

        Returns a diagnostics dict (min/mean porosity, min k ratio, etc.).
        """
        mesh, props = self.mesh, self.props
        ijk = mesh.dof_to_ijk

        # Volume fraction occupied by damage products, per active DOF.
        vol_frac = np.zeros(mesh.n_active)
        if self._has_precip:
            vol_frac += reactive_state[self._i_precip, :] / self.rho_p
        if self._has_fines:
            vol_frac += reactive_state[self._i_fines, :] / self.rho_f
        if self._has_surf_ads:
            # Adsorbed surfactant occupies a (small) pore volume too.
            vol_frac += reactive_state[self._i_surf_ads, :] / self.rho_s

        # Scatter to grid and form new porosity (bounded).
        phi0_dof = self.phi0[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        phi_new_dof = np.clip(phi0_dof - vol_frac, self.phi_min, 0.999999)

        # Permeability from the chosen law (vectorized, per DOF).
        k0_dof = self.k0[ijk[:, 0], ijk[:, 1], ijk[:, 2]]
        k_new_dof = self.perm_fn(phi_new_dof, phi0_dof, k0_dof,
                                 **self.perm_kwargs)

        # --- Polymer Residual Resistance Factor (RRF) ----------------------
        # Retained polymer reduces permeability to subsequent flow by an extra
        # factor 1/RRF, where RRF rises from 1 (no polymer) toward rrf_max as
        # retention approaches sigma_ref. This is the classic permeability-
        # reduction-from-polymer mechanism, DISTINCT from pore-volume occupancy
        # (so it is applied as a multiplicative permeability penalty, not a
        # porosity loss). RRF persists even after polymer passes (irreversible
        # retention), which is the experimentally observed behaviour.
        rrf_factor = np.ones(mesh.n_active)
        if self._has_polymer and self.rrf_max > 1.0:
            sigma_poly = reactive_state[self._i_polymer, :]
            frac = np.clip(sigma_poly / self.sigma_ref_polymer, 0.0, 1.0)
            RRF = 1.0 + (self.rrf_max - 1.0) * frac
            rrf_factor = 1.0 / RRF
            k_new_dof = k_new_dof * rrf_factor

        k_new_dof = np.maximum(k_new_dof, self.k_min)   # positivity enforcement

        # Write back into the shared full-grid property arrays.
        props.phi[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = phi_new_dof
        props.k[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = k_new_dof

        # Diagnostics.
        k_ratio = k_new_dof / k0_dof
        return {
            "phi_min": float(phi_new_dof.min()),
            "phi_mean": float(phi_new_dof.mean()),
            "k_ratio_min": float(k_ratio.min()),
            "k_ratio_mean": float(k_ratio.mean()),
            "max_vol_frac": float(vol_frac.max()),
        }

    # -----------------------------------------------------------------------
    def damage_index_grid(self):
        """
        Formation Damage Index (FDI) = 1 - k/k0 per cell, in [0,1].
        0 = pristine, 1 = fully damaged. Returned on the full grid (NaN outside).
        """
        mesh = self.mesh
        FDI = np.full((mesh.nx, mesh.ny, mesh.nz), np.nan)
        act = mesh.active
        FDI[act] = 1.0 - self.props.k[act] / self.k0[act]
        return FDI
