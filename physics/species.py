"""
ASPiRe-3D : core/species.py
===============================================================================
Species definitions and the pluggable REACTION-MODEL interface for the
multi-species reactive transport backbone.

DESIGN PHILOSOPHY
-----------------
Every transported quantity in an ASP flood -- salinity, alkali, surfactant,
polymer, suspended fines, precipitate -- obeys the SAME advection-dispersion
equation. They differ only in:
    * inlet (injected) concentration,
    * dispersion coefficient,
    * MOBILITY: mobile (travels with the fluid) vs immobile (attached to rock),
    * and inter-species reaction source/sink terms.

So the backbone stores N concentration fields and steps them through one shared
transport operator, then applies a REACTION MODEL. This file defines:
    (1) the Species data object,
    (2) the SpeciesRegistry (ordered collection -> stable indexing),
    (3) the ReactionModel interface + a NullReactionModel (Phase 3 default:
        pure conservative transport, no chemistry yet).

This is the seam where real kinetics (precipitation, fines detachment/
attachment) will later plug in WITHOUT touching the transport machinery.
===============================================================================
"""

import numpy as np


class Species:
    """
    One transported (or immobile) chemical/physical component.

    Attributes
    ----------
    name : str
        Identifier, e.g. 'salinity', 'alkali', 'surfactant', 'polymer',
        'fines_suspended', 'precipitate'.
    mobile : bool
        True  -> advects + disperses with the aqueous phase (dissolved/suspended)
        False -> immobile (attached/precipitated); transport is skipped, only
                 reactions change it. Modelling immobile species in the same
                 framework keeps the code uniform.
    inlet_value : float
        Injected concentration at the inlet face (for mobile species). Units are
        species-dependent and only need internal consistency in Phase 3
        (e.g. kg/m^3, mol/m^3, or a normalised fraction).
    molecular_diffusion : float
        Species molecular diffusion coefficient D_m [m^2/s]. Fines have ~0;
        small ions have ~1e-9.
    initial_value : float
        Initial in-core concentration (resident brine composition, etc.).
    """

    def __init__(self, name, mobile=True, inlet_value=0.0,
                 molecular_diffusion=1.0e-9, initial_value=0.0):
        self.name = str(name)
        self.mobile = bool(mobile)
        self.inlet_value = float(inlet_value)
        self.molecular_diffusion = float(molecular_diffusion)
        self.initial_value = float(initial_value)

    def __repr__(self):
        kind = "mobile" if self.mobile else "IMMOBILE"
        return (f"Species('{self.name}', {kind}, "
                f"inlet={self.inlet_value:g}, D_m={self.molecular_diffusion:g}, "
                f"init={self.initial_value:g})")


class SpeciesRegistry:
    """
    Ordered registry of species -> gives every species a stable integer index
    used to slice the concentration state array C[species_index, dof].
    """

    def __init__(self, species_list=None):
        self._species = []
        self._index = {}
        if species_list:
            for s in species_list:
                self.add(s)

    def add(self, species):
        if species.name in self._index:
            raise ValueError(f"duplicate species name '{species.name}'")
        self._index[species.name] = len(self._species)
        self._species.append(species)
        return self

    def index(self, name):
        return self._index[name]

    def get(self, name):
        return self._species[self._index[name]]

    @property
    def names(self):
        return [s.name for s in self._species]

    @property
    def mobile_mask(self):
        """Boolean array (n_species,) -- True where the species is mobile."""
        return np.array([s.mobile for s in self._species], dtype=bool)

    def __len__(self):
        return len(self._species)

    def __iter__(self):
        return iter(self._species)

    def summary(self):
        lines = [f"SpeciesRegistry ({len(self)} species)"]
        for i, s in enumerate(self._species):
            lines.append(f"  [{i}] {s!r}")
        return "\n".join(lines)


# ===========================================================================
#  REACTION-MODEL INTERFACE
# ===========================================================================
class ReactionModel:
    """
    Interface for inter-species reaction source/sink terms.

    A reaction model receives the current multi-species concentration state and
    returns the per-species REACTION RATE for each cell, with the SAME shape as
    the state: rates[species_index, dof]  [concentration / second].

    The transport backbone applies these via operator splitting:
        1. transport all mobile species one step,
        2. integrate dC/dt = reaction_rates over the same dt.

    Sign convention: a positive rate INCREASES that species' concentration.
    Mass-conserving reactions must have rates that sum consistently across the
    species they couple (e.g. dissolved ion lost = precipitate gained, scaled by
    stoichiometry) -- the model is responsible for that bookkeeping.

    Phase 3 ships only the NullReactionModel (no chemistry). Real kinetics
    subclass this and override `rates`.
    """

    def rates(self, state, registry, mesh, properties):
        """
        Parameters
        ----------
        state : (n_species, n_active) array  -- current concentrations
        registry : SpeciesRegistry
        mesh, properties : simulator context

        Returns
        -------
        (n_species, n_active) array of dC/dt reaction rates.
        """
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__


class NullReactionModel(ReactionModel):
    """
    No reactions: every rate is zero. With this model the backbone performs
    PURE CONSERVATIVE multi-species transport -- the Phase 3 objective. It lets
    us validate that each species advects/disperses correctly and independently
    before any chemistry is introduced.
    """

    def rates(self, state, registry, mesh, properties):
        return np.zeros_like(state)
