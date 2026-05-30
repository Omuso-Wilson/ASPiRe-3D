"""
ASPiRe-3D : core/mesh.py
===============================================================================
Structured, uniform, cell-centered Cartesian finite-volume mesh.

ROLE IN THE ARCHITECTURE
------------------------
The mesh owns ONLY topology and metrics:
    - grid dimensions (nx, ny, nz) and physical cell sizes (dx, dy, dz),
    - cell-center coordinates,
    - face areas and cell volume,
    - the mapping between a 3D cell (i,j,k) and its linear index,
    - and -- crucially -- the ACTIVE-CELL degree-of-freedom (DOF) map.

It is deliberately geometry-agnostic: it does NOT know the core is a cylinder.
The geometry module (geometry.py) supplies an `active` mask; the mesh consumes
it to build the DOF numbering. This separation means we can later swap the
cylinder for a different sample shape without touching mesh logic.

WHY CELL-CENTERED?
------------------
Unknowns (pressure) live at cell centers; fluxes live at faces. This is the
canonical reservoir-simulation layout and makes the flux balance (sum of face
fluxes = source) map one-to-one onto matrix rows.

THE DOF MAP (the key idea)
--------------------------
A cylinder inscribed in a box leaves many "dead" corner cells. We do NOT want
unknowns or equations for dead cells. So we build two lookups:

    cell_id[i,j,k]  -> compact DOF index in [0, n_active), or -1 if inactive
    dof_to_ijk[dof] -> (i,j,k) of that DOF

The linear system has exactly n_active unknowns. Every later module that
assembles a matrix or a transport operator reuses this same map, guaranteeing
consistency across physics.
===============================================================================
"""

import numpy as np


class StructuredMesh:
    """Uniform Cartesian FVM mesh with an active-cell mask and DOF numbering."""

    def __init__(self, length_x, length_y, length_z, nx, ny, nz):
        """
        Parameters
        ----------
        length_x/y/z : float
            Physical extent of the bounding box [m]. For a core plug we align
            the flow axis with x (length), and y,z span the diameter.
        nx, ny, nz : int
            Number of cells along each axis.
        """
        # ---- Bounding-box dimensions ---------------------------------------
        self.Lx, self.Ly, self.Lz = float(length_x), float(length_y), float(length_z)
        self.nx, self.ny, self.nz = int(nx), int(ny), int(nz)
        self.n_cells_total = self.nx * self.ny * self.nz

        # ---- Uniform cell sizes --------------------------------------------
        # Uniform spacing keeps transmissibility expressions clean. Non-uniform
        # grids are a later refinement if near-inlet resolution is needed.
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.dz = self.Lz / self.nz

        # ---- Cell volume and face areas ------------------------------------
        # In a uniform grid these are constants, but we keep them as named
        # attributes so the assembly code reads like the math.
        self.cell_volume = self.dx * self.dy * self.dz   # [m^3]
        self.area_x = self.dy * self.dz   # area of a face whose normal is x [m^2]
        self.area_y = self.dx * self.dz   # area of a face whose normal is y [m^2]
        self.area_z = self.dx * self.dy   # area of a face whose normal is z [m^2]

        # ---- Cell-center coordinates (1D axes) -----------------------------
        # Center of cell i sits at (i + 0.5) * dx, etc. Stored as 1D arrays;
        # broadcast to 3D on demand to save memory.
        self.xc = (np.arange(self.nx) + 0.5) * self.dx
        self.yc = (np.arange(self.ny) + 0.5) * self.dy
        self.zc = (np.arange(self.nz) + 0.5) * self.dz

        # ---- Active mask and DOF map (filled by set_active_mask) -----------
        # Default: every cell active. geometry.py overrides this for a cylinder.
        self.active = np.ones((self.nx, self.ny, self.nz), dtype=bool)
        self.cell_id = None        # 3D int array: DOF index or -1
        self.dof_to_ijk = None     # (n_active, 3) int array
        self.n_active = None
        self._build_dof_map()

    # -----------------------------------------------------------------------
    def set_active_mask(self, mask):
        """
        Install the active-cell mask (True = rock/fluid, False = outside core)
        and rebuild the DOF numbering accordingly.
        """
        if mask.shape != (self.nx, self.ny, self.nz):
            raise ValueError("active mask shape must match (nx, ny, nz)")
        self.active = mask.astype(bool)
        self._build_dof_map()

    # -----------------------------------------------------------------------
    def _build_dof_map(self):
        """
        Construct compact DOF numbering over active cells only.

        We iterate in a fixed (i, j, k) order so the numbering is deterministic
        and reproducible -- important for debugging and for reproducible thesis
        results. The resulting matrix bandwidth is not optimized here; for
        core-plug-sized problems a direct sparse solve handles it comfortably.
        """
        self.cell_id = -np.ones((self.nx, self.ny, self.nz), dtype=np.int64)
        active_indices = np.argwhere(self.active)          # (n_active, 3)
        self.n_active = active_indices.shape[0]
        # Assign 0..n_active-1 to active cells in argwhere order (C-order).
        for dof, (i, j, k) in enumerate(active_indices):
            self.cell_id[i, j, k] = dof
        self.dof_to_ijk = active_indices.astype(np.int64)

    # -----------------------------------------------------------------------
    def neighbors(self, i, j, k):
        """
        Yield the (axis, area, spacing, (ni,nj,nk)) of each in-bounds,
        face-sharing neighbor of cell (i,j,k).

        Returns a list of tuples:
            (face_area, spacing, neighbor_index_tuple)
        Only structural in-bounds neighbors are returned; whether a neighbor
        is ACTIVE is decided by the assembler (an inactive neighbor becomes a
        no-flow boundary). This keeps mesh purely topological.
        """
        out = []
        # x-direction faces (area_x, spacing dx)
        if i - 1 >= 0:        out.append((self.area_x, self.dx, (i - 1, j, k)))
        if i + 1 < self.nx:   out.append((self.area_x, self.dx, (i + 1, j, k)))
        # y-direction faces
        if j - 1 >= 0:        out.append((self.area_y, self.dy, (i, j - 1, k)))
        if j + 1 < self.ny:   out.append((self.area_y, self.dy, (i, j + 1, k)))
        # z-direction faces
        if k - 1 >= 0:        out.append((self.area_z, self.dz, (i, j, k - 1)))
        if k + 1 < self.nz:   out.append((self.area_z, self.dz, (i, j, k + 1)))
        return out

    # -----------------------------------------------------------------------
    def summary(self):
        """Human-readable mesh report for logs and thesis appendices."""
        fill = 100.0 * self.n_active / self.n_cells_total
        return (
            "StructuredMesh\n"
            f"  bounding box   : {self.Lx:.4f} x {self.Ly:.4f} x {self.Lz:.4f} m\n"
            f"  grid           : {self.nx} x {self.ny} x {self.nz} "
            f"= {self.n_cells_total} cells\n"
            f"  cell size      : dx={self.dx:.4e}  dy={self.dy:.4e}  "
            f"dz={self.dz:.4e} m\n"
            f"  cell volume    : {self.cell_volume:.4e} m^3\n"
            f"  active cells   : {self.n_active}  ({fill:.1f}% of box)\n"
        )
