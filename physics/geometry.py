"""
ASPiRe-3D : core/geometry.py
===============================================================================
Cylindrical core-plug geometry embedded in the Cartesian mesh by MASKING.

STRATEGY (and its justification)
--------------------------------
A core plug is a right circular cylinder. Rather than build a body-fitted
curvilinear mesh (which complicates every flux and every future transport
operator), we embed the cylinder in a Cartesian box and tag each cell as
INSIDE or OUTSIDE the cylinder. This is an "embedded boundary" / cut-cell-lite
approach.

    - Flow axis      : x  (core length, e.g. 0.10 m)
    - Cross-section  : y-z plane (a disc of radius R = diameter/2)

A cell (i,j,k) is ACTIVE if its center (yc[j], zc[k]) lies within radius R of
the core's central axis. Faces between an active and an inactive cell are
treated downstream as NO-FLOW boundaries -- this represents the impermeable
core sleeve / Hassler holder in a real core-flood experiment, where fluid can
only enter the inlet face (x=0) and leave the outlet face (x=L).

APPROXIMATION
-------------
The lateral boundary is "staircased" rather than smooth. This is standard and
acceptable; the effective flow area is the count of active cells in a cross
section times the cell face area, and we report it so experimental
differential-pressure validation can use the true discretized area.
===============================================================================
"""

import numpy as np


def build_cylinder_mask(mesh, radius):
    """
    Return a boolean (nx, ny, nz) mask: True where the cell center lies inside
    the cylinder of given `radius` whose axis runs along x through the center
    of the y-z cross-section.

    Parameters
    ----------
    mesh : StructuredMesh
    radius : float
        Core plug radius [m] (= diameter / 2).
    """
    # Center of the cross-section (axis location) in physical coordinates.
    y_axis = 0.5 * mesh.Ly
    z_axis = 0.5 * mesh.Lz

    # Radial distance of every (j,k) cross-section cell center from the axis.
    # Build a 2D radius field once, then replicate along x.
    YC, ZC = np.meshgrid(mesh.yc, mesh.zc, indexing="ij")   # (ny, nz)
    radial_distance = np.sqrt((YC - y_axis) ** 2 + (ZC - z_axis) ** 2)
    inside_disc = radial_distance <= radius                  # (ny, nz) bool

    # The core is a prism of that disc along all x positions: broadcast.
    mask = np.broadcast_to(inside_disc[None, :, :],
                           (mesh.nx, mesh.ny, mesh.nz)).copy()
    return mask


def effective_cross_section_area(mesh):
    """
    Discrete (staircased) flow cross-section area [m^2]: number of active cells
    in a single x-slice times the x-face area. Assumes the mask is x-invariant
    (true for a straight cylinder), so slice i=0 is representative.
    """
    n_active_in_slice = int(np.count_nonzero(mesh.active[0, :, :]))
    return n_active_in_slice * mesh.area_x


def apply_cylindrical_core(mesh, diameter):
    """
    Convenience wrapper: build the cylinder mask for the given diameter and
    install it on the mesh, returning a short geometry report.
    """
    radius = 0.5 * diameter
    mask = build_cylinder_mask(mesh, radius)
    mesh.set_active_mask(mask)

    area_discrete = effective_cross_section_area(mesh)
    area_ideal = np.pi * radius ** 2
    report = (
        "CylindricalCore\n"
        f"  diameter           : {diameter:.4e} m  (radius {radius:.4e} m)\n"
        f"  ideal area         : {area_ideal:.4e} m^2\n"
        f"  discretized area   : {area_discrete:.4e} m^2 "
        f"({100.0 * area_discrete / area_ideal:.1f}% of ideal)\n"
    )
    return report
