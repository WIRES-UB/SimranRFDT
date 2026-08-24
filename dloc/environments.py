"""Build simulator scenes for the DLoc environments.

Geometry comes from the measurement files, not from this module.  The dataset
provides AP antenna coordinates and ground-truth robot positions in the SLAM
map frame, and those are used directly so the scene is the dataset's rather
than a reconstruction of it.

What this module does supply is the part the dataset does not record: wall
height, floor and ceiling, and which surface is made of what.  Those are
assumptions, they are listed in ``PROTOCOL.md``, and they are attached to every
scene built here so a run log can show what was assumed.

There is deliberately no fallback that invents AP positions or a room outline.
A scene can only be built from a loaded measurement, because a comparison
against real data is meaningless if the geometry was guessed.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdt.geometry import Mesh, box, quad                             # noqa: E402
from rfdt.materials import get_material                               # noqa: E402

import floorplan as fp                                                # noqa: E402
from dataset import DLocMeasurement, environment_of                   # noqa: E402

#: Wall height where the dataset does not record one.  Typical for the lab
#: spaces in the DLoc paper; recorded as an assumption rather than a fact.
DEFAULT_CEILING_M = 3.0

#: Margin added around the robot's route to place the walls, when the room
#: outline is not otherwise known.
DEFAULT_WALL_MARGIN_M = 0.5


@dataclass
class SceneAssumptions:
    """Every choice not taken from the measurement file.

    Carried alongside the mesh so a result can be reported together with what
    it assumed, instead of the assumptions living only in code.
    """

    wall_material: str = "plasterboard"
    floor_material: str = "concrete"
    ceiling_material: str = "ceiling_board"
    #: Jacobs only: the plasma/LCD screen wall behind AP3.
    screen_wall_material: str = "metal"
    #: Jacobs only: the wall AP4 sits behind.
    metal_wall_material: str = "metal"
    ceiling_height_m: float = DEFAULT_CEILING_M
    wall_margin_m: float = DEFAULT_WALL_MARGIN_M
    client_height_m: float = 1.0
    ap_height_m: float = 1.5
    notes: List[str] = field(default_factory=list)

    def describe(self) -> Dict[str, object]:
        """Assumptions as a dictionary, for the run log."""
        return {
            "wall_material": self.wall_material,
            "floor_material": self.floor_material,
            "ceiling_material": self.ceiling_material,
            "screen_wall_material": self.screen_wall_material,
            "metal_wall_material": self.metal_wall_material,
            "ceiling_height_m": self.ceiling_height_m,
            "wall_margin_m": self.wall_margin_m,
            "client_height_m": self.client_height_m,
            "ap_height_m": self.ap_height_m,
            "notes": list(self.notes),
        }


@dataclass
class DLocScene:
    """A simulator scene paired with the measurement it was built from."""

    mesh: Mesh
    ap_positions: torch.Tensor        # (n_ap, 3)
    client_positions: torch.Tensor    # (n_points, 3)
    assumptions: SceneAssumptions
    setup: str
    environment: str
    bounds: Dict[str, float]

    def summary(self) -> Dict[str, object]:
        """What was built, for the run log."""
        return {
            "setup": self.setup,
            "environment": self.environment,
            "n_surfaces": len(self.mesh.surfaces()),
            "n_triangles": self.mesh.n_tri,
            "n_ap": int(self.ap_positions.shape[0]),
            "n_client_positions": int(self.client_positions.shape[0]),
            "bounds": self.bounds,
            "materials": sorted(set(self.mesh.mat_names)),
            "assumptions": self.assumptions.describe(),
        }


def _shell(x0, x1, y0, y1, z1, wall, floor, ceiling) -> Mesh:
    """Rectangular room shell with inward-facing normals, per named surface.

    Built face by face rather than with :func:`rfdt.geometry.box` so each wall
    can take its own material, which Jacobs needs for the screen wall and the
    metal wall.
    """
    quads, mats, groups = [], [], []

    # floor, normal up
    quads.append(quad((x0, y0, 0.0), (x1, y0, 0.0), (x1, y1, 0.0), (x0, y1, 0.0)))
    mats.append(floor)
    groups.append("floor")
    # ceiling, normal down
    quads.append(quad((x0, y0, z1), (x0, y1, z1), (x1, y1, z1), (x1, y0, z1)))
    mats.append(ceiling)
    groups.append("ceiling")
    # four walls, wound so the right-hand-rule normal points into the room.
    # Getting this backwards is silent: the wall simply contributes no
    # specular path, so the assertion below is not optional.
    quads.append(quad((x0, y0, 0.0), (x0, y1, 0.0), (x0, y1, z1), (x0, y0, z1)))
    mats.append(wall)
    groups.append("wall_xmin")
    quads.append(quad((x1, y0, 0.0), (x1, y0, z1), (x1, y1, z1), (x1, y1, 0.0)))
    mats.append(wall)
    groups.append("wall_xmax")
    quads.append(quad((x0, y0, 0.0), (x0, y0, z1), (x1, y0, z1), (x1, y0, 0.0)))
    mats.append(wall)
    groups.append("wall_ymin")
    quads.append(quad((x0, y1, 0.0), (x1, y1, 0.0), (x1, y1, z1), (x0, y1, z1)))
    mats.append(wall)
    groups.append("wall_ymax")

    mesh = Mesh.from_quads(quads, mats, groups).weld()
    # verify the normals really do face inwards, since a wall facing the wrong
    # way contributes no specular path at all and does so silently
    centre = torch.tensor([[0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * z1]],
                          dtype=torch.float64)
    for si, s in enumerate(mesh.surfaces()):
        tri = mesh.tri()[s.tri0]
        p0 = tri[0]
        n = torch.cross(tri[1] - tri[0], tri[2] - tri[0], dim=-1)
        n = n / n.norm()
        if float(((centre - p0) * n).sum()) <= 0:
            raise ValueError(f"surface {s.group} has an outward normal; the "
                             "room shell must face inwards")
    return mesh


def shell_from_polygon(poly: np.ndarray, height: float, wall: str,
                       floor: str, ceiling: str,
                       wall_materials: Optional[Dict[int, str]] = None) -> Mesh:
    """Build a room shell from an arbitrary floor-plan outline.

    ``poly`` is an ``(n, 2)`` counter-clockwise polygon in metres.  Each edge
    becomes one vertical wall facet, so a wall can be given its own material by
    index through ``wall_materials``.  Floor and ceiling are triangulated; the
    mesh merges the coplanar triangles back into a single facet each, so the
    tracer sees one floor rather than dozens of slivers.

    Winding matters and is not cosmetic.  For a counter-clockwise outline the
    interior lies to the left of each edge, and the vertex order used below
    puts the wall normal on that side.  A clockwise polygon would build the
    room inside out, with every wall facing away from the space, which is
    silent rather than an error, so the result is asserted at the end.
    """
    poly = np.asarray(poly, dtype=float)
    n = len(poly)
    if n < 3:
        raise ValueError("a floor plan needs at least 3 corners")
    quads, mats, groups = [], [], []

    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        # (a,0) -> (a,h) -> (b,h) -> (b,0) puts the normal to the left of a->b
        quads.append(quad((a[0], a[1], 0.0), (a[0], a[1], height),
                          (b[0], b[1], height), (b[0], b[1], 0.0)))
        mats.append((wall_materials or {}).get(i, wall))
        groups.append(f"wall{i:03d}")

    tris = fp.triangulate(poly)
    if not tris:
        raise ValueError("floor plan could not be triangulated")
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    fmats: List[str] = []
    fgroups: List[str] = []
    for level, mat, name, flip in ((0.0, floor, "floor", False),
                                   (height, ceiling, "ceiling", True)):
        base = len(verts)
        verts.extend([[float(p[0]), float(p[1]), level] for p in poly])
        for (a, b, c) in tris:
            # the ceiling is wound the other way so its normal points down
            tri = (base + a, base + c, base + b) if flip else (base + a, base + b, base + c)
            faces.append(list(tri))
            fmats.append(mat)
            fgroups.append(f"{name}.{len(faces)}")

    mesh = Mesh.from_quads(quads, mats, groups)
    caps = Mesh(torch.tensor(verts, dtype=torch.float64), np.asarray(faces),
                fmats, fgroups)
    mesh = mesh.merged(caps).weld()

    _assert_faces_inward(mesh, poly, height)
    return mesh


def _point_in_polygon(pt: np.ndarray, poly: np.ndarray) -> bool:
    """Ray-casting point-in-polygon test, exact for non-convex outlines."""
    x, y = float(pt[0]), float(pt[1])
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            xint = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
            if x < xint:
                inside = not inside
    return inside


def _assert_faces_inward(mesh: Mesh, poly: np.ndarray, height: float) -> None:
    """Check every surface reflects into the room, and say which if not.

    Each wall is tested *locally*: step a short way along its own normal from
    its own midpoint, and require that point to be inside the floor plan.  A
    single interior reference point does not work for a non-convex outline,
    where the centroid can sit on, or the far side of, a reflex wall.
    """
    scale = max(float(np.ptp(poly[:, 0])), float(np.ptp(poly[:, 1])), 1.0)
    step = 1e-3 * scale
    bad = []
    for s_ in mesh.surfaces():
        tri = mesh.tri()[s_.tri0].detach().numpy()
        nrm = np.cross(tri[1] - tri[0], tri[2] - tri[0])
        norm = np.linalg.norm(nrm)
        if norm < 1e-12:
            bad.append(f"{s_.group} (degenerate)")
            continue
        nrm = nrm / norm
        if abs(nrm[2]) > 0.9:                      # floor or ceiling
            want_up = s_.group.startswith("floor")
            if (nrm[2] > 0) != want_up:
                bad.append(s_.group)
            continue
        centre = np.array([float(np.mean([v[0] for v in tri])),
                           float(np.mean([v[1] for v in tri]))])
        if not _point_in_polygon(centre + nrm[:2] * step, poly):
            bad.append(s_.group)
    if bad:
        raise ValueError(
            f"{len(bad)} surface(s) face out of the room: {bad[:6]}. "
            "The floor-plan polygon must be wound counter-clockwise.")


def assign_wall_material_near(mesh: Mesh, poly: np.ndarray,
                              anchors: Sequence[Sequence[float]],
                              material: str, radius: float) -> List[int]:
    """Give the ``material`` to every wall segment near any anchor point.

    Features like the plasma screens are not a whole boundary wall: they cover
    a stretch of it, and in Jacobs they appear on two opposite sides.  A whole
    wall cannot express that, but a run of segments from the floor-plan outline
    can.  Returns the indices assigned, for the run log.
    """
    n = len(poly)
    hit = []
    for i in range(n):
        mid = 0.5 * (np.asarray(poly[i]) + np.asarray(poly[(i + 1) % n]))
        for a in anchors:
            if float(np.linalg.norm(mid - np.asarray(a, dtype=float))) <= radius:
                mesh.set_group_material(f"wall{i:03d}", material)
                hit.append(i)
                break
    return hit


def add_column(mesh: Mesh, centre: Sequence[float], size: Sequence[float],
               material: str, group: str = "blockage") -> Mesh:
    """Add a free-standing column or short partition to a scene.

    The DLoc photograph labels the structure beside AP4 as a "Blockage": a
    narrow vertical element the access point sits behind, not a boundary wall.
    Modelling it as a whole wall would shadow far more of the room than it
    really does, so it goes in as its own solid.
    """
    cx, cy = float(centre[0]), float(centre[1])
    cz = float(size[2]) / 2.0
    return mesh.merged(box((cx, cy, cz), tuple(float(v) for v in size),
                           material, group)).weld()


def build_scene(meas: DLocMeasurement,
                assumptions: Optional[SceneAssumptions] = None,
                subsample: Optional[int] = None) -> DLocScene:
    """Build a scene from a loaded measurement.

    Parameters
    ----------
    meas       : the measurement, which supplies AP coordinates, robot
                 positions and hence the room extent.
    assumptions: wall height and materials; defaults are those in PROTOCOL.md.
    subsample  : keep every n-th robot position.  The full route is 12,500
                 points per setup, which is more than is needed to characterise
                 agreement and is slow to trace; subsampling is a runtime
                 choice and is recorded in the log.

    The room outline is taken from the extent of the robot's own route plus a
    margin, because the dataset gives the route and the AP positions but not a
    wall polygon.  This is an approximation and is listed as one.
    """
    a = assumptions or SceneAssumptions()
    env = environment_of(meas.setup)

    labels = meas.labels
    aps = meas.ap_centroids()
    # the walls must at least enclose both the route and the access points
    pts = np.vstack([labels, aps])
    lo = pts.min(axis=0) - a.wall_margin_m
    hi = pts.max(axis=0) + a.wall_margin_m
    bounds = {"x_min": float(lo[0]), "x_max": float(hi[0]),
              "y_min": float(lo[1]), "y_max": float(hi[1]),
              "z_max": a.ceiling_height_m}

    a.notes.append(
        "Room outline taken as the bounding box of the robot route and the "
        f"access points, inflated by {a.wall_margin_m} m. The dataset does not "
        "provide a wall polygon.")
    a.notes.append(
        f"Ceiling height {a.ceiling_height_m} m assumed; not recorded in the "
        "dataset.")

    mesh = _shell(lo[0], hi[0], lo[1], hi[1], a.ceiling_height_m,
                  a.wall_material, a.floor_material, a.ceiling_material)

    if env == "jacobs":
        # the screen wall and the metal wall are real features of Jacobs, but
        # which wall each corresponds to depends on the map frame, so it is
        # resolved from the AP positions rather than hard-coded
        mesh = _apply_jacobs_materials(mesh, aps, a)

    client_xy = labels if subsample in (None, 1) else labels[::int(subsample)]
    if subsample not in (None, 1):
        a.notes.append(f"Robot positions subsampled by {subsample}: "
                       f"{len(labels)} to {len(client_xy)} points.")
    client = torch.tensor(
        np.column_stack([client_xy, np.full(len(client_xy), a.client_height_m)]),
        dtype=torch.float64)
    ap_pos = torch.tensor(
        np.column_stack([aps, np.full(len(aps), a.ap_height_m)]),
        dtype=torch.float64)

    return DLocScene(mesh=mesh, ap_positions=ap_pos, client_positions=client,
                     assumptions=a, setup=meas.setup, environment=env,
                     bounds=bounds)


def _apply_jacobs_materials(mesh: Mesh, aps: np.ndarray,
                            a: SceneAssumptions) -> Mesh:
    """Assign the screen wall and the metal wall in the Jacobs environment.

    The DLoc paper describes a wall of plasma television screens behind AP3,
    and AP4 hidden behind a wall.  Which of the four walls those are depends on
    the map frame, so each is chosen as the wall nearest the relevant AP rather
    than being hard-coded to a compass direction.
    """
    if len(aps) < 4:
        a.notes.append("Fewer than 4 APs; Jacobs screen and metal walls not applied.")
        return mesh

    verts = mesh.vertices.detach().numpy()
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    walls = {
        "wall_xmin": np.array([lo[0], 0.5 * (lo[1] + hi[1])]),
        "wall_xmax": np.array([hi[0], 0.5 * (lo[1] + hi[1])]),
        "wall_ymin": np.array([0.5 * (lo[0] + hi[0]), lo[1]]),
        "wall_ymax": np.array([0.5 * (lo[0] + hi[0]), hi[1]]),
    }

    def nearest_wall(point, exclude=()):
        """Name of the wall whose midpoint is closest to ``point``."""
        cand = {k: v for k, v in walls.items() if k not in exclude}
        return min(cand, key=lambda k: float(np.linalg.norm(cand[k] - point)))

    screen_wall = nearest_wall(aps[2])                       # AP3
    metal_wall = nearest_wall(aps[3], exclude=(screen_wall,))  # AP4

    mesh.set_group_material(screen_wall, a.screen_wall_material)
    mesh.set_group_material(metal_wall, a.metal_wall_material)
    a.notes.append(
        f"Jacobs: {screen_wall} set to {a.screen_wall_material} as the plasma "
        f"screen wall behind AP3; {metal_wall} set to {a.metal_wall_material} "
        "as the wall AP4 sits behind. Wall identity inferred from AP position, "
        "not from the dataset.")
    return mesh


def check_positions_inside(scene: DLocScene) -> List[str]:
    """Report robot positions that fall outside the room shell.

    A receiver outside the walls produces meaningless results silently, which
    is the same class of error as a route passing through furniture.
    """
    p = scene.client_positions.numpy()
    b = scene.bounds
    bad = ((p[:, 0] < b["x_min"]) | (p[:, 0] > b["x_max"]) |
           (p[:, 1] < b["y_min"]) | (p[:, 1] > b["y_max"]) |
           (p[:, 2] < 0.0) | (p[:, 2] > b["z_max"]))
    if not bad.any():
        return []
    idx = np.nonzero(bad)[0]
    return [f"{len(idx)} client position(s) outside the room shell, "
            f"first at index {int(idx[0])}: {p[idx[0]].round(3).tolist()}"]