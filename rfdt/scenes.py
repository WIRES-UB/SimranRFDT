"""Indoor scenes and robot trajectories.

Scenes are triangle meshes with per-surface materials; every builder returns a
*welded* mesh so that edge adjacency (and therefore the diffraction wedges of
Eq. 7) is correct.

Surface groups ("wall", "floor", "ceiling", "cabinet", ...) can be
re-assigned a material after construction with
``mesh.set_group_material("wall", "concrete")``, which is how the material
sweep experiment varies one surface class at a time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .geometry import Mesh, as_t, box, plate

FDTYPE = torch.float64


# ---------------------------------------------------------------------------
# rooms
# ---------------------------------------------------------------------------
def empty_room(size=(6.0, 5.0, 2.8), wall="concrete", floor="wood",
               ceiling="ceiling_board") -> Mesh:
    """Rectangular room shell with inward-facing normals.

    The default floor is wood rather than the ITU "floorboard" entry because
    that regression is specified only for 50-100 GHz, and these scenes are
    also simulated at 5 GHz.  See :func:`rfdt.materials.Material.f_range`.
    """
    sx, sy, sz = map(float, size)
    shell = box((sx / 2, sy / 2, sz / 2), (sx, sy, sz), wall, "room", outward=False)
    # split the shell into named groups so materials can be varied per class
    groups, mats = [], []
    for g in shell.face_groups:
        if g.startswith("room_zmin"):
            groups.append("floor" + g[len("room_zmin"):])
            mats.append(floor)
        elif g.startswith("room_zmax"):
            groups.append("ceiling" + g[len("room_zmax"):])
            mats.append(ceiling)
        else:
            groups.append("wall" + g[len("room"):])
            mats.append(wall)
    return Mesh(shell.vertices, shell.faces, mats, groups).weld()


#: Axis-aligned furniture of :func:`furnished_room`, as
#: ``group -> (center, size)`` in metres.  Kept as data so that trajectories
#: can be validated against it (see :func:`validate_trajectory`).
FURNITURE = {
    "cabinet":   ((0.50, 4.50, 0.90), (0.80, 0.60, 1.80)),
    "table":     ((4.30, 3.50, 0.38), (1.40, 0.80, 0.75)),
    "partition": ((3.00, 1.75, 1.20), (0.12, 3.50, 2.40)),
}


def furnished_room(size=(6.0, 5.0, 2.8), wall="concrete", floor="wood",
                   ceiling="ceiling_board", cabinet="metal", table="wood",
                   partition: Optional[str] = "plasterboard") -> Mesh:
    """Indoor scene for the mobile-robot experiments.

    A 6 x 5 x 2.8 m room containing a metal cabinet, a wooden table and a
    plasterboard partition.  The partition runs from the ``y = 0`` wall to
    ``y = 3.5`` at ``x = 3`` and stands 2.4 m tall, so with the access point in
    the ``x < 3`` half of the room it casts a genuine NLOS shadow over the far
    half while leaving a gap at ``y > 3.5`` through which the robot can drive.
    That gives every route a LoS stretch, an NLOS stretch and a shadow-boundary
    crossing, which is where the differentiable visibility term matters most.
    """
    m = empty_room(size, wall, floor, ceiling)
    mats = {"cabinet": cabinet, "table": table, "partition": partition}
    for group, (center, dims) in FURNITURE.items():
        if mats[group] is None:
            continue
        m = m.merged(box(center, dims, mats[group], group))
    return m.weld()


def furniture_boxes(margin: float = 0.0):
    """Axis-aligned bounds ``(lo, hi)`` of each solid, optionally inflated.

    ``margin`` grows every box, which is how a robot radius is accounted for
    when checking that a route stays clear.
    """
    out = {}
    for group, (center, dims) in FURNITURE.items():
        c = np.asarray(center, dtype=float)
        h = np.asarray(dims, dtype=float) / 2.0 + margin
        out[group] = (c - h, c + h)
    return out


def validate_trajectory(positions: torch.Tensor, margin: float = 0.15,
                        raise_on_error: bool = True) -> List[str]:
    """Check that no sampled robot position lies inside a solid.

    A receiver inside furniture is physically meaningless: the specular
    reflection off the enclosing face points the wrong way, so paths are
    silently dropped and the predicted power can be wrong by tens of dB
    without any error being raised.  Call this on every route.

    Returns the list of problems found (empty when the route is clear).
    """
    p = positions.detach().numpy().reshape(-1, 3)
    problems: List[str] = []
    for group, (lo, hi) in furniture_boxes(margin).items():
        inside = np.all((p >= lo) & (p <= hi), axis=-1)
        if inside.any():
            idx = np.nonzero(inside)[0]
            problems.append(
                f"{len(idx)} position(s) inside '{group}' "
                f"(first at index {int(idx[0])}: {p[idx[0]].round(3).tolist()})")
    if problems and raise_on_error:
        raise ValueError("trajectory intersects furniture: " + "; ".join(problems))
    return problems


def plate_scene(plate_size=(1.0, 1.0), material="metal",
                center=(0.0, 0.0, 0.0)) -> Mesh:
    """A single finite reflector in free space (used for the Fig. 3 study).

    The plate lies in the ``z = center_z`` plane with an upward normal, so the
    specular point can be walked across a free edge (wedge index ``n = 2``).
    """
    return plate(center, plate_size, "z", material, "plate").weld()


def obstacle_scene(board_material="plastic_board", board_size=(1.2, 1.2),
                   board_x=1.0, target_material="metal",
                   target_center=(2.5, 0.0, 0.0), target_size=(0.6, 0.6, 0.6)) -> Mesh:
    """Radar -> thin board -> target, the NLOS occlusion setup of Fig. 15(a).

    The board is a vertical slab of the chosen material; the radar sees the
    target only through it, so the received level is governed by the slab
    transmission of Eq. 56-59.
    """
    m = plate((board_x, target_center[1], target_center[2]), board_size, "x",
              board_material, "board")
    m = m.merged(box(target_center, target_size, target_material, "target"))
    return m.weld()


def double_slit_scene(gap=0.6, slit=0.15, wall_x=1.5, height=2.0,
                      material="metal") -> Mesh:
    """Two coplanar half-screens forming a pair of slits.

    This is the classic double-slit interference scene used in Sec. 6.1 to
    verify gradients: the field behind the screen is entirely built from
    diffraction at four free edges, so it is exquisitely sensitive to the
    differentiability of the visibility term.
    """
    half = height / 2.0
    y_edges = [(-half, -gap / 2 - slit), (-gap / 2, gap / 2), (gap / 2 + slit, half)]
    quads, mats, groups = [], [], []
    for i, (y0, y1) in enumerate(y_edges):
        quads.append([(wall_x, y0, -half), (wall_x, y1, -half),
                      (wall_x, y1, half), (wall_x, y0, half)])
        mats.append(material)
        groups.append(f"screen{i}")
    return Mesh.from_quads(quads, mats, groups).weld()


# ---------------------------------------------------------------------------
# robot trajectories
# ---------------------------------------------------------------------------
@dataclass
class Trajectory:
    """Sampled robot path: positions, velocities and timestamps."""

    positions: torch.Tensor      # (N, 3)
    velocities: torch.Tensor     # (N, 3)
    times: torch.Tensor          # (N,)
    heading: torch.Tensor        # (N, 3) unit forward vector

    def __len__(self) -> int:
        """Number of sampled positions along the trajectory."""
        return int(self.positions.shape[0])

    @property
    def arclength(self) -> torch.Tensor:
        """Cumulative distance travelled at each sample [m]."""
        d = self.positions[1:] - self.positions[:-1]
        s = torch.cat([torch.zeros(1, dtype=FDTYPE), d.norm(dim=-1).cumsum(0)])
        return s


def waypoint_trajectory(waypoints: Sequence[Sequence[float]], speed: float = 0.5,
                        n_samples: int = 60) -> Trajectory:
    """Constant-speed path through waypoints, resampled uniformly in arclength.

    ``speed`` [m/s] sets the timestamps, which in turn set the Doppler shifts
    of App. E.3.
    """
    wp = as_t(waypoints)
    seg = wp[1:] - wp[:-1]
    seg_len = seg.norm(dim=-1)
    cum = torch.cat([torch.zeros(1, dtype=FDTYPE), seg_len.cumsum(0)])
    total = float(cum[-1])
    s = torch.linspace(0.0, total, n_samples, dtype=FDTYPE)

    pos, vel = [], []
    for si in s:
        j = int(torch.clamp(torch.searchsorted(cum, si, right=True) - 1, 0, len(seg) - 1))
        frac = (si - cum[j]) / seg_len[j].clamp_min(1e-9)
        pos.append(wp[j] + frac * seg[j])
        vel.append(seg[j] / seg_len[j].clamp_min(1e-9) * speed)
    positions = torch.stack(pos)
    velocities = torch.stack(vel)
    times = s / speed
    heading = velocities / velocities.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return Trajectory(positions, velocities, times, heading)


def survey_trajectory(z=0.9, speed=0.5, n_samples: int = 72,
                      validate: bool = True) -> Trajectory:
    """Default mobile-robot route through :func:`furnished_room`.

    The route starts next to the access point in the LoS half, drives north,
    rounds the open end of the partition at ``y > 3.5``, crosses into the
    shadowed half and returns along the far wall.  It therefore sweeps the
    robot through LoS, a shadow-boundary crossing and deep NLOS, which is the
    regime where a differentiable visibility term changes the answer.

    ``z`` is the height of the robot-mounted antenna, above the 0.755 m table.
    """
    wp = [(0.80, 0.80, z), (0.80, 3.00, z), (2.40, 4.30, z),
          (5.20, 4.30, z), (5.20, 1.00, z), (3.60, 0.80, z)]
    traj = waypoint_trajectory(wp, speed, n_samples)
    if validate:
        validate_trajectory(traj.positions)
    return traj


def grid_positions(size=(6.0, 5.0), z=1.0, nx=41, ny=35,
                   margin=0.15) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Regular sample grid for coverage / radio maps (Fig. 13)."""
    x = torch.linspace(margin, float(size[0]) - margin, nx, dtype=FDTYPE)
    y = torch.linspace(margin, float(size[1]) - margin, ny, dtype=FDTYPE)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1),
                       torch.full((nx * ny,), float(z), dtype=FDTYPE)], dim=-1)
    return pts, (nx, ny)