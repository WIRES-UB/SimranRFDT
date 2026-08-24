"""Turn a DLoc occupancy-grid map into a floor-plan polygon.

The DLoc repository ships one grayscale occupancy grid per setup under
``ref/`` (``jacobs_default.png``, ``jacobs_aug16_1.png``, and so on).  White is
the free space the SLAM robot mapped, black is everything else.  These are the
"maps provided within" that the room outline should come from, rather than a
bounding box around the robot's route.

The pipeline is deliberately dependency-light, no OpenCV or scikit-image:

    threshold  ->  keep the largest free-space blob  ->  trace its boundary
               ->  simplify the contour  ->  scale pixels to metres

Scale is *not* inferred from the image.  Pixels become metres only when a
metres-per-pixel factor is supplied, which comes from the measurement file's
spatial axes.  Extracting a shape and guessing its size would produce a
plausible-looking room of the wrong dimensions, so :func:`to_metres` requires
the scale explicitly and there is no default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class FloorPlan:
    """A room outline extracted from an occupancy grid.

    ``polygon`` is closed implicitly (last vertex joins the first) and wound
    counter-clockwise, which is what the scene builder assumes when it decides
    which way a wall faces.
    """

    polygon: np.ndarray            # (n, 2)
    units: str                     # "pixels" or "metres"
    source: str
    grid_shape: Tuple[int, int]
    metres_per_pixel: Optional[float] = None

    @property
    def n_vertices(self) -> int:
        """Number of corners in the outline."""
        return int(self.polygon.shape[0])

    def area(self) -> float:
        """Shoelace area, in whatever units the polygon is in."""
        x, y = self.polygon[:, 0], self.polygon[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    def extent(self) -> Tuple[float, float]:
        """Bounding-box width and height of the outline."""
        lo, hi = self.polygon.min(axis=0), self.polygon.max(axis=0)
        return float(hi[0] - lo[0]), float(hi[1] - lo[1])

    def summary(self) -> dict:
        """Description for the run log."""
        w, h = self.extent()
        return {"source": os.path.basename(self.source), "units": self.units,
                "n_vertices": self.n_vertices, "area": self.area(),
                "extent": [w, h], "grid_shape": list(self.grid_shape),
                "metres_per_pixel": self.metres_per_pixel}


def load_occupancy(path: str, threshold: int = 128) -> np.ndarray:
    """Read an occupancy grid as a boolean array, True where the space is free.

    The DLoc maps are white-on-black, so anything brighter than ``threshold``
    is treated as free space.
    """
    from PIL import Image

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    img = Image.open(path).convert("L")
    # image row 0 is the top; flip so that y increases upwards, matching the
    # convention the dataset's coordinates use
    return np.flipud(np.array(img)) > threshold


def largest_component(free: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected region of free space.

    Occupancy grids often carry stray specks from SLAM noise.  Tracing the
    boundary of the whole image would follow those instead of the room, so the
    largest 4-connected blob is isolated first.
    """
    h, w = free.shape
    seen = np.zeros_like(free, dtype=bool)
    best: Optional[np.ndarray] = None
    best_size = 0
    for sy in range(h):
        for sx in range(w):
            if not free[sy, sx] or seen[sy, sx]:
                continue
            # iterative flood fill; recursion would blow the stack on big grids
            stack = [(sy, sx)]
            seen[sy, sx] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            if len(cells) > best_size:
                best_size = len(cells)
                mask = np.zeros_like(free)
                idx = np.array(cells)
                mask[idx[:, 0], idx[:, 1]] = True
                best = mask
    if best is None:
        raise ValueError("no free space found in the occupancy grid")
    return best


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill enclosed holes, so the outline is the outer boundary only.

    Interior obstacles show up as holes in the free-space mask.  They are real,
    but they belong in the scene as separate objects rather than as part of the
    room outline, so the outline is taken from the filled mask.
    """
    h, w = mask.shape
    outside = np.zeros_like(mask, dtype=bool)
    stack = []
    for x in range(w):
        for y in (0, h - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if not mask[y, x] and not outside[y, x]:
                outside[y, x] = True
                stack.append((y, x))
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx] and not outside[ny, nx]:
                outside[ny, nx] = True
                stack.append((ny, nx))
    return mask | ~outside


def trace_boundary(mask: np.ndarray) -> np.ndarray:
    """Moore-neighbourhood boundary trace of a filled binary region.

    Returns the outline as pixel coordinates in ``(x, y)`` order, walking the
    region once.  Simple and adequate here because the region is a single
    filled blob by this point.
    """
    h, w = mask.shape
    start = None
    for y in range(h):
        xs = np.nonzero(mask[y])[0]
        if xs.size:
            start = (y, int(xs[0]))
            break
    if start is None:
        raise ValueError("empty mask")

    # eight neighbours, clockwise from west
    nbrs = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    contour = [start]
    current = start
    backtrack = 0
    for _ in range(8 * mask.sum() + 16):
        found = False
        for k in range(8):
            d = nbrs[(backtrack + k) % 8]
            ny, nx = current[0] + d[0], current[1] + d[1]
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx]:
                # re-enter the search from the direction we came from
                backtrack = ((backtrack + k) + 5) % 8
                current = (ny, nx)
                found = True
                break
        if not found:
            break
        if current == start and len(contour) > 2:
            break
        contour.append(current)
    pts = np.array([[c[1], c[0]] for c in contour], dtype=float)
    return _ensure_ccw(pts)


def _signed_area(poly: np.ndarray) -> float:
    """Shoelace signed area; positive when the winding is counter-clockwise."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Return the polygon wound counter-clockwise.

    The scene builder derives wall normals from the winding, so this is not
    cosmetic: a clockwise outline would build a room inside out.
    """
    return poly if _signed_area(poly) > 0 else poly[::-1]


def simplify(poly: np.ndarray, epsilon: float) -> np.ndarray:
    """Ramer-Douglas-Peucker simplification of a closed polygon.

    A traced contour has one vertex per boundary pixel, which is far more
    detail than a ray tracer needs and would make every wall a separate facet.
    ``epsilon`` is the maximum deviation allowed, in the polygon's own units.
    """
    def rdp(points: np.ndarray) -> np.ndarray:
        """Recursive simplification of an open polyline."""
        if len(points) < 3:
            return points
        a, b = points[0], points[-1]
        ab = b - a
        norm = np.linalg.norm(ab)
        if norm < 1e-12:
            d = np.linalg.norm(points - a, axis=1)
        else:
            rel = points - a
            d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / norm
        i = int(np.argmax(d))
        if d[i] <= epsilon:
            return np.vstack([a, b])
        return np.vstack([rdp(points[:i + 1])[:-1], rdp(points[i:])])

    closed = np.vstack([poly, poly[:1]])
    out = rdp(closed)
    if len(out) > 1 and np.allclose(out[0], out[-1]):
        out = out[:-1]
    return _ensure_ccw(out)


def extract(path: str, epsilon_px: float = 2.0,
            threshold: int = 128) -> FloorPlan:
    """Full pipeline: occupancy grid image to a simplified pixel polygon.

    ``epsilon_px`` trades outline fidelity against facet count.  At 2 px the
    Jacobs map reduces from thousands of boundary pixels to a few dozen
    corners, which is what a ray tracer wants.
    """
    free = load_occupancy(path, threshold)
    blob = fill_holes(largest_component(free))
    contour = trace_boundary(blob)
    poly = simplify(contour, epsilon_px)
    return FloorPlan(polygon=poly, units="pixels", source=path,
                     grid_shape=tuple(free.shape))


def to_metres(plan: FloorPlan, metres_per_pixel: float,
              origin_px: Sequence[float] = (0.0, 0.0)) -> FloorPlan:
    """Scale a pixel polygon into metres.

    There is no default scale on purpose.  The occupancy grids carry no
    physical units, so the factor has to come from the measurement file's
    spatial axes.  Guessing it would produce a correctly shaped room of the
    wrong size, which is harder to notice than an obviously wrong shape.
    """
    if metres_per_pixel <= 0:
        raise ValueError("metres_per_pixel must be positive")
    poly = (plan.polygon - np.asarray(origin_px, dtype=float)) * metres_per_pixel
    return FloorPlan(polygon=poly, units="metres", source=plan.source,
                     grid_shape=plan.grid_shape,
                     metres_per_pixel=metres_per_pixel)


def scale_from_extent(plan: FloorPlan, known_width_m: float) -> float:
    """Infer metres per pixel from a known physical width.

    A stopgap for when the measurement file is not yet available: the DLoc
    paper quotes collection areas of 18 x 8 m for Jacobs and 8 x 5 m for
    Atkinson.  Those are the areas the robot covered, not necessarily the full
    mapped extent, so a scale derived this way is approximate and any scene
    built on it should say so.
    """
    w, _ = plan.extent()
    if w <= 0:
        raise ValueError("degenerate polygon")
    return known_width_m / w


def triangulate(poly: np.ndarray) -> List[Tuple[int, int, int]]:
    """Ear-clipping triangulation of a simple polygon.

    Needed for the floor and ceiling, which are one non-convex facet each.
    The mesh merges coplanar triangles back into a single surface afterwards,
    so the triangulation is an implementation detail the physics never sees.
    """
    n = len(poly)
    if n < 3:
        return []
    idx = list(range(n)) if _signed_area(poly) > 0 else list(range(n))[::-1]

    def is_convex(a, b, c):
        """True when corner b turns left, i.e. is convex in a CCW polygon."""
        ab, bc = poly[b] - poly[a], poly[c] - poly[b]
        return ab[0] * bc[1] - ab[1] * bc[0] > 0

    def inside(p, a, b, c):
        """Point-in-triangle test by barycentric sign."""
        v0, v1, v2 = poly[c] - poly[a], poly[b] - poly[a], p - poly[a]
        d00, d01, d02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
        d11, d12 = np.dot(v1, v1), np.dot(v1, v2)
        den = d00 * d11 - d01 * d01
        if abs(den) < 1e-18:
            return False
        u = (d11 * d02 - d01 * d12) / den
        v = (d00 * d12 - d01 * d02) / den
        return u >= -1e-12 and v >= -1e-12 and u + v <= 1 + 1e-12

    tris: List[Tuple[int, int, int]] = []
    guard = 0
    while len(idx) > 3 and guard < 10 * n:
        guard += 1
        clipped = False
        for i in range(len(idx)):
            a, b, c = idx[i - 1], idx[i], idx[(i + 1) % len(idx)]
            if not is_convex(a, b, c):
                continue
            if any(inside(poly[j], a, b, c) for j in idx if j not in (a, b, c)):
                continue
            tris.append((a, b, c))
            idx.pop(i)
            clipped = True
            break
        if not clipped:
            break            # degenerate; fall through with what we have
    if len(idx) == 3:
        tris.append((idx[0], idx[1], idx[2]))
    return tris