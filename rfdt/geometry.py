"""Differentiable mesh geometry for the reparameterised method of images.

Implements Appendix E.1 of RFDT:

  * ``S(v0, v1, v2)``  -> supporting plane (point, normal)         (Eq. 50)
  * ``mirror``          -> image of a point about a plane           (Eq. 51)
  * ``intersect_plane`` -> ray / infinite-plane intersection        (Eq. 53)
  * ``barycentric``     -> in-triangle coordinates (u, w)           (Eq. 55)

The key RFDT property is that reflection points are solved on the *infinite
supporting planes*, so they always exist and are smooth functions of the
vertices; validity is then handled by the transition weight instead of by a
hit/miss test (Sec. 3.2).

Also provides:
  * a vectorised Moller-Trumbore test for the BVH-style shadow queries used by
    the secondary-visibility term (Eq. 9),
  * the signed in-plane distance to the nearest triangle edge (the geometric
    argument of the RFDT weight),
  * wedge construction from mesh edge adjacency, including the wedge index
    ``n = W / pi`` needed by the diffraction coefficient (Eq. 7).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

FDTYPE = torch.float64
EPS = 1e-12


def as_t(x, dtype=FDTYPE) -> torch.Tensor:
    """Coerce a scalar, sequence or tensor to a float64 torch tensor."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.tensor(np.asarray(x, dtype=np.float64), dtype=dtype)


def normalize(v: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Unit-length version of ``v`` along ``dim``, safe against zero length."""
    return v / v.norm(dim=dim, keepdim=True).clamp_min(EPS)


# ---------------------------------------------------------------------------
# planes, mirrors, intersections  (Eq. 50-53)
# ---------------------------------------------------------------------------
def supporting_plane(tri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 50: ``S(v0,v1,v2) = (v0, (v1-v0) x (v2-v0))`` with a unit normal.

    ``tri`` has shape ``(..., 3, 3)``.  Returns ``(p0, n)``.
    """
    v0, v1, v2 = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    n = torch.cross(v1 - v0, v2 - v0, dim=-1)
    return v0, normalize(n)


def mirror(p: torch.Tensor, p0: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Eq. 51: image of ``p`` about the plane ``(p0, n)`` (``n`` unit)."""
    return p - 2.0 * ((p - p0) * n).sum(-1, keepdim=True) * n


def intersect_plane(a: torch.Tensor, b: torch.Tensor, p0: torch.Tensor,
                    n: torch.Tensor) -> torch.Tensor:
    """Eq. 53: intersection of line ``a -> b`` with the infinite plane.

    Differentiable everywhere except for rays exactly parallel to the plane,
    where the denominator is clamped.
    """
    d = b - a
    denom = (d * n).sum(-1, keepdim=True)
    denom = torch.where(denom.abs() < EPS, torch.full_like(denom, EPS), denom)
    t = ((p0 - a) * n).sum(-1, keepdim=True) / denom
    return a + t * d


def barycentric(p: torch.Tensor, tri: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 55: barycentric coordinates ``(u, w)`` of ``p`` in ``tri``."""
    v0 = tri[..., 0, :]
    e1 = tri[..., 1, :] - v0
    e2 = tri[..., 2, :] - v0
    t = p - v0
    d11 = (e1 * e1).sum(-1)
    d12 = (e1 * e2).sum(-1)
    d22 = (e2 * e2).sum(-1)
    t1 = (e1 * t).sum(-1)
    t2 = (e2 * t).sum(-1)
    det = (d11 * d22 - d12 * d12).clamp_min(EPS)
    u = (d22 * t1 - d12 * t2) / det
    w = (d11 * t2 - d12 * t1) / det
    return u, w


# ---------------------------------------------------------------------------
# signed edge distance: geometric argument of the RFDT weight
# ---------------------------------------------------------------------------
def signed_edge_distance(p: torch.Tensor, tri: torch.Tensor
                         ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Signed in-plane distance from ``p`` to the nearest edge of ``tri``.

    Positive inside the triangle, negative outside, zero exactly on an edge:
    a smooth, physically meaningful replacement for the sign of the
    barycentric in-triangle test of Eq. 3.

    Returns ``(d_min, edge_index, edge_dir)`` where ``edge_index`` selects the
    closest of the three edges (used to build the diffraction wedge) and
    ``edge_dir`` is that edge's unit direction.
    """
    _, n = supporting_plane(tri)
    ds, dirs = [], []
    for i in range(3):
        a = tri[..., i, :]
        b = tri[..., (i + 1) % 3, :]
        e = b - a
        eu = normalize(e)
        # inward in-plane normal of this edge
        m = normalize(torch.cross(n, eu, dim=-1))
        # orient m towards the opposite vertex so "inside" is positive
        c = tri[..., (i + 2) % 3, :]
        s = torch.sign(((c - a) * m).sum(-1, keepdim=True))
        s = torch.where(s == 0, torch.ones_like(s), s)
        m = m * s
        ds.append(((p - a) * m).sum(-1))
        dirs.append(eu)
    d = torch.stack(ds, dim=-1)                       # (..., 3)
    idx = torch.argmin(d, dim=-1)
    d_min = torch.gather(d, -1, idx.unsqueeze(-1)).squeeze(-1)
    # dirs come from `tri` alone and may be broadcast against `p`; expand them
    # to the broadcast batch shape before gathering the closest edge
    edge_dir = torch.stack(dirs, dim=-2).expand(*d.shape, 3)   # (..., 3, 3)
    edge_dir = torch.gather(
        edge_dir, -2, idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 3)
    ).squeeze(-2)
    return d_min, idx, edge_dir


def surface_edge_distance(p: torch.Tensor, boundary: torch.Tensor,
                          normal: torch.Tensor, centroid: torch.Tensor
                          ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Signed in-plane distance from ``p`` to a facet outline.

    The facet-level counterpart of :func:`signed_edge_distance`: positive
    inside, negative outside, and blind to interior triangulation edges.

    Parameters
    ----------
    p        : ``(..., 3)`` query points, assumed to lie in the facet plane.
    boundary : ``(E, 2, 3)`` outline edge endpoints.
    normal   : ``(3,)`` facet unit normal.
    centroid : ``(3,)`` a point strictly inside the facet, used once to orient
               each edge's inward direction.

    Returns ``(d_min, edge_dir)``: the signed distance to the nearest outline
    edge and that edge's unit direction (needed for the UTD ``beta0`` factor).

    The min-over-edges form is exact for convex facets, which is what the
    scene builders produce; for a non-convex outline it under-estimates the
    distance near reflex corners, which is conservative.
    """
    a = boundary[:, 0, :]                                  # (E, 3)
    b = boundary[:, 1, :]
    e = normalize(b - a)
    m = normalize(torch.cross(normal.expand_as(e), e, dim=-1))
    # orient every edge normal towards the facet interior
    s = torch.sign(((centroid - a) * m).sum(-1, keepdim=True))
    s = torch.where(s == 0, torch.ones_like(s), s)
    m = m * s
    d = ((p.unsqueeze(-2) - a) * m).sum(-1)                # (..., E)
    idx = torch.argmin(d, dim=-1)
    d_min = torch.gather(d, -1, idx.unsqueeze(-1)).squeeze(-1)
    edge_dir = e.expand(*d.shape, 3)
    edge_dir = torch.gather(
        edge_dir, -2, idx.unsqueeze(-1).unsqueeze(-1).expand(*idx.shape, 1, 3)
    ).squeeze(-2)
    return d_min, edge_dir


# ---------------------------------------------------------------------------
# Moller-Trumbore occlusion test (shadow rays / secondary visibility)
# ---------------------------------------------------------------------------
def ray_triangle(orig: torch.Tensor, direc: torch.Tensor, tri: torch.Tensor,
                 eps: float = 1e-9) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorised Moller-Trumbore.

    ``orig``/``direc``: ``(N, 3)``; ``tri``: ``(T, 3, 3)``.
    Returns ``(t, hit)`` of shape ``(N, T)``: ray parameter and hit mask.
    """
    v0 = tri[:, 0, :].unsqueeze(0)
    e1 = (tri[:, 1, :] - tri[:, 0, :]).unsqueeze(0)
    e2 = (tri[:, 2, :] - tri[:, 0, :]).unsqueeze(0)
    o = orig.unsqueeze(1)
    d = direc.unsqueeze(1)
    pv = torch.cross(d.expand_as(e2), e2.expand(d.shape[0], -1, -1), dim=-1)
    det = (e1 * pv).sum(-1)
    ok = det.abs() > eps
    inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
    tv = o - v0
    u = (tv * pv).sum(-1) * inv
    qv = torch.cross(tv, e1.expand_as(tv), dim=-1)
    v = (d * qv).sum(-1) * inv
    t = (e2 * qv).sum(-1) * inv
    hit = ok & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > eps)
    return t, hit


def segment_clearance(a: torch.Tensor, b: torch.Tensor, tri: torch.Tensor,
                      ignore: Optional[Sequence[int]] = None
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Blockage of segments ``a -> b`` by ``tri``.

    Returns ``(blocked, first_tri)``: whether any triangle (other than those in
    ``ignore``) lies strictly between the endpoints, and the index of the
    nearest such triangle (-1 if clear).
    """
    d = b - a
    length = d.norm(dim=-1, keepdim=True).clamp_min(EPS)
    t, hit = ray_triangle(a, d / length, tri)
    inside = hit & (t < length - 1e-6)
    if ignore is not None and len(ignore) > 0:
        mask = torch.zeros(tri.shape[0], dtype=torch.bool)
        mask[list(ignore)] = True
        inside = inside & (~mask).unsqueeze(0)
    big = torch.full_like(t, float("inf"))
    tt = torch.where(inside, t, big)
    tmin, imin = tt.min(dim=-1)
    blocked = torch.isfinite(tmin)
    first = torch.where(blocked, imin, torch.full_like(imin, -1))
    return blocked, first


# ---------------------------------------------------------------------------
# mesh container
# ---------------------------------------------------------------------------
@dataclass
class Wedge:
    """A diffracting edge shared by (at most) two faces, Fig. 3.

    ``n_index = W / pi`` with ``W`` the exterior (air-side) wedge angle:
    2 for a free edge / half plane, 1.5 for a convex 90-degree corner, 1 for a
    flat (coplanar) junction, which produces no diffraction and is skipped.
    """

    v0: int
    v1: int
    tri_a: int
    tri_b: Optional[int]
    n_index: float


@dataclass
class Surface:
    """A maximal set of coplanar, edge-connected triangles: one flat facet.

    Triangulation is a representation detail, not physics.  A wall split into
    two triangles has an interior diagonal that is *not* a geometric edge, and
    treating it as one is exactly the tessellation dependence the paper
    criticises in the soften-triangle baseline (App. F.1: "even for the same
    3D-scanned scene, different mesh topologies or triangulations may yield
    inconsistent outcomes").

    All edge-sensitive physics therefore runs per surface: the RFDT transition
    weight measures distance to the *facet* boundary, and a reflection off the
    facet is a single path rather than one per triangle.

    Attributes
    ----------
    tris     : member triangle indices; ``tris[0]`` defines the plane.
    boundary : ordered ``(v0, v1)`` vertex-index pairs of the real outline.
    """

    tris: List[int]
    boundary: List[Tuple[int, int]]
    material: str
    group: str
    solid: bool = False
    depth: float = 0.0

    @property
    def tri0(self) -> int:
        """Representative triangle, used to define the supporting plane."""
        return self.tris[0]


class Mesh:
    """Triangle soup with per-face materials and edge adjacency.

    Normals are expected to point into the air region (towards the room
    interior for walls, outwards for solid furniture); the scene builders in
    :mod:`rfdt.scenes` guarantee this, and it is what makes the wedge angles
    and the reflection side well defined.
    """

    def __init__(self, vertices: torch.Tensor, faces: np.ndarray,
                 mat_names: List[str], face_groups: Optional[List[str]] = None,
                 face_solid: Optional[List[bool]] = None,
                 face_depth: Optional[List[float]] = None):
        """Store the mesh and its per-face metadata.

        ``face_solid`` and ``face_depth`` describe whether a face bounds a
        closed volume and how deep that volume is along the face normal; they
        are what lets the tracer model penetration through a body correctly
        rather than as two independent slabs.
        """
        self.vertices = as_t(vertices)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.mat_names = list(mat_names)
        self.face_groups = list(face_groups) if face_groups is not None \
            else [f"face{i}" for i in range(len(self.faces))]
        #: True when the face bounds a closed solid, so a ray that crosses it
        #: enters or leaves a volume rather than passing through a thin slab.
        self.face_solid = list(face_solid) if face_solid is not None \
            else [False] * len(self.faces)
        #: Extent of that solid along the face normal [m]; a ray crossing the
        #: solid travels this far through it at normal incidence, so each of
        #: the two crossed faces accounts for half.
        self.face_depth = list(face_depth) if face_depth is not None \
            else [0.0] * len(self.faces)
        assert len(self.mat_names) == len(self.faces)
        self._wedges: Optional[List[Wedge]] = None
        self._surfaces: Optional[List["Surface"]] = None

    # basics ---------------------------------------------------------------
    @property
    def n_tri(self) -> int:
        """Number of triangles in the mesh."""
        return len(self.faces)

    def tri(self, vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``(T, 3, 3)`` vertex tensor; pass ``vertices`` to substitute a
        perturbed / learnable vertex set (keeps the graph for d/d vertex)."""
        v = self.vertices if vertices is None else vertices
        return v[torch.as_tensor(self.faces)]

    def normals(self, vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Unit outward normal of every triangle, shape ``(T, 3)``."""
        return supporting_plane(self.tri(vertices))[1]

    def centroids(self, vertices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Centroid of every triangle, shape ``(T, 3)``."""
        return self.tri(vertices).mean(dim=-2)

    def material_of(self, i: int) -> str:
        """Material name assigned to triangle ``i``."""
        return self.mat_names[i]

    def set_group_material(self, group: str, material: str) -> None:
        """Reassign the material of every face in a named group (e.g. "wall")."""
        for i, g in enumerate(self.face_groups):
            if g == group or g.startswith(group + "."):
                self.mat_names[i] = material

    def groups(self) -> List[str]:
        """Distinct surface group names, in order of first appearance."""
        out = []
        for g in self.face_groups:
            base = g.split(".")[0]
            if base not in out:
                out.append(base)
        return out

    # adjacency / wedges ---------------------------------------------------
    def wedges(self) -> List[Wedge]:
        """All diffracting edges of the mesh, computed once and cached.

        Edges shared by two coplanar triangles get ``n_index == 1`` and produce
        no diffraction; free edges get ``n_index == 2``.
        """
        if self._wedges is not None:
            return self._wedges
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for ti, f in enumerate(self.faces):
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                key = (min(a, b), max(a, b))
                edge_map.setdefault(key, []).append(ti)

        v = self.vertices.detach()
        tris = self.tri().detach()
        nrm = self.normals().detach()
        out: List[Wedge] = []
        for (a, b), ts in edge_map.items():
            e = normalize(v[b] - v[a])
            if len(ts) == 1:
                out.append(Wedge(a, b, ts[0], None, 2.0))
                continue
            ta, tb = ts[0], ts[1]
            n_a = nrm[ta]
            # in-plane direction from the edge towards each face's centroid
            def interior_dir(ti):
                """Unit in-plane direction from the edge towards triangle ``ti``'s centroid."""
                c = tris[ti].mean(dim=0)
                w = c - v[a]
                w = w - (w * e).sum() * e
                return normalize(w)
            d_a, d_b = interior_dir(ta), interior_dir(tb)
            ang = torch.acos((d_a * d_b).sum().clamp(-1.0, 1.0))
            # if face B lies on the outward side of face A the air wedge is
            # the small angle, otherwise it wraps the long way round
            if float((n_a * d_b).sum()) >= 0.0:
                W = float(ang)
            else:
                W = float(2.0 * torch.pi - ang)
            out.append(Wedge(a, b, ta, tb, W / float(np.pi)))
        self._wedges = out
        return out

    def wedges_of_face(self, ti: int) -> List[Wedge]:
        """Every wedge that triangle ``ti`` participates in."""
        return [w for w in self.wedges() if w.tri_a == ti or w.tri_b == ti]

    def surfaces(self, tol: float = 1e-6) -> List[Surface]:
        """Group triangles into coplanar, edge-connected facets.

        Two triangles sharing an edge are merged when their normals agree to
        ``tol``.  The resulting facet's boundary is the set of edges that are
        not shared internally, which is the outline that actually diffracts.
        """
        if self._surfaces is not None:
            return self._surfaces
        nrm = self.normals().detach()
        edge_map: Dict[Tuple[int, int], List[int]] = {}
        for ti, f in enumerate(self.faces):
            for i in range(3):
                a, b = int(f[i]), int(f[(i + 1) % 3])
                edge_map.setdefault((min(a, b), max(a, b)), []).append(ti)

        # union-find over coplanar neighbours
        parent = list(range(self.n_tri))

        def find(x):
            """Union-find root of ``x``, with path compression."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            """Merge the sets containing ``x`` and ``y``."""
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[max(rx, ry)] = min(rx, ry)

        for (a, b), ts in edge_map.items():
            if len(ts) != 2:
                continue
            ta, tb = ts
            if self.mat_names[ta] != self.mat_names[tb]:
                continue
            if float((nrm[ta] * nrm[tb]).sum()) > 1.0 - tol:
                union(ta, tb)

        groups: Dict[int, List[int]] = {}
        for ti in range(self.n_tri):
            groups.setdefault(find(ti), []).append(ti)

        out: List[Surface] = []
        for root, tris in sorted(groups.items()):
            member = set(tris)
            boundary: List[Tuple[int, int]] = []
            for ti in tris:
                f = self.faces[ti]
                for i in range(3):
                    a, b = int(f[i]), int(f[(i + 1) % 3])
                    shared = edge_map[(min(a, b), max(a, b))]
                    # an edge is on the outline unless another member triangle
                    # of this same facet also uses it
                    if not any(t in member and t != ti for t in shared):
                        boundary.append((a, b))
            out.append(Surface(sorted(tris), boundary, self.mat_names[tris[0]],
                               self.face_groups[tris[0]].split(".")[0],
                               self.face_solid[tris[0]], self.face_depth[tris[0]]))
        self._surfaces = out
        return out

    def surface_of_face(self) -> List[int]:
        """Map each triangle index to the index of its surface."""
        lut = [0] * self.n_tri
        for si, s in enumerate(self.surfaces()):
            for t in s.tris:
                lut[t] = si
        return lut

    # construction helpers -------------------------------------------------
    @staticmethod
    def from_quads(quads: Sequence[Sequence[Sequence[float]]], materials: Sequence[str],
                   groups: Optional[Sequence[str]] = None,
                   solid: Optional[Sequence[bool]] = None,
                   depth: Optional[Sequence[float]] = None) -> "Mesh":
        """Build a mesh from quads given as 4 corner points in winding order.

        The resulting normal follows the right-hand rule of the given winding,
        so callers control which side is "air".  ``solid`` and ``depth`` carry
        the per-quad volume metadata described on :class:`Mesh`.
        """
        verts: List[List[float]] = []
        faces: List[List[int]] = []
        mats: List[str] = []
        grps: List[str] = []
        sol: List[bool] = []
        dep: List[float] = []
        for qi, q in enumerate(quads):
            base = len(verts)
            verts.extend([list(map(float, p)) for p in q])
            faces.append([base + 0, base + 1, base + 2])
            faces.append([base + 0, base + 2, base + 3])
            mats.extend([materials[qi]] * 2)
            g = groups[qi] if groups is not None else f"quad{qi}"
            grps.extend([f"{g}.0", f"{g}.1"])
            sol.extend([bool(solid[qi]) if solid is not None else False] * 2)
            dep.extend([float(depth[qi]) if depth is not None else 0.0] * 2)
        return Mesh(as_t(verts), np.asarray(faces), mats, grps, sol, dep)

    def weld(self, tol: float = 1e-6) -> "Mesh":
        """Merge coincident vertices so that edge adjacency is well defined.

        ``from_quads`` duplicates corners per quad; without welding every edge
        would look like a free edge (``n = 2``) and the room shell would radiate
        spurious half-plane diffraction from each wall junction.
        """
        v = self.vertices.detach().numpy()
        key = np.round(v / tol).astype(np.int64)
        _, first, inverse = np.unique(key, axis=0, return_index=True,
                                      return_inverse=True)
        order = np.argsort(first)
        remap = np.empty(len(first), dtype=np.int64)
        remap[order] = np.arange(len(first))
        new_v = self.vertices[torch.as_tensor(first[order])]
        new_f = remap[inverse.reshape(-1)][self.faces]
        return Mesh(new_v, new_f, list(self.mat_names), list(self.face_groups),
                    list(self.face_solid), list(self.face_depth))

    def merged(self, other: "Mesh") -> "Mesh":
        """Concatenate another mesh into this one, preserving all metadata."""
        v = torch.cat([self.vertices, other.vertices], dim=0)
        f = np.concatenate([self.faces, other.faces + len(self.vertices)], axis=0)
        return Mesh(v, f, self.mat_names + other.mat_names,
                    self.face_groups + other.face_groups,
                    self.face_solid + other.face_solid,
                    self.face_depth + other.face_depth)


# ---------------------------------------------------------------------------
# primitive builders
# ---------------------------------------------------------------------------
def quad(p0, p1, p2, p3) -> List[List[float]]:
    """Collect four corner points into a quad, in winding order."""
    return [list(p0), list(p1), list(p2), list(p3)]


def box(center, size, material: str, group: str = "box",
        outward: bool = True) -> Mesh:
    """Axis-aligned box with normals pointing outwards (``outward=True``) or
    inwards (used for room shells)."""
    cx, cy, cz = map(float, center)
    sx, sy, sz = (float(s) / 2.0 for s in size)
    x0, x1 = cx - sx, cx + sx
    y0, y1 = cy - sy, cy + sy
    z0, z1 = cz - sz, cz + sz
    faces = {
        "xmin": (quad((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0)), (-1, 0, 0)),
        "xmax": (quad((x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)), (1, 0, 0)),
        "ymin": (quad((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)), (0, -1, 0)),
        "ymax": (quad((x0, y1, z0), (x0, y1, z1), (x1, y1, z1), (x1, y1, z0)), (0, 1, 0)),
        "zmin": (quad((x0, y0, z0), (x0, y1, z0), (x1, y1, z0), (x1, y0, z0)), (0, 0, -1)),
        "zmax": (quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)), (0, 0, 1)),
    }
    # extent of the box along each face normal, i.e. how far a ray entering
    # through that face travels inside the solid at normal incidence
    depths = {"xmin": 2 * sx, "xmax": 2 * sx, "ymin": 2 * sy,
              "ymax": 2 * sy, "zmin": 2 * sz, "zmax": 2 * sz}
    quads, mats, grps, sol, dep = [], [], [], [], []
    for name, (q, want) in faces.items():
        v = as_t(q)
        n = torch.cross(v[1] - v[0], v[2] - v[0], dim=-1)
        n = n / n.norm()
        if float((n * as_t(want)).sum()) < 0:
            q = [q[0], q[3], q[2], q[1]]           # flip winding
        if not outward:
            q = [q[0], q[3], q[2], q[1]]
        quads.append(q)
        mats.append(material)
        grps.append(f"{group}_{name}")
        # only an outward-facing box is a solid a ray can travel through; an
        # inward-facing one is a room shell, whose walls are thin slabs
        sol.append(bool(outward))
        dep.append(depths[name])
    return Mesh.from_quads(quads, mats, grps, sol, dep)


def plate(center, size, normal_axis: str, material: str, group: str = "plate",
          flip: bool = False) -> Mesh:
    """A single finite rectangular plate (one quad -> two triangles).

    The winding is chosen so the normal points along ``+normal_axis``; pass
    ``flip=True`` to reverse it.  This matters: a facet only reflects towards
    the side its normal points to, so a wall built facing away from the
    transmitter silently contributes no specular path at all.
    """
    cx, cy, cz = map(float, center)
    a, b = (float(s) / 2.0 for s in size)
    if normal_axis == "x":
        q = quad((cx, cy - a, cz - b), (cx, cy + a, cz - b), (cx, cy + a, cz + b), (cx, cy - a, cz + b))
    elif normal_axis == "y":
        q = quad((cx - a, cy, cz - b), (cx - a, cy, cz + b), (cx + a, cy, cz + b), (cx + a, cy, cz - b))
    else:
        q = quad((cx - a, cy - b, cz), (cx + a, cy - b, cz), (cx + a, cy + b, cz), (cx - a, cy + b, cz))
    if flip:
        q = [q[0], q[3], q[2], q[1]]
    return Mesh.from_quads([q], [material], [group])