"""RFDT differentiable RF ray tracer.

Forward model (Eq. 1):

    E(p_rx) = sum_{P in Omega(p_tx, p_rx)}  W(P) * A_P * exp(-j k L_P)

with three RFDT-specific ingredients:

1. **Reparameterised method of images** (Sec. 3.2, App. E.1).  Reflection points
   are solved on the *infinite supporting planes* of the candidate facets, so
   ``L_P``, ``A_P`` and the phase are continuous and differentiable w.r.t.
   every vertex, ``p_tx`` and ``p_rx``: paths are never created or destroyed.

2. **Physically consistent path validity** (Sec. 3.3, Eq. 11).  The binary
   in-triangle test is replaced by the product of UTD transition functions

       W_RFDT(P) = prod_i F(k L_i a_i) ,

   which decays continuously to zero as a reflection point approaches an edge.
   ``weighting="heaviside"`` (Eq. 3) and ``weighting="sigmoid"`` (Eq. 4)
   reproduce the two baselines for comparison.

3. **Edge diffraction** (Eq. 6, 7, 9) supplies the energy that the specular
   term gives up at a wedge, so the *total* field is continuous and
   energy-conserving (Fig. 3, right).  Secondary visibility (occlusion by
   unrelated objects) uses the same transition function, plus a transmission
   term so that penetration through a material is modelled.

Everything edge-sensitive operates on *surfaces* (maximal coplanar facets),
never on individual triangles.  A wall split into two triangles has an
interior diagonal that is not a geometric edge; measuring the transition
weight against it would put a spurious null down the middle of the wall and
make the result depend on the triangulation.

Two-stage path search, as in Sec. 3.2: a cheap detached pass prunes candidate
facet sequences, then the exact differentiable solve runs only on the
survivors.  Sequences whose reflection point falls slightly *outside* a facet
are deliberately kept, because they are exactly the transition-region paths
that carry the gradient.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from . import transition as tf
from .antennas import Receiver, Transmitter
from .geometry import (Mesh, Surface, Wedge, as_t, intersect_plane, mirror,
                       normalize, supporting_plane, surface_edge_distance)
from .materials import (C0, Material, MaterialParams, absorption_factor,
                        fresnel, get_material, interface_transmission,
                        reflection_coefficient, slab_transmission)

FDTYPE = torch.float64
CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
# configuration and results
# ---------------------------------------------------------------------------
@dataclass
class TracerConfig:
    """Switches controlling the physics and the path search."""

    #: maximum number of specular reflections per path
    max_order: int = 2
    #: path-validity model: "rfdt" (Eq. 11), "heaviside" (Eq. 3), "sigmoid" (Eq. 4)
    weighting: str = "rfdt"
    #: sharpness k of the soften-triangle baseline
    sigmoid_k: float = 40.0
    #: include first-order wedge diffraction (Eq. 6, 7)
    enable_diffraction: bool = True
    #: smooth secondary-visibility weight of Eq. 9, else a hard binary mask
    smooth_occlusion: bool = True
    #: add the field transmitted *through* blocking material (App. E.2)
    enable_transmission: bool = True
    #: scalar polarisation: "perp" (TE, default), "par" (TM), "unpolarised"
    polarisation: str = "perp"
    #: surface-roughness model reducing the coherent specular field:
    #: "miller_brown" (default), "rayleigh", or "none" for the ideally smooth
    #: interface the Fresnel equations assume.  Not part of RFDT; see
    #: materials.roughness_factor for the derivation, for why the default is
    #: the Miller-Brown form rather than the more commonly quoted Rayleigh
    #: one, and for the energy bookkeeping this deliberately leaves open.
    surface_roughness: str = "miller_brown"
    #: keep candidates whose reflection point is at most this far outside a
    #: facet [m]; those are the transition-region paths that carry gradient
    prune_margin: float = 0.5
    #: skip diffraction from concave junctions (n < 1), outside UTD validity
    min_wedge_index: float = 1.05


@dataclass
class Paths:
    """Per-path propagation data for a batch of receiver positions.

    Shapes are ``(R, P)`` for scalars and ``(R, P, 3)`` for directions, with
    ``R`` receivers and ``P`` paths.  ``gain`` is the complex field ratio, so
    ``|sum_p gain|**2`` is the received/transmitted power ratio.
    """

    gain: torch.Tensor
    length: torch.Tensor
    delay: torch.Tensor
    dep_dir: torch.Tensor
    arr_dir: torch.Tensor
    order: torch.Tensor
    kind: List[str] = field(default_factory=list)
    #: geometry of each path as ``(R, n_nodes, 3)``: transmitter, every
    #: interaction point in order, then receiver.  Kept so a path can be drawn
    #: or inspected, not only summed.
    nodes: List[torch.Tensor] = field(default_factory=list)

    def field(self) -> torch.Tensor:
        """Coherent sum over paths, Eq. 1.

        RFDT insists on coherent (complex) summation rather than the scalar
        power summation used by typical coverage tools, because the phase
        interference is what keeps the field continuous at shadow boundaries
        (Sec. 6.2, App. A "Coherent vs Non-Coherent RT").
        """
        return self.gain.sum(dim=-1)

    def power_dbm(self, tx_power_dbm: float) -> torch.Tensor:
        """Received power [dBm] from the coherently combined field."""
        p = (self.field().abs() ** 2).clamp_min(1e-30)
        return tx_power_dbm + 10.0 * torch.log10(p)

    def noncoherent_power_dbm(self, tx_power_dbm: float) -> torch.Tensor:
        """Received power [dBm] under scalar power summation.

        The approximation used by Sionna's ``radiomap`` solver (App. C.4);
        provided so the two can be compared directly.
        """
        p = (self.gain.abs() ** 2).sum(-1).clamp_min(1e-30)
        return tx_power_dbm + 10.0 * torch.log10(p)

    def n_paths(self) -> int:
        """Number of propagation paths held."""
        return int(self.gain.shape[-1])

    def strongest(self, n: int = 10):
        """Indices of the ``n`` strongest paths, by mean power over receivers."""
        p = (self.gain.abs() ** 2).mean(dim=0)
        return torch.argsort(p, descending=True)[:n]

    def select(self, mask) -> "Paths":
        """Sub-select paths by boolean mask or index tensor."""
        idx = torch.as_tensor(mask)
        if idx.dtype == torch.bool:
            keep = [i for i, m in enumerate(idx.tolist()) if m]
        else:
            keep = idx.tolist()
        return Paths(self.gain[..., keep], self.length[..., keep],
                     self.delay[..., keep], self.dep_dir[..., keep, :],
                     self.arr_dir[..., keep, :], self.order[keep],
                     [self.kind[i] for i in keep],
                     [self.nodes[i] for i in keep] if self.nodes else [])


def _cat_paths(items: List[Paths]) -> Paths:
    """Concatenate path families along the path axis."""
    items = [p for p in items if p is not None and p.n_paths() > 0]
    if not items:
        raise ValueError("no propagation paths found")
    return Paths(
        torch.cat([p.gain for p in items], dim=-1),
        torch.cat([p.length for p in items], dim=-1),
        torch.cat([p.delay for p in items], dim=-1),
        torch.cat([p.dep_dir for p in items], dim=-2),
        torch.cat([p.arr_dir for p in items], dim=-2),
        torch.cat([p.order for p in items], dim=0),
        [k for p in items for k in p.kind],
        [n for p in items for n in p.nodes],
    )


# ---------------------------------------------------------------------------
# tracer
# ---------------------------------------------------------------------------
class RFDTracer:
    """Differentiable RF ray tracer over a triangle mesh.

    Reflection and occlusion are computed per coplanar facet
    (:class:`~rfdt.geometry.Surface`); diffraction is computed per wedge.
    """

    def __init__(self, mesh: Mesh, config: Optional[TracerConfig] = None):
        """Bind a mesh and a configuration, and precompute facet data.

        Facet grouping and boundary indices are computed once here because
        they depend only on the mesh, not on the transmitter or receiver.
        """
        self.mesh = mesh
        self.cfg = config or TracerConfig()
        self.surfaces: List[Surface] = mesh.surfaces()
        self._materials = {n: get_material(n) for n in set(mesh.mat_names)}
        # boundary vertex indices per surface, as (E, 2) arrays
        self._bnd = [np.asarray(s.boundary, dtype=np.int64) for s in self.surfaces]

    # geometry helpers -----------------------------------------------------
    def _plane(self, si: int, vertices=None):
        """Supporting plane ``(p0, n)`` of surface ``si`` (Eq. 50)."""
        tri = self.mesh.tri(vertices)[self.surfaces[si].tri0]
        return supporting_plane(tri)

    def _boundary(self, si: int, vertices=None) -> torch.Tensor:
        """Outline edge endpoints of surface ``si`` as ``(E, 2, 3)``."""
        v = self.mesh.vertices if vertices is None else vertices
        return v[torch.as_tensor(self._bnd[si])]

    def _centroid(self, si: int, vertices=None) -> torch.Tensor:
        """Interior reference point of surface ``si``, used to orient edges."""
        return self._boundary(si, vertices).reshape(-1, 3).mean(dim=0)

    def _edge_distance(self, p: torch.Tensor, si: int, vertices=None):
        """Signed distance from ``p`` to the outline of surface ``si``."""
        _, n = self._plane(si, vertices)
        return surface_edge_distance(p, self._boundary(si, vertices), n,
                                     self._centroid(si, vertices))

    def material(self, si: int) -> Material:
        """Material of surface ``si``."""
        return self._materials[self.surfaces[si].material]

    def _params_for(self, si: int, overrides: Optional[Dict[str, MaterialParams]]):
        """Learnable material override for surface ``si``, if any."""
        if not overrides:
            return None
        return overrides.get(self.surfaces[si].material)

    # ------------------------------------------------------------------
    # stage 1: candidate sequence pruning (detached, Sec. 3.2)
    # ------------------------------------------------------------------
    def candidate_sequences(self, p_tx: torch.Tensor, p_rx: torch.Tensor,
                            order: int) -> List[Tuple[int, ...]]:
        """Facet sequences worth solving exactly, for a batch of receivers.

        A cheap detached pass: a sequence survives if, for at least one
        receiver, every bounce lands within ``prune_margin`` of its facet and
        the ray actually reflects off the front side.  Because the exact solve
        assigns weight ``F(x) = 0`` to any reflection point outside its facet,
        widening the margin can only add zero-weight paths, which is verified
        in ``tests/test_tracer.py``.
        """
        S = len(self.surfaces)
        tx = p_tx.detach().reshape(1, 3)
        rx = p_rx.detach().reshape(-1, 3)
        planes = [self._plane(si) for si in range(S)]
        p0 = torch.stack([p for p, _ in planes]).detach()
        nn = torch.stack([n for _, n in planes]).detach()

        front_tx = ((tx - p0) * nn).sum(-1) > 1e-6                     # (S,)
        front_rx = ((rx.unsqueeze(1) - p0.unsqueeze(0))
                    * nn.unsqueeze(0)).sum(-1) > 1e-6                  # (R, S)

        cands: List[Tuple[int, ...]] = []
        for s in itertools.product(range(S), repeat=order):
            if any(s[i] == s[i + 1] for i in range(order - 1)):
                continue
            if not bool(front_tx[s[0]]) or not bool(front_rx[:, s[-1]].any()):
                continue
            pts = self._solve_reflection_points(tx.expand(rx.shape[0], 3), rx, s,
                                                detached=True)
            if bool(self._feasible(tx.expand(rx.shape[0], 3), rx, s, pts).any()):
                cands.append(s)
        return cands

    def _feasible(self, p_tx, p_rx, seq, pts) -> torch.Tensor:
        """Detached mask: does this sequence reflect plausibly for each Rx?"""
        m = len(seq)
        good = torch.ones(pts.shape[0], dtype=torch.bool)
        for i, si in enumerate(seq):
            d, _ = self._edge_distance(pts[:, i, :], si)
            good = good & (d > -self.cfg.prune_margin)
            _, n = self._plane(si)
            inc = pts[:, i, :] - (p_tx if i == 0 else pts[:, i - 1, :])
            out = (p_rx if i == m - 1 else pts[:, i + 1, :]) - pts[:, i, :]
            good = good & ((inc * n).sum(-1) < 0) & ((out * n).sum(-1) > 0)
        return good

    # ------------------------------------------------------------------
    # stage 2: exact reparameterised solve (Eq. 50-54)
    # ------------------------------------------------------------------
    def _solve_reflection_points(self, p_tx: torch.Tensor, p_rx: torch.Tensor,
                                 seq: Sequence[int], vertices=None,
                                 detached: bool = False) -> torch.Tensor:
        """Reflection points on the infinite supporting planes, Eq. 52-53.

        The receiver is mirrored back through the facet planes in reverse
        order, then a forward sweep intersects each plane in turn.  For a
        sequence ``(D1, ..., Dm)``:

            K_i     = p_rx mirrored through D_m, D_{m-1}, ..., D_i
            p_ref^i = intersect(line(p_ref^{i-1}, K_i), plane(D_i))

        (The paper's Eq. 53 prints ``p_rx^{m-i}``; the consistent index is
        ``K_i``, i.e. the image that still has planes ``D_i .. D_m`` folded in,
        which is what reproduces the classical mirror construction.)

        Returns ``(R, m, 3)``.  The result always exists, which is precisely
        the property that removes the discontinuity of Sec. 3.1.
        """
        m = len(seq)
        planes = [self._plane(si, vertices) for si in seq]
        if detached:
            planes = [(p.detach(), n.detach()) for p, n in planes]

        images: List[torch.Tensor] = [None] * m
        cur = p_rx
        for i in range(m - 1, -1, -1):
            p0, n = planes[i]
            cur = mirror(cur, p0, n)
            images[i] = cur

        pts, prev = [], p_tx
        for i in range(m):
            p0, n = planes[i]
            prev = intersect_plane(prev, images[i], p0, n)
            pts.append(prev)
        return torch.stack(pts, dim=-2)

    # ------------------------------------------------------------------
    # secondary visibility (Eq. 9) and material penetration (App. E.2)
    # ------------------------------------------------------------------
    def _segment_weight(self, a: torch.Tensor, b: torch.Tensor, k: float,
                        exclude: Sequence[int] = (), vertices=None,
                        mat_overrides=None) -> torch.Tensor:
        """Smooth transmission / shadow weight for the segment ``a -> b``.

        Every facet except those in ``exclude`` is tested.  For a facet the
        segment crosses, the "goes around" term is ``F(k L a(tau))`` with
        ``tau`` the signed clearance to the silhouette, and the "goes through"
        term is the material transmission:

            w = F(x(tau+)) + T_mat * F(x(tau-))

        Both vanish at ``tau = 0``, where the diffracted field of Eq. 6 takes
        over, giving continuity with energy conservation.

        ``T_mat`` distinguishes two cases (App. E.2).  A thin plate or a room
        wall is a slab, so one crossing applies both interfaces and the full
        thickness.  A closed solid is a volume: the segment crosses two of its
        facets, so each contributes one interface and half the traversal,
        which together give two interfaces and one full path through the body.
        """
        f_hz = C0 * k / (2.0 * np.pi)
        keep = [si for si in range(len(self.surfaces)) if si not in set(exclude)]
        shp = a.shape[:-1]
        w = torch.ones(shp, dtype=CDTYPE)
        if not keep:
            return w

        d = b - a
        seg_len = d.norm(dim=-1).clamp_min(1e-9)
        du = d / seg_len.unsqueeze(-1)

        for si in keep:
            p0, n = self._plane(si, vertices)
            denom = (d * n).sum(-1)
            active = denom.abs() > 1e-9
            s = ((p0 - a) * n).sum(-1) / torch.where(active, denom,
                                                     torch.ones_like(denom))
            crosses = active & (s > 1e-6) & (s < 1.0 - 1e-6)
            if not bool(crosses.any()):
                continue

            hit = a + s.unsqueeze(-1) * d
            d_edge, _ = self._edge_distance(hit, si, vertices)
            tau = -d_edge                        # > 0 when the segment misses

            if not self.cfg.smooth_occlusion:
                hard = torch.where(tau > 0, torch.ones_like(tau), torch.zeros_like(tau))
                w = w * torch.where(crosses, hard.to(CDTYPE), torch.ones_like(w))
                continue

            s1 = (s * seg_len).clamp_min(1e-6)
            s2 = ((1.0 - s) * seg_len).clamp_min(1e-6)
            L = tf.distance_parameter(s1, s2)
            w_face = tf.transition_F(tf.edge_argument(tau.clamp_min(0.0), L, k))

            if self.cfg.enable_transmission:
                surf = self.surfaces[si]
                mat = self.material(si)
                params = self._params_for(si, mat_overrides)
                cos_ti = (du * n).sum(-1).abs().clamp(1e-6, 1.0)
                f_block = tf.transition_F(
                    tf.edge_argument((-tau).clamp_min(0.0), L, k))
                if surf.solid:
                    # one interface plus half the traversal per crossed face;
                    # the reciprocal (direction-independent) interface form is
                    # used so that grazing clips do not depend on ray direction
                    cos_tt = fresnel(f_hz, cos_ti, mat, params2=params)["cos_tt"]
                    t_mat = (interface_transmission(
                        f_hz, cos_ti, mat,
                        polarisation=self.cfg.polarisation, params=params)
                        * absorption_factor(f_hz, 0.5 * surf.depth, mat,
                                            cos_tt, params))
                else:
                    t_mat = slab_transmission(f_hz, cos_ti, mat,
                                              polarisation=self.cfg.polarisation,
                                              params=params)
                w_face = w_face + t_mat * f_block

            w = w * torch.where(crosses, w_face, torch.ones_like(w_face))
        return w

    # ------------------------------------------------------------------
    # path families
    # ------------------------------------------------------------------
    def _los(self, tx: Transmitter, rx_pos: torch.Tensor, rx: Receiver,
             vertices=None, mat_overrides=None) -> Paths:
        """Direct line-of-sight path, attenuated by the secondary-visibility
        weight so that it fades smoothly into a shadow instead of switching."""
        k, lam = tx.k, tx.wavelength
        p_tx = tx.position.reshape(1, 3).expand_as(rx_pos)
        d = rx_pos - p_tx
        L = d.norm(dim=-1).clamp_min(1e-6)
        u = d / L.unsqueeze(-1)
        g = tx.antenna.field_gain(u) * rx.antenna.field_gain(-u)
        w = self._segment_weight(p_tx, rx_pos, k, (), vertices, mat_overrides)
        amp = (lam / (4.0 * np.pi * L)) * g
        gain = w * amp.to(CDTYPE) * torch.exp(-1j * (k * L).to(CDTYPE))
        return Paths(gain.unsqueeze(-1), L.unsqueeze(-1), (L / C0).unsqueeze(-1),
                     u.unsqueeze(-2), u.unsqueeze(-2),
                     torch.zeros(1, dtype=torch.long), ["los"],
                     [torch.stack([p_tx, rx_pos], dim=-2)])

    def _specular(self, tx: Transmitter, rx_pos: torch.Tensor, rx: Receiver,
                  seq: Tuple[int, ...], vertices=None, mat_overrides=None) -> Paths:
        """One specular reflection sequence, fully differentiable (Eq. 10, 11, 17).

        Path length, amplitude and phase come from the reparameterised
        construction; the Fresnel coefficient of Eq. 56 is applied at each
        bounce; validity is the product of transition weights of Eq. 11.
        """
        k, lam, f_hz = tx.k, tx.wavelength, tx.frequency
        R = rx_pos.shape[0]
        p_tx = tx.position.reshape(1, 3).expand(R, 3)

        pts = self._solve_reflection_points(p_tx, rx_pos, seq, vertices)
        nodes = [p_tx] + [pts[:, i, :] for i in range(len(seq))] + [rx_pos]
        segs = [nodes[i + 1] - nodes[i] for i in range(len(nodes) - 1)]
        lens = [s.norm(dim=-1).clamp_min(1e-9) for s in segs]
        dirs = [s / l.unsqueeze(-1) for s, l in zip(segs, lens)]
        L_tot = torch.stack(lens, dim=-1).sum(-1)

        weight = torch.ones(R, dtype=CDTYPE)
        refl = torch.ones(R, dtype=CDTYPE)
        for i, si in enumerate(seq):
            _, n = self._plane(si, vertices)
            cos_ti = (-dirs[i] * n).sum(-1).abs().clamp(1e-6, 1.0)
            refl = refl * reflection_coefficient(
                f_hz, cos_ti, self.material(si),
                polarisation=self.cfg.polarisation,
                params=self._params_for(si, mat_overrides),
                roughness=self.cfg.surface_roughness)

            d_edge, e_dir = self._edge_distance(pts[:, i, :], si, vertices)
            if self.cfg.weighting == "heaviside":
                weight = weight * tf.weight_heaviside(d_edge)
            elif self.cfg.weighting == "sigmoid":
                weight = weight * tf.weight_sigmoid(d_edge, self.cfg.sigmoid_k)
            else:
                # beta0 is the angle the ray makes with the nearest edge.  The
                # incident and reflected rays make equal angles with the
                # surface normal but not, in general, with the edge, so taking
                # either one alone would make the weight depend on which end
                # of the path is the transmitter.  The geometric mean of the
                # two is symmetric under path reversal, which keeps the
                # simulator reciprocal to machine precision.
                sin_in = torch.sqrt(
                    (1.0 - (dirs[i] * e_dir).sum(-1) ** 2).clamp_min(1e-12))
                sin_out = torch.sqrt(
                    (1.0 - (dirs[i + 1] * e_dir).sum(-1) ** 2).clamp_min(1e-12))
                sin_b0 = torch.sqrt(sin_in * sin_out)
                Lp = tf.distance_parameter(lens[i], lens[i + 1], sin_b0)
                weight = weight * tf.weight_rfdt(
                    tf.edge_argument(d_edge, Lp, k, sin_b0))

        # secondary visibility of every segment, excluding the facets the path
        # is deliberately reflecting off
        for i in range(len(segs)):
            excl = {seq[i - 1] if i > 0 else -1, seq[i] if i < len(seq) else -1}
            weight = weight * self._segment_weight(
                nodes[i], nodes[i + 1], k, tuple(e for e in excl if e >= 0),
                vertices, mat_overrides)

        g = tx.antenna.field_gain(dirs[0]) * rx.antenna.field_gain(-dirs[-1])
        amp = (lam / (4.0 * np.pi * L_tot)) * g
        gain = weight * refl * amp.to(CDTYPE) * torch.exp(-1j * (k * L_tot).to(CDTYPE))
        return Paths(gain.unsqueeze(-1), L_tot.unsqueeze(-1),
                     (L_tot / C0).unsqueeze(-1), dirs[0].unsqueeze(-2),
                     dirs[-1].unsqueeze(-2),
                     torch.full((1,), len(seq), dtype=torch.long),
                     ["refl" + "".join(f"-{s}" for s in seq)],
                     [torch.stack(nodes, dim=-2)])

    def _diffraction(self, tx: Transmitter, rx_pos: torch.Tensor, rx: Receiver,
                     wedge: Wedge, vertices=None, mat_overrides=None) -> Paths:
        """First-order wedge diffraction, Eq. 6 with the coefficient of Eq. 7.

        The diffraction point is the Fermat point of Eq. 40.  For a straight
        edge the stationarity condition of Eq. 42 has the closed form

            t* = (h_rx t_tx + h_tx t_rx) / (h_tx + h_rx)

        with ``t`` the projections of the endpoints onto the edge and ``h``
        their perpendicular distances: an exact, differentiable solution,
        so no iterative root find is needed and the gradient is clean.
        """
        k, lam = tx.k, tx.wavelength
        R = rx_pos.shape[0]
        v = self.mesh.vertices if vertices is None else vertices
        v0, v1 = v[wedge.v0], v[wedge.v1]
        e_vec = v1 - v0
        e_len = e_vec.norm().clamp_min(1e-9)
        e = e_vec / e_len
        p_tx = tx.position.reshape(1, 3).expand(R, 3)

        def decompose(p):
            """Projection along the edge and perpendicular distance to it."""
            r = p - v0
            t = (r * e).sum(-1)
            return t, (r - t.unsqueeze(-1) * e).norm(dim=-1).clamp_min(1e-9)

        t_a, h_a = decompose(p_tx)
        t_b, h_b = decompose(rx_pos)
        t_star = ((h_b * t_a + h_a * t_b) / (h_a + h_b).clamp_min(1e-12))
        t_star = t_star.clamp(0.0, float(e_len))
        p_d = v0 + t_star.unsqueeze(-1) * e

        s_p = (p_d - p_tx).norm(dim=-1).clamp_min(1e-6)      # s'
        s_o = (rx_pos - p_d).norm(dim=-1).clamp_min(1e-6)    # s
        u_in = (p_d - p_tx) / s_p.unsqueeze(-1)
        u_out = (rx_pos - p_d) / s_o.unsqueeze(-1)
        # Keller's cone: the incident and diffracted rays make the same angle
        # beta0 with the edge, so either would do.  They differ only where
        # t_star has been clamped to a finite edge's endpoint, and there taking
        # just one of them would make the result depend on which end is the
        # source.  The geometric mean is reciprocal and identical on the cone.
        sin_in = torch.sqrt((1.0 - (u_in * e).sum(-1) ** 2).clamp_min(1e-12))
        sin_out = torch.sqrt((1.0 - (u_out * e).sum(-1) ** 2).clamp_min(1e-12))
        sin_b0 = torch.sqrt(sin_in * sin_out)
        L = tf.distance_parameter(s_p, s_o, sin_b0)

        # local wedge frame: b1 along face A's interior direction, b2 air-side
        tris = self.mesh.tri(vertices)
        w1 = tris[wedge.tri_a].mean(dim=0) - v0
        b1 = normalize(w1 - (w1 * e).sum() * e)
        b2 = normalize(torch.cross(e, b1, dim=-1))
        n_a = supporting_plane(tris[wedge.tri_a])[1]
        if float((n_a * b2).sum()) < 0:
            b2 = -b2

        def wedge_angle(u):
            """Angle of ``u`` about the edge, measured from face A into the air."""
            up = normalize(u - (u * e).sum(-1, keepdim=True) * e)
            ang = torch.atan2((up * b2).sum(-1), (up * b1).sum(-1))
            return torch.where(ang < 0, ang + 2.0 * np.pi, ang)

        phi_i = wedge_angle(-u_in)
        phi_d = wedge_angle(u_out)
        n_idx = torch.as_tensor(wedge.n_index, dtype=FDTYPE)

        # finite-conductivity UTD: weight the reflection-boundary terms by the
        # Fresnel coefficients of the two wedge faces, which makes diffraction
        # material dependent rather than assuming a perfect conductor
        lut = self.mesh.surface_of_face()
        si_a = lut[wedge.tri_a]
        si_b = lut[wedge.tri_b] if wedge.tri_b is not None else si_a
        # Luebbers' heuristic evaluates each face's coefficient at its grazing
        # angle.  Using the incidence angle alone would make the coefficient
        # depend on which end of the path is the source; the geometric mean of
        # the incidence and diffraction angles is symmetric under path
        # reversal and reduces to the standard value when they coincide, which
        # is the specular-on-edge case where these terms matter most.
        def _sym_grazing(angle_fn):
            """Reciprocal grazing cosine from the two ray angles."""
            return torch.sqrt(angle_fn(phi_i).abs() * angle_fn(phi_d).abs()
                              ).clamp(1e-6, 1.0)

        cos_0 = _sym_grazing(torch.sin)
        cos_n = _sym_grazing(lambda a: torch.sin(n_idx * np.pi - a))
        # The wedge-face coefficients get the same roughness treatment as an
        # ordinary bounce: they represent reflection off those faces, so a
        # rough face must reduce them by the same coherence factor.  cos_0 and
        # cos_n are already the reciprocal grazing cosines of _sym_grazing, so
        # the roughness factor inherits that symmetry and reciprocity holds.
        r0 = reflection_coefficient(tx.frequency, cos_0, self.material(si_a),
                                    self.cfg.polarisation,
                                    self._params_for(si_a, mat_overrides),
                                    roughness=self.cfg.surface_roughness)
        rn = reflection_coefficient(tx.frequency, cos_n, self.material(si_b),
                                    self.cfg.polarisation,
                                    self._params_for(si_b, mat_overrides),
                                    roughness=self.cfg.surface_roughness)

        D = tf.diffraction_coefficient(phi_i, phi_d, n_idx.expand(R), k, L,
                                       sin_b0, r0, rn)
        A = tf.spreading_factor(s_p, s_o)
        L_tot = s_p + s_o
        g = tx.antenna.field_gain(u_in) * rx.antenna.field_gain(-u_out)
        e_inc = (lam / (4.0 * np.pi * s_p)) * g
        gain = (e_inc.to(CDTYPE) * D * A.to(CDTYPE)
                * torch.exp(-1j * (k * L_tot).to(CDTYPE)))

        # both legs of the diffracted ray must themselves be unobstructed
        excl = (si_a, si_b)
        gain = gain * self._segment_weight(p_tx, p_d, k, excl, vertices, mat_overrides)
        gain = gain * self._segment_weight(p_d, rx_pos, k, excl, vertices, mat_overrides)

        return Paths(gain.unsqueeze(-1), L_tot.unsqueeze(-1),
                     (L_tot / C0).unsqueeze(-1), u_in.unsqueeze(-2),
                     u_out.unsqueeze(-2), torch.ones(1, dtype=torch.long),
                     [f"diff-{wedge.v0}_{wedge.v1}"],
                     [torch.stack([p_tx, p_d, rx_pos], dim=-2)])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def trace(self, tx: Transmitter, rx: Receiver,
              rx_positions: Optional[torch.Tensor] = None,
              vertices: Optional[torch.Tensor] = None,
              mat_overrides: Optional[Dict[str, MaterialParams]] = None,
              sequences: Optional[List[Tuple[int, ...]]] = None) -> Paths:
        """Trace all paths from ``tx`` to one or many receiver positions.

        Parameters
        ----------
        rx_positions  : ``(R, 3)`` batch; defaults to ``rx.position``.
        vertices      : optional replacement vertex tensor, so that gradients
                        can flow to the geometry (Eq. 10).
        mat_overrides : learnable material parameters keyed by material name.
        sequences     : precomputed candidate facet sequences; pass the result
                        of :meth:`candidate_sequences` to reuse the search
                        across an optimisation loop.
        """
        rx_pos = rx.position.reshape(1, 3) if rx_positions is None else as_t(rx_positions)
        if rx_pos.dim() == 1:
            rx_pos = rx_pos.reshape(1, 3)

        families = [self._los(tx, rx_pos, rx, vertices, mat_overrides)]
        if sequences is None:
            sequences = []
            for order in range(1, self.cfg.max_order + 1):
                sequences += self.candidate_sequences(tx.position, rx_pos, order)
        for seq in sequences:
            families.append(self._specular(tx, rx_pos, rx, seq, vertices, mat_overrides))

        if self.cfg.enable_diffraction:
            for w in self.mesh.wedges():
                if w.n_index < self.cfg.min_wedge_index:
                    continue
                families.append(self._diffraction(tx, rx_pos, rx, w, vertices,
                                                  mat_overrides))
        return _cat_paths(families)

    @staticmethod
    def doppler(paths: Paths, wavelength: float,
                v_tx: Optional[torch.Tensor] = None,
                v_rx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Per-path Doppler shift ``df = -(1/lambda) dL/dt`` (App. E.3).

        Differentiating the path length w.r.t. the endpoint positions gives
        ``dL/dt = -u_dep . v_tx + u_arr . v_rx``, so a receiver approaching the
        transmitter produces a positive shift.
        """
        df = torch.zeros(paths.length.shape, dtype=FDTYPE)
        if v_tx is not None:
            df = df + (paths.dep_dir * as_t(v_tx).reshape(-1, 1, 3)).sum(-1) / wavelength
        if v_rx is not None:
            df = df - (paths.arr_dir * as_t(v_rx).reshape(-1, 1, 3)).sum(-1) / wavelength
        return df