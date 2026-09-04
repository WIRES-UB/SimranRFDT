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
                        lobe_normalisation, scattering_coefficient,
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
    #: path-validity model: "rfdt" (Eq. 11), "heaviside" (Eq. 3), "sigmoid"
    #: (Eq. 4), or "fresnel", which uses the weight the exact half-plane
    #: solution actually calls for rather than the UTD transition function.
    #: See transition.weight_fresnel: the two differ by a factor of two at a
    #: boundary and, more importantly here, the UTD weight is identically zero
    #: throughout a shadow and therefore has an identically zero gradient
    #: there, which is the failure mode the smooth weight exists to remove.
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
    #: highest diffraction order.  1 is single-edge diffraction only; 2 adds
    #: edge-to-edge double diffraction, which is what carries energy into deep
    #: shadow where the first-order term has almost none to give.
    #:
    #: The default is 1 on cost, not on physics.  The furnished room has 36
    #: diffracting edges and so 1260 ordered pairs, and evaluating them costs
    #: six to seventeen times a first-order trace depending on how many
    #: reflection orders it is measured against.
    #:
    #: What that buys depends entirely on the scene, and experiment 9 measures
    #: both ends of it.  Behind two knife edges in series in free space, single
    #: diffraction predicts not a small field but exactly zero, and the cascade
    #: is the whole answer.  Inside a reflective room it is worth under a dB,
    #: because a partition there never produces a deep shadow at all: wall
    #: reflections fill it in and sit 15 to 25 dB above every diffracted term.
    max_diffraction_order: int = 1
    #: include the slope-diffraction term at the second edge.  Ordinary
    #: diffraction uses only the value of the incident field at the edge; when
    #: that field varies rapidly across the edge, which is exactly the case
    #: when the second edge sits near the first one's shadow boundary, the
    #: derivative term is comparable in size and leaving it out is not a small
    #: approximation.
    enable_slope_diffraction: bool = True
    #: alternating-minimisation steps for the two-edge Fermat point.  The total
    #: path length is jointly convex in the two edge parameters, so this
    #: converges to the global minimum.
    #:
    #: Twenty-five rather than the six that look visually converged on a
    #: single pair.  Pairs whose stationary point lands on the end of a finite
    #: edge converge much more slowly, and the doubly diffracted total is a sum
    #: of more than a thousand terms that very nearly cancel, so it has almost
    #: no tolerance for per-term phase noise.  Measured reciprocity error of
    #: the total field against this setting: 7e-3 dB at 12 steps, 8e-4 dB at
    #: 25, 4e-5 dB at 50.  The Fermat solve is cheap next to the coefficients,
    #: so buying accuracy here costs little.
    double_diffraction_iterations: int = 25
    #: radiate the energy that surface roughness removes from the specular
    #: direction, instead of deleting it.  Off by default only because it is
    #: new and costs a patch sum per facet; it is not optional physics once
    #: roughness is enabled, since without it the simulator loses up to 95 per
    #: cent of a rough surface's power at millimetre wave.
    enable_diffuse: bool = False
    #: scattering patches per axis on each facet, so this squared per facet
    diffuse_patches: int = 5
    #: directivity of the scattering lobe about the specular direction.  Larger
    #: is more forward-scattering; 4 is a common choice for building surfaces.
    diffuse_alpha: float = 4.0
    #: minimum separation between the two diffraction points, in wavelengths.
    #: The ray-optical cascade needs the second edge to lie in the far field of
    #: the first, and it fails hard rather than gracefully when it does not:
    #: the spreading factor carries 1/sqrt(s12), so two edges meeting at a
    #: corner drive the amplitude to infinity.  That configuration is not a
    #: cascade at all but the separate canonical problem of corner diffraction,
    #: which this does not implement.  Pairs sharing a vertex are rejected
    #: outright and the rest are held to this separation.
    double_diffraction_min_separation_wavelengths: float = 1.0
    #: discard a doubly-diffracted candidate whose purely geometric amplitude
    #: is more than this far below the free-space direct field.  The bound uses
    #: no diffraction coefficients, so it is cheap, and the losslessness of the
    #: choice is asserted in the tests rather than assumed.
    double_diffraction_dynamic_range_db: float = 80.0


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

    def near_field_report(self, wavelength: float,
                          antenna_aperture_m: Optional[float] = None,
                          short_segment_wavelengths: float = 10.0
                          ) -> Dict[str, float]:
        """How far the ray-optical assumption has been pushed, as numbers.

        Every interaction here assumes the field arriving at it is locally a
        plane wave, which is a statement about distances in wavelengths.  The
        assumption was previously unstated, so this reports it rather than
        leaving it to be discovered.

        The obvious test turns out to be the wrong one and is deliberately not
        used.  A specular bounce's effective aperture is its first Fresnel
        zone, of diameter ``2 sqrt(lambda L1 L2 / (L1 + L2))``, whose Fraunhofer
        distance is ``8 L1 L2 / (L1 + L2)``.  That is always several times
        larger than the interaction distance itself, so a reflection is never in
        the far field of its own aperture, at any frequency or geometry.  This
        is not a defect to report: image theory is exact for an infinite plane
        at any distance, and the criterion simply does not apply to that
        interaction.  Applying it anyway would flag every path in every scene
        and mean nothing.

        What does govern validity is reported instead:

        ``min_segment_wavelengths``
            The shortest hop anywhere, in wavelengths.  Ray optics needs each
            hop to be many; a hop of order one wavelength is not a ray.
        ``fraction_power_with_short_segment``
            Share of the received *power* arriving through paths that contain a
            hop shorter than the threshold.  This is the number to quote, and
            it is not the same as the share of paths: a candidate search
            produces many geometrically degenerate paths, some with segments of
            literally zero length, which carry no energy at all.  Counting them
            says a quarter of the paths are suspect when in truth almost none
            of the power is.  ``fraction_paths_with_short_segment`` is kept
            beside it precisely so the two can be compared.
        ``max_fresnel_radius_m``
            Largest first Fresnel zone radius over the interactions.  A facet
            smaller than this does not reflect fully, and while the validity
            weight handles that smoothly, the ray treatment behind it is
            degrading.
        ``antenna_fraunhofer_m``, ``fraction_first_hop_inside_fraunhofer``
            Only when an aperture is given, since an antenna's own near field
            is a real aperture problem rather than a Fresnel-zone one.  This
            model has no aperture of its own, so the caller must supply the
            physical one.
        """
        if not self.nodes:
            return {"paths": 0}
        seg_min, fres_max, short = [], [], []
        first_hop = []
        weights = []
        for j, nodes in enumerate(self.nodes):
            weights.append(self.gain[:, j].abs() ** 2)
            d = nodes[:, 1:, :] - nodes[:, :-1, :]
            lengths = d.norm(dim=-1)                    # (R, hops)
            seg_min.append(lengths.min(dim=-1).values)
            first_hop.append(lengths[:, 0])
            short.append((lengths < short_segment_wavelengths * wavelength)
                         .any(dim=-1))
            if lengths.shape[-1] >= 2:
                a, b = lengths[:, :-1], lengths[:, 1:]
                fres_max.append(torch.sqrt(
                    (wavelength * a * b / (a + b).clamp_min(1e-12))
                    .clamp_min(0.0)).max(dim=-1).values)
        seg = torch.cat(seg_min)
        flag = torch.cat(short)
        power = torch.cat(weights)
        total = power.sum().clamp_min(1e-300)
        # the shortest hop that actually carries energy, which is the one the
        # ray assumption has to survive; the global minimum is dominated by
        # degenerate candidates of zero length and zero contribution
        carries = power > power.max() * 1e-6
        seg_live = seg[carries] if bool(carries.any()) else seg
        out = {
            "paths": int(seg.numel()),
            "min_segment_wavelengths": float(seg.min() / wavelength),
            "min_segment_wavelengths_carrying_power":
                float(seg_live.min() / wavelength),
            "median_segment_wavelengths": float(seg.median() / wavelength),
            "fraction_power_with_short_segment":
                float((power * flag.to(FDTYPE)).sum() / total),
            "fraction_paths_with_short_segment": float(flag.to(FDTYPE).mean()),
            "short_segment_threshold_wavelengths": float(short_segment_wavelengths),
        }
        if fres_max:
            out["max_fresnel_radius_m"] = float(torch.cat(fres_max).max())
        if antenna_aperture_m:
            ff = 2.0 * float(antenna_aperture_m) ** 2 / wavelength
            out["antenna_fraunhofer_m"] = ff
            out["fraction_first_hop_inside_fraunhofer"] = float(
                (torch.cat(first_hop) < ff).to(FDTYPE).mean())
        return out

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
            # named use_fresnel, not fresnel: this scope imports the Fresnel
            # coefficient function of that name from materials, and shadowing
            # it fails only later, inside the solid-object branch
            use_fresnel = self.cfg.weighting == "fresnel"
            if use_fresnel:
                # The exact weight satisfies F(s) + F(-s) = 1 identically, so
                # "goes around" and "goes through" become a true partition of
                # the incident energy rather than two separately clamped terms
                # that happen not to sum to anything in particular.  With the
                # UTD weight the pair sums to F(|x|), which falls to zero at
                # the silhouette: energy simply disappears there and the
                # diffraction term is relied on to put it back.
                s_arg = tf.edge_argument_fresnel(tau, L, k)
                w_face = tf.weight_fresnel(s_arg)
            else:
                w_face = tf.transition_F(tf.edge_argument(tau.clamp_min(0.0), L, k))

            if self.cfg.enable_transmission:
                surf = self.surfaces[si]
                mat = self.material(si)
                params = self._params_for(si, mat_overrides)
                cos_ti = (du * n).sum(-1).abs().clamp(1e-6, 1.0)
                f_block = (tf.weight_fresnel(-s_arg) if use_fresnel
                           else tf.transition_F(
                               tf.edge_argument((-tau).clamp_min(0.0), L, k)))
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
            elif self.cfg.weighting == "fresnel":
                sin_in = torch.sqrt(
                    (1.0 - (dirs[i] * e_dir).sum(-1) ** 2).clamp_min(1e-12))
                sin_out = torch.sqrt(
                    (1.0 - (dirs[i + 1] * e_dir).sum(-1) ** 2).clamp_min(1e-12))
                sin_b0 = torch.sqrt(sin_in * sin_out)
                Lp = tf.distance_parameter(lens[i], lens[i + 1], sin_b0)
                weight = weight * tf.weight_fresnel(
                    tf.edge_argument_fresnel(d_edge, Lp, k, sin_b0))
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

    # ------------------------------------------------------------------
    # edge geometry shared by first- and second-order diffraction
    # ------------------------------------------------------------------
    def _edge_geom(self, wedge: Wedge, vertices=None):
        """Everything about one diffracting edge that does not depend on a ray.

        Returns the edge origin and unit vector, its length, the local frame
        ``(b1, b2)`` in which wedge angles are measured, the wedge index and
        the two face surfaces.  Extracted so that single and double
        diffraction cannot drift apart: the frame convention in particular has
        to be identical or the two orders would disagree about which side of a
        wedge is which.
        """
        v = self.mesh.vertices if vertices is None else vertices
        v0, v1 = v[wedge.v0], v[wedge.v1]
        e_vec = v1 - v0
        e_len = e_vec.norm().clamp_min(1e-9)
        e = e_vec / e_len
        tris = self.mesh.tri(vertices)
        w1 = tris[wedge.tri_a].mean(dim=0) - v0
        b1 = normalize(w1 - (w1 * e).sum() * e)
        b2 = normalize(torch.cross(e, b1, dim=-1))
        n_a = supporting_plane(tris[wedge.tri_a])[1]
        if float((n_a * b2).sum()) < 0:
            b2 = -b2
        lut = self.mesh.surface_of_face()
        si_a = lut[wedge.tri_a]
        si_b = lut[wedge.tri_b] if wedge.tri_b is not None else si_a
        return {"v0": v0, "e": e, "e_len": e_len, "b1": b1, "b2": b2,
                "n_idx": torch.as_tensor(wedge.n_index, dtype=FDTYPE),
                "si_a": si_a, "si_b": si_b, "wedge": wedge}

    @staticmethod
    def _fermat_on_edge(a: torch.Tensor, b: torch.Tensor, geom) -> torch.Tensor:
        """Point on a straight edge minimising ``|a - p| + |p - b|``.

        Stationarity of the total path length gives the closed form

            t* = (h_b t_a + h_a t_b) / (h_a + h_b)

        with ``t`` the projections of the two endpoints onto the edge line and
        ``h`` their perpendicular distances.  Exact and differentiable, so no
        root find is needed and the gradient stays clean.  The clamp handles a
        stationary point that falls off the end of a finite edge, where the
        true minimum is the nearer endpoint.
        """
        v0, e, e_len = geom["v0"], geom["e"], geom["e_len"]

        def decompose(p):
            r = p - v0
            t = (r * e).sum(-1)
            return t, (r - t.unsqueeze(-1) * e).norm(dim=-1).clamp_min(1e-9)

        t_a, h_a = decompose(a)
        t_b, h_b = decompose(b)
        t = (h_b * t_a + h_a * t_b) / (h_a + h_b).clamp_min(1e-12)
        return t.clamp(0.0, float(e_len))

    @staticmethod
    def _wedge_angle(u: torch.Tensor, geom) -> torch.Tensor:
        """Angle of direction ``u`` about the edge, from face A into the air."""
        e, b1, b2 = geom["e"], geom["b1"], geom["b2"]
        up = normalize(u - (u * e).sum(-1, keepdim=True) * e)
        ang = torch.atan2((up * b2).sum(-1), (up * b1).sum(-1))
        return torch.where(ang < 0, ang + 2.0 * np.pi, ang)

    @staticmethod
    def _sin_beta0(u_in: torch.Tensor, u_out: torch.Tensor,
                   e: torch.Tensor) -> torch.Tensor:
        """Reciprocal cone angle from the incident and diffracted directions.

        On Keller's cone the two angles are equal and either would do.  They
        differ only where the stationary point has been clamped to the end of a
        finite edge, and there taking one alone would make the answer depend on
        which end of the path is the source.  The geometric mean is symmetric
        under path reversal and identical on the cone.
        """
        sin_in = torch.sqrt((1.0 - (u_in * e).sum(-1) ** 2).clamp_min(1e-12))
        sin_out = torch.sqrt((1.0 - (u_out * e).sum(-1) ** 2).clamp_min(1e-12))
        return torch.sqrt(sin_in * sin_out)

    def _face_reflections(self, tx: Transmitter, phi_i, phi_d, geom,
                          mat_overrides):
        """Luebbers' face coefficients for a wedge, evaluated reciprocally.

        The heuristic weights each reflection-boundary term by the Fresnel
        coefficient of the corresponding wedge face at its grazing angle.
        Using the incidence angle alone would make the coefficient depend on
        which end of the path is the source, so the geometric mean of the
        incidence and diffraction values is used; it reduces to the standard
        value when they coincide, which is the specular-on-edge case where
        these terms matter most.
        """
        n_idx = geom["n_idx"]

        def sym(angle_fn):
            # Clamp inside the square root, not after it.  A ray grazing
            # exactly along a wedge face drives the product to zero, where the
            # square root's derivative is infinite; clamping the result
            # afterwards fixes the value and leaves the backward pass emitting
            # NaN for the whole batch.  Single diffraction rarely lands exactly
            # on that angle, so this sat latent until edge-to-edge cascades
            # started hitting it routinely.
            prod = (angle_fn(phi_i).abs() * angle_fn(phi_d).abs())
            return torch.sqrt(prod.clamp_min(1e-12)).clamp(1e-6, 1.0)

        cos_0 = sym(torch.sin)
        cos_n = sym(lambda a: torch.sin(n_idx * np.pi - a))
        r0 = reflection_coefficient(
            tx.frequency, cos_0, self.material(geom["si_a"]),
            self.cfg.polarisation,
            self._params_for(geom["si_a"], mat_overrides),
            roughness=self.cfg.surface_roughness)
        rn = reflection_coefficient(
            tx.frequency, cos_n, self.material(geom["si_b"]),
            self.cfg.polarisation,
            self._params_for(geom["si_b"], mat_overrides),
            roughness=self.cfg.surface_roughness)
        return r0, rn

    def _diffraction(self, tx: Transmitter, rx_pos: torch.Tensor, rx: Receiver,
                     wedge: Wedge, vertices=None, mat_overrides=None) -> Paths:
        """First-order wedge diffraction, Eq. 6 with the coefficient of Eq. 7."""
        k, lam = tx.k, tx.wavelength
        R = rx_pos.shape[0]
        geom = self._edge_geom(wedge, vertices)
        p_tx = tx.position.reshape(1, 3).expand(R, 3)

        t_star = self._fermat_on_edge(p_tx, rx_pos, geom)
        p_d = geom["v0"] + t_star.unsqueeze(-1) * geom["e"]

        s_p = (p_d - p_tx).norm(dim=-1).clamp_min(1e-6)      # s'
        s_o = (rx_pos - p_d).norm(dim=-1).clamp_min(1e-6)    # s
        u_in = (p_d - p_tx) / s_p.unsqueeze(-1)
        u_out = (rx_pos - p_d) / s_o.unsqueeze(-1)
        sin_b0 = self._sin_beta0(u_in, u_out, geom["e"])
        L = tf.distance_parameter(s_p, s_o, sin_b0)

        phi_i = self._wedge_angle(-u_in, geom)
        phi_d = self._wedge_angle(u_out, geom)
        r0, rn = self._face_reflections(tx, phi_i, phi_d, geom, mat_overrides)

        D = tf.diffraction_coefficient(phi_i, phi_d, geom["n_idx"].expand(R),
                                       k, L, sin_b0, r0, rn)
        A = tf.spreading_factor(s_p, s_o)
        L_tot = s_p + s_o
        g = tx.antenna.field_gain(u_in) * rx.antenna.field_gain(-u_out)
        e_inc = (lam / (4.0 * np.pi * s_p)) * g
        gain = (e_inc.to(CDTYPE) * D * A.to(CDTYPE)
                * torch.exp(-1j * (k * L_tot).to(CDTYPE)))

        # both legs of the diffracted ray must themselves be unobstructed
        excl = (geom["si_a"], geom["si_b"])
        gain = gain * self._segment_weight(p_tx, p_d, k, excl, vertices, mat_overrides)
        gain = gain * self._segment_weight(p_d, rx_pos, k, excl, vertices, mat_overrides)

        return Paths(gain.unsqueeze(-1), L_tot.unsqueeze(-1),
                     (L_tot / C0).unsqueeze(-1), u_in.unsqueeze(-2),
                     u_out.unsqueeze(-2), torch.ones(1, dtype=torch.long),
                     [f"diff-{wedge.v0}_{wedge.v1}"],
                     [torch.stack([p_tx, p_d, rx_pos], dim=-2)])

    # ------------------------------------------------------------------
    # second-order (edge-to-edge) diffraction
    # ------------------------------------------------------------------
    def _double_fermat(self, p_tx: torch.Tensor, rx_pos: torch.Tensor,
                       ga, gb):
        """Stationary path Tx -> edge A -> edge B -> Rx, by alternating steps.

        The total length ``|Tx - P1| + |P1 - P2| + |P2 - Rx|`` is a sum of
        norms of affine functions of the two edge parameters, so it is jointly
        convex in them.  Alternating exact minimisation over one parameter at a
        time therefore converges to the global minimum rather than to some
        nearby stationary point, and each step is the same closed form the
        single-edge case uses.  Unrolling a fixed number of steps keeps the
        whole construction differentiable by autograd, with no implicit
        function theorem needed.
        """
        p2 = gb["v0"] + 0.5 * float(gb["e_len"]) * gb["e"]
        p2 = p2.reshape(1, 3).expand_as(rx_pos)
        p1 = None
        for _ in range(max(1, int(self.cfg.double_diffraction_iterations))):
            t1 = self._fermat_on_edge(p_tx, p2, ga)
            p1 = ga["v0"] + t1.unsqueeze(-1) * ga["e"]
            t2 = self._fermat_on_edge(p1, rx_pos, gb)
            p2 = gb["v0"] + t2.unsqueeze(-1) * gb["e"]
        return p1, p2

    def _double_diffraction(self, tx: Transmitter, rx_pos: torch.Tensor,
                            rx: Receiver, wa: Wedge, wb: Wedge,
                            vertices=None, mat_overrides=None):
        """Diffraction at one edge and then at a second, Tx -> A -> B -> Rx.

        Why this is not simply the coefficient applied twice
        ----------------------------------------------------
        Amplitude bookkeeping first.  The field leaving edge A spreads as an
        astigmatic wave whose caustic distance at edge B is ``s1 + s12``, not
        ``s12``, because the wave remembers how far it travelled before the
        first edge.  Using ``s12`` alone, which is what a naive cascade does,
        gives an amplitude that is not reciprocal: swapping transmitter and
        receiver changes the answer by the ratio of the two end distances.
        With the correct caustic distance the whole product collapses to

            sqrt(1 / (s1 * s12 * s2 * (s1 + s12 + s2)))

        which is manifestly symmetric under reversing the path, and the tests
        check that it holds to machine precision.

        Second, the validity question.  Ordinary diffraction assumes the field
        arriving at the edge is locally a plane wave, so that its value at the
        edge is the whole story.  The field arriving at edge B is a diffracted
        field, and near edge A's shadow boundary it varies rapidly across
        edge B.  There the value alone is not enough and the slope term below
        supplies the missing derivative.  Where even that is insufficient, a
        uniform double-diffraction coefficient with a two-variable transition
        function is required, which this does not implement.  Rather than
        approximate that regime, pairs closer than
        ``double_diffraction_min_separation_wavelengths`` are rejected outright
        and pairs sharing a vertex are excluded as corners.
        """
        need_slope = bool(self.cfg.enable_slope_diffraction)
        k, lam = tx.k, tx.wavelength
        R = rx_pos.shape[0]
        ga = self._edge_geom(wa, vertices)
        gb = self._edge_geom(wb, vertices)
        p_tx = tx.position.reshape(1, 3).expand(R, 3)

        p1, p2 = self._double_fermat(p_tx, rx_pos, ga, gb)

        min_sep = (self.cfg.double_diffraction_min_separation_wavelengths
                   * lam)
        s1 = (p1 - p_tx).norm(dim=-1).clamp_min(1e-6)
        s12_raw = (p2 - p1).norm(dim=-1)
        s2 = (rx_pos - p2).norm(dim=-1).clamp_min(1e-6)
        # Clamp before computing anything, not after.  The separation test
        # below discards the degenerate entries, but masking with torch.where
        # does not stop the discarded branch from being evaluated, and its
        # backward pass multiplies a zero by the infinity that 1/sqrt(s12)
        # produces at s12 = 0.  That yields NaN gradients for the whole batch
        # while leaving the forward values perfectly correct.  Clamping first
        # keeps the rejected branch finite so the mask can do its job.
        s12 = s12_raw.clamp_min(max(min_sep, 1e-6))
        L_tot = s1 + s12 + s2

        u_in = (p1 - p_tx) / s1.unsqueeze(-1)
        u_mid = (p2 - p1) / s12.unsqueeze(-1)
        u_out = (rx_pos - p2) / s2.unsqueeze(-1)

        sin_b1 = self._sin_beta0(u_in, u_mid, ga["e"])
        sin_b2 = self._sin_beta0(u_mid, u_out, gb["e"])
        # the equivalent source for the second edge sits at the caustic
        # distance of the wave leaving the first, which is s1 + s12
        rho2 = s1 + s12

        # Distance parameters, symmetrised.  UTD defines L from the distance to
        # the source on one side and to the observation point on the other, and
        # in a cascade those are different things depending on which end of the
        # path is the transmitter: travelling one way the second edge is fed by
        # an equivalent source at s1 + s12 and observed at s2, travelling the
        # other it is fed at s2 and observed at the first edge, s12 away.  The
        # amplitude works out symmetric on its own, but these do not, and the
        # result was several dB of reciprocity error that no amount of solver
        # convergence removed because it was not a numerical error at all.
        #
        # The geometric mean of the two directions is reciprocal by
        # construction and reduces to the standard value when they agree.  It
        # is the same device already used for the cone angle and for Luebbers'
        # face coefficients, and it is a symmetrisation rather than a
        # derivation: the rigorous treatment is a joint two-edge coefficient
        # with a two-variable transition function, which this does not
        # implement.
        def sym_L(fwd, rev, sin_b):
            return torch.sqrt((fwd * rev).clamp_min(1e-30)) * sin_b ** 2

        L1 = sym_L(s1 * s12 / (s1 + s12).clamp_min(1e-12),
                   (s12 + s2) * s1 / (s1 + s12 + s2).clamp_min(1e-12), sin_b1)
        L2 = sym_L(rho2 * s2 / (rho2 + s2).clamp_min(1e-12),
                   s12 * s2 / (s12 + s2).clamp_min(1e-12), sin_b2)

        phi_i1 = self._wedge_angle(-u_in, ga)
        phi_d1 = self._wedge_angle(u_mid, ga)
        phi_i2 = self._wedge_angle(-u_mid, gb)
        phi_d2 = self._wedge_angle(u_out, gb)

        r0a, rna = self._face_reflections(tx, phi_i1, phi_d1, ga, mat_overrides)
        r0b, rnb = self._face_reflections(tx, phi_i2, phi_d2, gb, mat_overrides)

        # The coefficients are built through closures so the slope term can
        # differentiate them with respect to one angle.  The face reflection
        # weights are held fixed inside the closure: the rapid variation that
        # slope diffraction exists to capture lives in the transition terms,
        # while the Fresnel weights vary slowly across the edge, so freezing
        # them is a controlled approximation rather than an oversight.
        def make_D1(angle):
            return tf.diffraction_coefficient(
                phi_i1, angle, ga["n_idx"].expand(R), k, L1, sin_b1, r0a, rna)

        def make_D2(angle):
            return tf.diffraction_coefficient(
                angle, phi_d2, gb["n_idx"].expand(R), k, L2, sin_b2, r0b, rnb)

        D1 = make_D1(phi_d1)
        D2 = make_D2(phi_i2)


        A1 = tf.spreading_factor(s1, s12)
        A2 = tf.spreading_factor(rho2, s2)
        g = tx.antenna.field_gain(u_in) * rx.antenna.field_gain(-u_out)
        e_inc = (lam / (4.0 * np.pi * s1)) * g
        phase = torch.exp(-1j * (k * L_tot).to(CDTYPE))
        common = e_inc.to(CDTYPE) * A1.to(CDTYPE) * A2.to(CDTYPE) * phase

        gain = common * D1 * D2

        if need_slope:
            # Slope diffraction.  The incident field at edge B varies across
            # the edge only through edge A's diffraction angle, which changes
            # at the rate 1/s12 for a transverse step, so
            #     dE_i/dn' = e_inc * A1 * (dD1/dphi_d1) / s12,
            # and the second edge responds to that gradient through dD2/dphi_i2
            # with the standard 1/(jk) prefactor.  Both derivatives come from
            # autograd rather than a hand-differentiated coefficient, which is
            # the one place in this simulator where the differentiability built
            # for inverse rendering pays off in the forward direction as well.
            # The pair is reciprocal because the wedge coefficient is symmetric
            # in its two angles, so reversing the path swaps which derivative
            # is which and leaves the product alone.
            keep = torch.is_grad_enabled()
            d1_slope = self._angle_derivative(make_D1, phi_d1, keep)
            d2_slope = self._angle_derivative(make_D2, phi_i2, keep)
            slope = (common * d1_slope * d2_slope
                     / (1j * k * s12).to(CDTYPE))
            gain = gain + slope

        # A pair can be a valid cascade for some receivers on a route and
        # degenerate for others, so the separation condition is enforced per
        # receiver rather than only when selecting candidates.  Without this a
        # single receiver position at which the two stationary points collapse
        # would inject the 1/sqrt(s12) singularity into an otherwise sound path.
        gain = torch.where(s12_raw > min_sep, gain, torch.zeros_like(gain))

        excl = (ga["si_a"], ga["si_b"], gb["si_a"], gb["si_b"])
        gain = gain * self._segment_weight(p_tx, p1, k, excl, vertices, mat_overrides)
        gain = gain * self._segment_weight(p1, p2, k, excl, vertices, mat_overrides)
        gain = gain * self._segment_weight(p2, rx_pos, k, excl, vertices, mat_overrides)

        return Paths(gain.unsqueeze(-1), L_tot.unsqueeze(-1),
                     (L_tot / C0).unsqueeze(-1), u_in.unsqueeze(-2),
                     u_out.unsqueeze(-2),
                     torch.full((1,), 2, dtype=torch.long),
                     [f"diff2-{wa.v0}_{wa.v1}-{wb.v0}_{wb.v1}"],
                     [torch.stack([p_tx, p1, p2, rx_pos], dim=-2)])

    @staticmethod
    def _angle_derivative(make_D, angle: torch.Tensor,
                          keep_graph: bool) -> torch.Tensor:
        """dD/dangle for a complex coefficient, in either grad mode.

        The derivative is taken with respect to a zero perturbation added to
        the angle rather than with respect to the angle itself.  That is what
        makes this work under ``torch.no_grad`` and in a plain forward
        evaluation, where the angle has no graph behind it at all and asking
        autograd for a derivative with respect to it simply fails.  The
        perturbation is a leaf that always requires grad, so a graph exists in
        every case, and its derivative at zero equals the one wanted.

        The alternative of detaching the angle into a leaf would work here and
        silently sever the main term's dependence on the geometry, leaving the
        forward values right and every gradient to the scene zero.

        ``keep_graph`` propagates the caller's grad mode: with it the result is
        itself differentiable, so a slope-diffracted path still carries
        gradients back to scene parameters.
        """
        with torch.enable_grad():
            eps = torch.zeros_like(angle, requires_grad=True)
            D = make_D(angle + eps)
            gr = torch.autograd.grad(D.real.sum(), eps, create_graph=keep_graph,
                                     retain_graph=True)[0]
            gi = torch.autograd.grad(D.imag.sum(), eps, create_graph=keep_graph,
                                     retain_graph=True)[0]
        return torch.complex(gr, gi)

    @staticmethod
    def _point_segment_distance(p: torch.Tensor, geom) -> torch.Tensor:
        """Exact shortest distance from points to a finite edge."""
        v0, e, e_len = geom["v0"], geom["e"], float(geom["e_len"])
        t = ((p - v0) * e).sum(-1).clamp(0.0, e_len)
        return (p - (v0 + t.unsqueeze(-1) * e)).norm(dim=-1)

    @staticmethod
    def _segment_segment_distance(ga, gb) -> float:
        """Exact shortest distance between two finite edges.

        Standard closed form with the degenerate parallel case handled, sampled
        nowhere and iterated nowhere so it costs almost nothing.  It is used
        only as a lower bound on the middle leg, which is what makes the
        candidate bound rigorous rather than a guess.
        """
        p0, u, lu = ga["v0"], ga["e"], float(ga["e_len"])
        q0, v, lv = gb["v0"], gb["e"], float(gb["e_len"])
        w0 = p0 - q0
        a = 1.0
        b = float((u * v).sum())
        c = 1.0
        d = float((u * w0).sum())
        e_ = float((v * w0).sum())
        den = a * c - b * b
        if abs(den) < 1e-12:                     # parallel edges
            sc, tc = 0.0, (e_ / c if abs(c) > 1e-12 else 0.0)
        else:
            sc = (b * e_ - c * d) / den
            tc = (a * e_ - b * d) / den
        sc = min(max(sc, 0.0), lu)
        tc = min(max(tc, 0.0), lv)
        return float(((p0 + sc * u) - (q0 + tc * v)).norm())

    def _double_candidates(self, tx: Transmitter, rx_pos: torch.Tensor,
                           wedges: Sequence[Wedge], vertices=None):
        """Ordered edge pairs worth evaluating.

        Three filters, and it is worth being exact about what each one earns,
        because only the first currently earns anything.  Measured on the
        furnished room at 5 GHz, from 1260 ordered pairs:

          * sharing a vertex, rejected: 144.  Not an optimisation but a
            correctness requirement, since those are corners rather than
            cascades and their amplitude is singular.
          * closer than the far-field separation, rejected: 0.  The room's
            edges are metres apart and a wavelength is 6 cm, so this only ever
            fires on finely tessellated geometry.
          * below the amplitude bound, rejected: 0.

        The amplitude bound is rigorous, being built from the shortest possible
        length of each leg, and it is useless, which is worth understanding
        rather than tuning.  Almost all of a cascade's attenuation lives in the
        two diffraction coefficients, each typically tens of dB down, and
        almost none of it in the spreading factor the bound is made of.  A
        bound that ignores the coefficients therefore sits far above every real
        contribution.  Making it bite would need a bound on the coefficients
        themselves, which the damping makes possible in principle and which is
        not attempted here; the honest present position is that the cost of
        second-order diffraction is the cost of evaluating all 1116 surviving
        pairs, and that is why it is off by default.
        """
        p_tx = tx.position.reshape(1, 3).expand(rx_pos.shape[0], 3)
        direct = (rx_pos - p_tx).norm(dim=-1).clamp_min(1e-6)
        floor = 10.0 ** (-self.cfg.double_diffraction_dynamic_range_db / 20.0)
        min_sep = (self.cfg.double_diffraction_min_separation_wavelengths
                   * tx.wavelength)
        geoms = [self._edge_geom(w, vertices) for w in wedges]

        def amplitude_bound(s1, s12, s2):
            return torch.sqrt(1.0 / (s1 * s12 * s2 *
                                     (s1 + s12 + s2)).clamp_min(1e-30))

        with torch.no_grad():
            d_tx = [self._point_segment_distance(p_tx, g).clamp_min(1e-6)
                    for g in geoms]
            d_rx = [self._point_segment_distance(rx_pos, g).clamp_min(1e-6)
                    for g in geoms]
            stage1 = []
            for i, wi in enumerate(wedges):
                for j, wj in enumerate(wedges):
                    if i == j:
                        continue
                    if {wi.v0, wi.v1} & {wj.v0, wj.v1}:
                        continue                 # corner, not a cascade
                    sep = self._segment_segment_distance(geoms[i], geoms[j])
                    if sep < min_sep:
                        continue
                    if not bool((amplitude_bound(d_tx[i], sep, d_rx[j])
                                 * direct > floor).any()):
                        continue
                    stage1.append((i, j))

            keep = []
            for i, j in stage1:
                p1, p2 = self._double_fermat(p_tx, rx_pos, geoms[i], geoms[j])
                s1 = (p1 - p_tx).norm(dim=-1).clamp_min(1e-6)
                s12 = (p2 - p1).norm(dim=-1).clamp_min(1e-6)
                s2 = (rx_pos - p2).norm(dim=-1).clamp_min(1e-6)
                ok = (s12 > min_sep) & (amplitude_bound(s1, s12, s2)
                                        * direct > floor)
                if bool(ok.any()):
                    keep.append((wedges[i], wedges[j]))
        return keep

    # ------------------------------------------------------------------
    # diffuse scattering
    # ------------------------------------------------------------------
    def _facet_patches(self, si: int, n_per_axis: int, vertices=None):
        """Patch centres and per-patch area covering facet ``si``.

        A regular grid over the facet's in-plane bounding box, keeping the
        cells whose centre lies inside the outline.  Crude on purpose: the
        diffuse field is a sum of many small contributions whose phases spread
        over many wavelengths, so it is insensitive to exactly where the
        samples sit, and a quadrature clever enough to matter would be a
        quadrature slow enough to notice.

        Uses the same signed edge distance as the validity weight, so it is
        exact for the convex outlines the scene builders produce and
        conservative for a reflex corner, in the same way and for the same
        reason.
        """
        p0, n = self._plane(si, vertices)
        bnd = self._boundary(si, vertices).reshape(-1, 3)
        ref = bnd[1] - bnd[0]
        u = normalize(ref - (ref * n).sum() * n)
        w = torch.cross(n, u, dim=-1)
        rel = bnd - p0
        cu, cw = (rel * u).sum(-1), (rel * w).sum(-1)
        m = int(max(1, n_per_axis))
        step = (torch.arange(m, dtype=FDTYPE) + 0.5) / m
        gu = cu.min() + step * (cu.max() - cu.min())
        gw = cw.min() + step * (cw.max() - cw.min())
        area = ((cu.max() - cu.min()) * (cw.max() - cw.min())) / (m * m)
        uu, ww = torch.meshgrid(gu, gw, indexing="ij")
        pts = (p0 + uu.reshape(-1, 1) * u + ww.reshape(-1, 1) * w)
        inside = self._edge_distance(pts, si, vertices)[0] > 0
        return pts[inside], area

    def _diffuse(self, tx: Transmitter, rx_pos: torch.Tensor, rx: Receiver,
                 si: int, vertices=None, mat_overrides=None):
        """Energy scattered off facet ``si`` away from the specular direction.

        Surface roughness removes power from the specular direction, and that
        power is scattered, not absorbed.  Modelling the removal without the
        return is not a small approximation: at 77 GHz it deletes 95 per cent of
        a concrete wall's reflected power.  This puts it back.

        Each patch re-radiates with a lobe about the specular direction, and the
        amplitude follows from conservation rather than from a fitted constant.
        The power intercepted by a patch is its area times the incident power
        density times the obliquity, a fraction ``S**2`` of that is scattered,
        and spreading it over the lobe gives

            |E_s| = S |E_i| sqrt(dA cos / F_alpha) sqrt(lobe) / r2,

        where ``F_alpha`` is the lobe's solid-angle integral.  Normalising by
        that integral, rather than by its normal-incidence value, is what keeps
        the model conservative when the specular direction is tilted and part of
        the lobe falls below the horizon.

        Two choices keep this reciprocal.  The obliquity is the geometric mean
        of the incident and scattered cosines rather than the incident one
        alone, and the scattering coefficient is evaluated at that same
        symmetric cosine.  The lobe needs no such treatment: it is symmetric
        already, because reversing a path about a specular direction leaves the
        angle from it unchanged.

        Phases are the true path lengths, so the patches interfere rather than
        being added as powers.  That is what gives the diffuse field its delay
        spread, which is the whole reason it matters for a channel model.
        """
        k, lam, f_hz = tx.k, tx.wavelength, tx.frequency
        R = rx_pos.shape[0]
        pts, area = self._facet_patches(si, self.cfg.diffuse_patches, vertices)
        if pts.numel() == 0:
            return None
        _, n = self._plane(si, vertices)
        p_tx = tx.position.reshape(1, 3)

        q = pts.reshape(1, -1, 3)                       # (1, P, 3)
        d_in = q - p_tx.reshape(1, 1, 3)
        r1 = d_in.norm(dim=-1).clamp_min(1e-6)
        u_in = d_in / r1.unsqueeze(-1)
        d_out = rx_pos.reshape(R, 1, 3) - q
        r2 = d_out.norm(dim=-1).clamp_min(1e-6)
        u_out = d_out / r2.unsqueeze(-1)

        cos_i = -(u_in * n).sum(-1)                     # positive if illuminated
        cos_s = (u_out * n).sum(-1)                     # positive if visible
        lit = (cos_i > 1e-6) & (cos_s > 1e-6)
        if not bool(lit.any()):
            return None
        cos_sym = torch.sqrt((cos_i.clamp_min(1e-6) * cos_s.clamp_min(1e-6)))

        spec = u_in + 2.0 * cos_i.unsqueeze(-1) * n     # mirror of the incoming ray
        cos_psi = (u_out * spec).sum(-1).clamp(-1.0, 1.0)
        lobe = ((1.0 + cos_psi) / 2.0).clamp_min(0.0) ** self.cfg.diffuse_alpha
        params = self._params_for(si, mat_overrides)
        sigma = (params.roughness if params is not None
                 and params.roughness is not None
                 else self.material(si).roughness_sigma)

        # Energy conservation and reciprocity pull in opposite directions here,
        # and they do it twice.  Conservation fixes the normalisation from the
        # incident side alone: the patch intercepts a power set by cos_i, keeps
        # a fraction S(cos_i)**2 of it, and spreads that over a lobe whose
        # integral is F(cos_i).  Every one of those is a property of the
        # incident direction, so the resulting amplitude is not symmetric under
        # swapping transmitter and receiver, and using it costs about 0.2 dB of
        # reciprocity.  Evaluating them at the symmetric cosine instead is
        # reciprocal and loses about 8 per cent of the scattered energy.
        #
        # Taking the geometric mean of the two normalisations is exactly
        # reciprocal by construction, since swapping the ends swaps the two
        # factors, and it conserves energy exactly whenever the two directions
        # agree.  This is the same device already used for the cone angle, for
        # Luebbers' face coefficients and for the cascade distance parameter,
        # and like those it is a symmetrisation rather than a derivation.
        def side(cos_a):
            c = cos_a.clamp(1e-6, 1.0)
            S = scattering_coefficient(f_hz, c, sigma, self.cfg.surface_roughness)
            F = lobe_normalisation(self.cfg.diffuse_alpha, c).clamp_min(1e-12)
            return S * torch.sqrt((area * c / F).clamp_min(0.0))

        weight = torch.sqrt((side(cos_i) * side(cos_s)).clamp_min(0.0))

        e_inc = (lam / (4.0 * np.pi * r1)) * tx.antenna.field_gain(
            u_in.reshape(-1, 3)).reshape(r1.shape)
        amp = (e_inc * weight * torch.sqrt(lobe) / r2
               * rx.antenna.field_gain(-u_out.reshape(-1, 3)).reshape(r2.shape))
        amp = torch.where(lit, amp, torch.zeros_like(amp))
        L_tot = r1 + r2
        gain = amp.to(CDTYPE) * torch.exp(-1j * (k * L_tot).to(CDTYPE))

        # both legs must reach the patch: a scattering point the transmitter
        # cannot see, or the receiver cannot see, contributes nothing
        P = pts.shape[0]
        qe = pts.reshape(1, -1, 3).expand(R, P, 3).reshape(-1, 3)
        txe = p_tx.expand(R * P, 3)
        rxe = rx_pos.reshape(R, 1, 3).expand(R, P, 3).reshape(-1, 3)
        w1 = self._segment_weight(txe, qe, k, (si,), vertices, mat_overrides)
        w2 = self._segment_weight(qe, rxe, k, (si,), vertices, mat_overrides)
        gain = gain * (w1 * w2).reshape(R, P)

        u_in_b = u_in.expand(R, P, 3)
        nodes = torch.stack([p_tx.expand(R, 3).reshape(R, 1, 3).expand(R, P, 3),
                             pts.reshape(1, P, 3).expand(R, P, 3),
                             rx_pos.reshape(R, 1, 3).expand(R, P, 3)], dim=-2)
        return Paths(gain, L_tot.expand(R, P), L_tot.expand(R, P) / C0,
                     u_in_b, u_out,
                     torch.ones(P, dtype=torch.long),
                     [f"diffuse-{si}-{i}" for i in range(P)],
                     [nodes[:, i] for i in range(P)])

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
            diffracting = [w for w in self.mesh.wedges()
                           if w.n_index >= self.cfg.min_wedge_index]
            for w in diffracting:
                families.append(self._diffraction(tx, rx_pos, rx, w, vertices,
                                                  mat_overrides))
            if self.cfg.max_diffraction_order >= 2:
                # Edge to edge.  In deep shadow the first-order term is nearly
                # zero, so this is not a correction to it but potentially the
                # dominant contribution, which is why it is enumerated rather
                # than treated as a refinement.
                for wa, wb in self._double_candidates(tx, rx_pos, diffracting,
                                                      vertices):
                    families.append(self._double_diffraction(
                        tx, rx_pos, rx, wa, wb, vertices, mat_overrides))

        if self.cfg.enable_diffuse:
            for si in range(len(self.surfaces)):
                fam = self._diffuse(tx, rx_pos, rx, si, vertices, mat_overrides)
                if fam is not None:
                    families.append(fam)
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