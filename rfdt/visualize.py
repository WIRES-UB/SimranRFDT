"""Drawing helpers: scenes, ray paths, impulse responses.

Nothing here affects the physics.  These functions turn the objects the tracer
already produces (a :class:`~rfdt.geometry.Mesh`, a
:class:`~rfdt.tracer.Paths`) into figures, so that a propagation result can be
looked at rather than only tabulated.

Ray paths are drawn from ``Paths.nodes``, which holds the actual traced
geometry: transmitter, every interaction point in order, then receiver.  The
polylines drawn are therefore the paths that were summed, not a sketch of them.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .geometry import Mesh

#: Colour per path family, used consistently across every figure.
PATH_COLOURS = {
    "los": "#e8a33d",       # direct
    "refl1": "#2e86c1",     # single bounce
    "refl2": "#7d3c98",     # double bounce
    "diff": "#27ae60",      # diffracted
}

#: Colour per material class, for scene drawings.
MATERIAL_COLOURS = {
    "concrete": "#b0b3b8", "wood": "#c08552", "ceiling_board": "#dfe3e6",
    "metal": "#7f8c8d", "plasterboard": "#e6d5b8", "glass": "#a8d8ea",
    "brick": "#b5651d", "marble": "#e8e8e8", "foam_board": "#c8e6c9",
}


def path_family(kind: str, order: int) -> str:
    """Map a path's label and order onto one of the four drawing families."""
    if kind == "los":
        return "los"
    if kind.startswith("diff"):
        return "diff"
    return "refl2" if order >= 2 else "refl1"


def surface_polygons(mesh: Mesh) -> List[Dict]:
    """Ordered corner loops of every facet, for polygon drawing.

    Returns one dict per surface with its corner coordinates, material and
    group name.  Corners are sorted by angle about the facet centroid, which
    is exact for the convex facets the scene builders produce.
    """
    out = []
    for s in mesh.surfaces():
        idx = sorted({i for e in s.boundary for i in e})
        pts = mesh.vertices[torch.as_tensor(list(idx))].detach().numpy()
        centre = pts.mean(axis=0)
        # build an in-plane basis and sort the corners around the centroid
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        n = np.linalg.norm(normal)
        if n < 1e-12:
            continue
        normal = normal / n
        u = pts[0] - centre
        u = u / max(np.linalg.norm(u), 1e-12)
        v = np.cross(normal, u)
        ang = np.arctan2((pts - centre) @ v, (pts - centre) @ u)
        out.append({"pts": pts[np.argsort(ang)], "material": s.material,
                    "group": s.group, "normal": normal, "centre": centre})
    return out


def draw_room_3d(ax, mesh: Mesh, hide_near_walls: bool = True,
                 label_materials: bool = True) -> None:
    """Draw a mesh as translucent 3-D polygons.

    ``hide_near_walls`` omits the two walls closest to the default viewpoint so
    the interior stays visible, which is the usual convention for drawing a
    room from outside.
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    skip = {"wall_xmax", "wall_ymin", "ceiling"} if hide_near_walls else set()
    for s in surface_polygons(mesh):
        if s["group"] in skip:
            continue
        solid = not s["group"].startswith(("wall", "floor", "ceiling"))
        colour = MATERIAL_COLOURS.get(s["material"], "#cccccc")
        poly = Poly3DCollection([s["pts"]], alpha=0.85 if solid else 0.22,
                                facecolor=colour,
                                edgecolor="#555555" if solid else "#999999",
                                linewidths=0.7 if solid else 0.4)
        ax.add_collection3d(poly)
        if label_materials and solid and s["group"].endswith("zmax"):
            c = s["centre"]
            ax.text(c[0], c[1], c[2] + 0.08, s["group"].split("_")[0],
                    fontsize=7, ha="center", color="#333333")


def draw_floorplan(ax, mesh: Mesh, annotate: bool = True) -> None:
    """Draw a top-down plan: room outline plus the footprint of each solid."""
    verts = mesh.vertices.detach().numpy()
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ax.add_patch(__import__("matplotlib.patches", fromlist=["Rectangle"]).Rectangle(
        (lo[0], lo[1]), hi[0] - lo[0], hi[1] - lo[1],
        fill=False, edgecolor="#333333", lw=1.8))

    from matplotlib.patches import Rectangle
    seen = set()
    for s in mesh.surfaces():
        base = s.group.split("_")[0]
        if base in ("wall", "floor", "ceiling") or base in seen:
            continue
        seen.add(base)
        members = [p for p in surface_polygons(mesh)
                   if p["group"].split("_")[0] == base]
        pts = np.concatenate([m["pts"] for m in members], axis=0)
        x0, y0 = pts[:, 0].min(), pts[:, 1].min()
        w, h = np.ptp(pts[:, 0]), np.ptp(pts[:, 1])
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=MATERIAL_COLOURS.get(
            s.material, "#cccccc"), edgecolor="#444444", lw=1.0, alpha=0.9))
        if annotate:
            ax.text(x0 + w / 2, y0 + h / 2, f"{base}\n({s.material})",
                    fontsize=7, ha="center", va="center", color="#222222")
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[1] - 0.3, hi[1] + 0.3)
    ax.set_aspect("equal")


def draw_paths(ax, paths, rx_index: int = 0, indices: Optional[Sequence[int]] = None,
               plan: bool = False, lw_scale: float = 1.0,
               label: bool = True, alpha: float = 0.9,
               max_paths: Optional[int] = None,
               view: str = "plan") -> Dict[str, int]:
    """Draw traced ray paths, with line width scaled by path strength.

    ``view`` selects the projection: ``"plan"`` for x-y, ``"elevation"`` for
    x-z, or ``"3d"``.  ``plan=True`` is kept as a shorthand for the x-y view.

    ``max_paths`` keeps only that many strongest paths.  A busy indoor scene
    produces tens of paths and drawing all of them is unreadable, so a figure
    that limits them should say so in its caption.  Returns a count of how many
    paths of each family were drawn, for building a legend.
    """
    if plan and view == "plan":
        view = "plan"
    mags_all = paths.gain[rx_index].abs().detach().numpy()
    if indices is None:
        indices = list(range(paths.n_paths()))
    indices = list(indices)
    if max_paths is not None and len(indices) > max_paths:
        indices = sorted(indices, key=lambda i: -mags_all[i])[:max_paths]
    mags = mags_all
    ref = mags.max() if mags.max() > 0 else 1.0
    counts: Dict[str, int] = {}
    drawn: Dict[str, bool] = {}
    for i in indices:
        node = paths.nodes[i][rx_index].detach().numpy()
        fam = path_family(paths.kind[i], int(paths.order[i]))
        rel = mags[i] / ref
        if rel < 1e-4:
            continue
        # line width spans 0.4 to 3.0 over a 40 dB range of path strength
        w = (0.4 + 2.6 * max(0.0, 1.0 + np.log10(max(rel, 1e-4)) / 4.0)) * lw_scale
        lab = None
        if label and fam not in drawn:
            lab = {"los": "direct", "refl1": "1 bounce",
                   "refl2": "2 bounces", "diff": "diffracted"}[fam]
            drawn[fam] = True
        if view == "3d":
            ax.plot(node[:, 0], node[:, 1], node[:, 2], color=PATH_COLOURS[fam],
                    lw=w, alpha=alpha, label=lab, zorder=3)
        else:
            c0, c1 = (0, 1) if view == "plan" else (0, 2)
            ax.plot(node[:, c0], node[:, c1], color=PATH_COLOURS[fam], lw=w,
                    alpha=alpha, label=lab, solid_capstyle="round", zorder=3)
            ax.plot(node[1:-1, c0], node[1:-1, c1], "o", color=PATH_COLOURS[fam],
                    ms=2.6 * lw_scale, zorder=4)
        counts[fam] = counts.get(fam, 0) + 1
    return counts


def draw_elevation(ax, mesh: Mesh) -> None:
    """Draw a side view (x-z projection): room outline and solid footprints.

    A side view shows what a top-down plan cannot, namely how high the
    transmitter is and whether a path clears an obstacle vertically.
    """
    from matplotlib.patches import Rectangle
    verts = mesh.vertices.detach().numpy()
    lo, hi = verts.min(axis=0), verts.max(axis=0)
    ax.add_patch(Rectangle((lo[0], lo[2]), hi[0] - lo[0], hi[2] - lo[2],
                           fill=False, edgecolor="#333333", lw=1.8))
    seen = set()
    for s_ in mesh.surfaces():
        base = s_.group.split("_")[0]
        if base in ("wall", "floor", "ceiling") or base in seen:
            continue
        seen.add(base)
        pts = np.concatenate([m["pts"] for m in surface_polygons(mesh)
                              if m["group"].split("_")[0] == base], axis=0)
        ax.add_patch(Rectangle((pts[:, 0].min(), pts[:, 2].min()),
                               np.ptp(pts[:, 0]), np.ptp(pts[:, 2]),
                               facecolor=MATERIAL_COLOURS.get(s_.material, "#ccc"),
                               edgecolor="#444444", lw=1.0, alpha=0.9))
    ax.set_xlim(lo[0] - 0.3, hi[0] + 0.3)
    ax.set_ylim(lo[2] - 0.2, hi[2] + 0.2)
    ax.set_aspect("equal")


def draw_endpoints(ax, tx_pos, rx_pos, plan: bool = False, size: float = 1.0,
                   view: str = "plan") -> None:
    """Mark the transmitter and receiver in the chosen projection."""
    tx = np.asarray(tx_pos, dtype=float).reshape(3)
    rx = np.asarray(rx_pos, dtype=float).reshape(3)
    if plan and view == "plan":
        view = "plan"
    if view in ("plan", "elevation"):
        c0, c1 = (0, 1) if view == "plan" else (0, 2)
        ax.plot(tx[c0], tx[c1], "*", ms=17 * size, color="#c0392b",
                mec="white", mew=1.2, label="Tx", zorder=6)
        ax.plot(rx[c0], rx[c1], "o", ms=10 * size, color="#1a5276",
                mec="white", mew=1.2, label="Rx", zorder=6)
    else:
        ax.plot([tx[0]], [tx[1]], [tx[2]], "*", ms=16 * size, color="#c0392b",
                mec="white", mew=1.0, label="Tx", zorder=6)
        ax.plot([rx[0]], [rx[1]], [rx[2]], "o", ms=9 * size, color="#1a5276",
                mec="white", mew=1.0, label="Rx", zorder=6)


def draw_cir(ax, paths, rx_index: int = 0, indices: Optional[Sequence[int]] = None,
             db_floor: float = -60.0, colour_by_family: bool = True) -> None:
    """Stem plot of the channel impulse response.

    One stem per path at its delay, with height the path gain in dB relative to
    the strongest path.  This is the raw output of the ray tracer before any
    signal-domain transform.
    """
    if indices is None:
        indices = range(paths.n_paths())
    delays = paths.delay[rx_index].detach().numpy() * 1e9
    mags = paths.gain[rx_index].abs().detach().numpy()
    ref = mags.max() if mags.max() > 0 else 1.0
    for i in indices:
        if mags[i] <= 0:
            continue
        db = 20 * np.log10(mags[i] / ref)
        if db < db_floor:
            continue
        fam = path_family(paths.kind[i], int(paths.order[i]))
        col = PATH_COLOURS[fam] if colour_by_family else "#2e86c1"
        ax.plot([delays[i], delays[i]], [db_floor, db], color=col, lw=1.6,
                solid_capstyle="butt")
        ax.plot([delays[i]], [db], "o", color=col, ms=4.5)
    ax.set_ylim(db_floor, 4)
    ax.set_xlabel("delay [ns]")
    ax.set_ylabel("path gain relative to strongest [dB]")


def transfer_function(paths, freqs: torch.Tensor, rx_index: int = 0,
                      indices: Optional[Sequence[int]] = None) -> torch.Tensor:
    """``H(f)`` from a chosen subset of paths.

    Uses the per-path narrowband model ``H(f) = sum_i alpha_i exp(-2 pi j f
    tau_i)`` with amplitudes taken at the frequency the paths were traced at.
    That is exact in the delay phase, which is what shapes the response, and
    approximates only the slow amplitude drift across the band.
    """
    from .signal import ofdm_channel
    if indices is None:
        indices = list(range(paths.n_paths()))
    idx = list(indices)
    d = paths.delay[rx_index:rx_index + 1, idx]
    g = paths.gain[rx_index:rx_index + 1, idx]
    return ofdm_channel(d, g, freqs).squeeze(0)


def style_3d(ax, mesh: Mesh, elev: float = 24.0, azim: float = -58.0) -> None:
    """Apply consistent limits, aspect and viewpoint to a 3-D room axis."""
    v = mesh.vertices.detach().numpy()
    lo, hi = v.min(axis=0), v.max(axis=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]))
    except AttributeError:                      # older matplotlib
        pass
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel("x [m]", fontsize=8)
    ax.set_ylabel("y [m]", fontsize=8)
    ax.set_zlabel("z [m]", fontsize=8)
    ax.tick_params(labelsize=7)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_alpha(0.04)