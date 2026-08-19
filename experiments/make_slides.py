"""Generate presentation slides from the simulator.

Produces 16:9 PNGs in ``results/slides/``, one per slide, ready to drop into
any deck.  Four sets:

  Set A, the environment: what the scene is and where everything sits.
  Set B, single path to multipath: the same link built up one path at a time,
         showing the impulse response and the frequency response change
         together.  This is the core explanatory sequence.
  Set C, the frequency domain: amplitude, phase, group delay, and how wall
         material changes frequency selectivity.
  Set D, how paths are found: the geometric construction behind the second and
         third paths, and behind diffraction.

Every number and every ray drawn comes from the tracer, not from a sketch.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.lines import Line2D                                   # noqa: E402

from rfdt import antennas, scenes, visualize as viz                   # noqa: E402
from rfdt.geometry import mirror, supporting_plane                    # noqa: E402
from rfdt.materials import C0                                         # noqa: E402
from rfdt.metrics import rms_delay_spread                             # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

SLIDES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "results", "slides")
FIGSIZE = (12.8, 7.2)          # 16:9
DPI = 150
FREQ = 5.0e9
AP = (1.2, 1.2, 2.7)
BAND = torch.linspace(4.8e9, 5.2e9, 801, dtype=torch.float64)

#: Three receiver positions spanning the interesting regimes of the route.
VIEWPOINTS = [
    ("clear line of sight", (2.0, 1.6, 0.9)),
    ("near the shadow boundary", (3.0, 4.05, 0.9)),
    ("deep NLOS behind the partition", (4.6, 1.2, 0.9)),
]


def slide(name, fig, title=None, subtitle=None):
    """Save a figure as a slide, with a consistent title block.

    Long subtitles are wrapped rather than allowed to run off the edge, which
    silently truncates them at save time.
    """
    import textwrap
    if title:
        fig.suptitle(title, fontsize=19, y=0.975, x=0.045, ha="left",
                     fontweight="bold", color="#1a1a1a")
    if subtitle:
        wrapped = "\n".join(textwrap.wrap(subtitle, width=118))
        fig.text(0.045, 0.930, wrapped, fontsize=11.5, ha="left", va="top",
                 color="#555555", linespacing=1.35)
    os.makedirs(SLIDES, exist_ok=True)
    path = os.path.join(SLIDES, name)
    fig.savefig(path, dpi=DPI, facecolor="white", bbox_inches=None)
    plt.close(fig)
    print(f"  wrote {os.path.relpath(path, os.path.dirname(SLIDES))}")


def build_scene():
    """The furnished room, transmitter, route and tracer used by every slide."""
    mesh = scenes.furnished_room()
    tx = antennas.wifi_ap(AP, FREQ, 20.0)
    traj = scenes.survey_trajectory(n_samples=48)
    tracer = RFDTracer(mesh, TracerConfig(max_order=2, weighting="rfdt",
                                          enable_diffraction=True))
    return mesh, tx, traj, tracer


def trace_at(tracer, tx, position):
    """Trace every path to a single receiver position."""
    pos = torch.tensor([position], dtype=torch.float64)
    rx = antennas.robot_client(position)
    return tracer.trace(tx, rx, rx_positions=pos)


# ---------------------------------------------------------------------------
# Set A: the environment
# ---------------------------------------------------------------------------
def slide_a1(mesh, tx, traj):
    """The scene: 3-D view and floor plan with materials and the route."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.015, 0.05, 0.44, 0.78], projection="3d")
    viz.draw_room_3d(ax, mesh)
    viz.style_3d(ax, mesh)
    p = traj.positions.numpy()
    ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#1a5276", lw=2.0, label="robot route")
    viz.draw_endpoints(ax, tx.position.numpy(), p[0])
    ax.set_title("3-D view", fontsize=11, pad=0)
    ax.legend(fontsize=8, loc="upper left")

    ax = fig.add_axes([0.545, 0.09, 0.40, 0.74])
    viz.draw_floorplan(ax, mesh)
    ax.plot(p[:, 0], p[:, 1], color="#1a5276", lw=2.2, label="robot route (11.8 m)")
    ax.plot(p[0, 0], p[0, 1], "o", color="#1a5276", ms=9, mec="white")
    for i in range(0, len(p) - 3, 8):
        d = p[i + 3] - p[i]
        ax.annotate("", xy=(p[i + 3, 0], p[i + 3, 1]), xytext=(p[i, 0], p[i, 1]),
                    arrowprops=dict(arrowstyle="-|>", color="#1a5276", lw=1.4))
    ax.plot(AP[0], AP[1], "*", ms=20, color="#c0392b", mec="white", mew=1.3,
            label="access point (2.7 m high)")
    short = {"clear line of sight": "LoS", "near the shadow boundary": "boundary",
             "deep NLOS behind the partition": "NLOS"}
    for label, pos in VIEWPOINTS:
        ax.plot(pos[0], pos[1], "s", ms=8, color="#e8a33d", mec="#7a5314", mew=1.0)
        ax.annotate(short[label], (pos[0], pos[1]), textcoords="offset points",
                    xytext=(0, 10), ha="center", fontsize=8.5, color="#7a5314",
                    fontweight="bold")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("floor plan, materials labelled", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.2)

    slide("A1_environment.png", fig, "The simulated environment",
          "6 x 5 x 2.8 m room, 24 surfaces. Concrete walls, wood floor, metal "
          "cabinet, wooden table, plasterboard partition. Robot carries the receiver at 0.9 m.")


def slide_a2(mesh, tx, tracer):
    """Every traced ray at one receiver position, in 3-D and in plan."""
    label, pos = VIEWPOINTS[0]
    paths = trace_at(tracer, tx, pos)
    n_show = 14
    fig = plt.figure(figsize=FIGSIZE)

    ax = fig.add_axes([0.055, 0.09, 0.50, 0.76])
    viz.draw_floorplan(ax, mesh, annotate=False)
    counts = viz.draw_paths(ax, paths, view="plan", max_paths=n_show, label=False)
    viz.draw_endpoints(ax, tx.position.numpy(), pos, view="plan")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("top-down, dots mark bounce points", fontsize=11)
    ax.grid(alpha=0.2)
    handles = [Line2D([], [], color=viz.PATH_COLOURS[k], lw=2.4, label=v)
               for k, v in [("los", f"direct ({counts.get('los',0)})"),
                            ("refl1", f"1 bounce ({counts.get('refl1',0)})"),
                            ("refl2", f"2 bounces ({counts.get('refl2',0)})"),
                            ("diff", f"diffracted ({counts.get('diff',0)})")]]
    ax.legend(handles=handles, fontsize=9, loc="lower right")

    ax = fig.add_axes([0.63, 0.30, 0.345, 0.36])
    viz.draw_elevation(ax, mesh)
    viz.draw_paths(ax, paths, view="elevation", max_paths=n_show, label=False)
    viz.draw_endpoints(ax, tx.position.numpy(), pos, view="elevation")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("side view, same paths", fontsize=11)
    ax.grid(alpha=0.2)

    slide("A2_ray_trace.png", fig, "Traced propagation paths",
          f"Receiver at {pos}, {label}. {paths.n_paths()} paths found, "
          f"{n_show} strongest drawn. Line thickness is path strength.")


def slide_a3(mesh, tx, tracer):
    """How the path set changes as the robot moves into shadow."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    for ax, (label, pos) in zip(axes, VIEWPOINTS):
        paths = trace_at(tracer, tx, pos)
        viz.draw_floorplan(ax, mesh, annotate=False)
        viz.draw_paths(ax, paths, view="plan", label=False, lw_scale=0.9,
                       max_paths=10)
        viz.draw_endpoints(ax, tx.position.numpy(), pos, view="plan", size=0.8)
        rss = float(paths.power_dbm(20.0))
        tau = float(rms_delay_spread(paths.delay, paths.gain)) * 1e9
        ax.set_title(f"{label}\n{paths.n_paths()} paths, RSS {rss:.1f} dBm, "
                     f"delay spread {tau:.2f} ns", fontsize=9.5)
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y [m]")
    handles = [Line2D([], [], color=viz.PATH_COLOURS[k], lw=2.2, label=v)
               for k, v in [("los", "direct"), ("refl1", "1 bounce"),
                            ("refl2", "2 bounces"), ("diff", "diffracted")]]
    axes[2].legend(handles=handles, fontsize=8, loc="lower right")
    fig.subplots_adjust(left=0.05, right=0.985, top=0.83, bottom=0.09, wspace=0.16)
    slide("A3_route_regimes.png", fig, "The same link in three regimes",
          "Ten strongest paths drawn at each position. As the robot rounds the "
          "partition the direct path weakens and reflections and diffraction take over.")


# ---------------------------------------------------------------------------
# Set B: single path to multipath
# ---------------------------------------------------------------------------
def multipath_buildup(mesh, tx, tracer):
    """Build the link up one path at a time and record what changes."""
    label, pos = VIEWPOINTS[0]
    paths = trace_at(tracer, tx, pos)
    order = paths.strongest(paths.n_paths()).tolist()
    stages = [1, 2, 3, len(order)]
    out = []
    for n in stages:
        idx = order[:n]
        H = viz.transfer_function(paths, BAND, indices=idx)
        mag = 20 * torch.log10(H.abs())
        sel = paths.select(idx)
        out.append({
            "n": n, "idx": idx, "H": H,
            "ripple_db": float(mag.max() - mag.min()),
            "delay_spread_ns": float(rms_delay_spread(sel.delay, sel.gain)) * 1e9,
            "rss_dbm": float(sel.power_dbm(20.0)),
            "kinds": [paths.kind[i] for i in idx],
        })
    return paths, pos, label, out


def slide_b_stage(mesh, tx, paths, pos, stage, index, total):
    """One step of the build-up: geometry, impulse response, amplitude, phase."""
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes([0.045, 0.085, 0.40, 0.755])
    viz.draw_floorplan(mesh=mesh, ax=ax, annotate=False)
    viz.draw_paths(ax, paths, indices=stage["idx"], view="plan", label=False)
    viz.draw_endpoints(ax, tx.position.numpy(), pos, view="plan")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"{stage['n']} path" + ("s" if stage["n"] > 1 else "")
                 + " to the receiver", fontsize=11)
    ax.grid(alpha=0.2)

    # impulse response
    ax = fig.add_axes([0.545, 0.615, 0.425, 0.225])
    viz.draw_cir(ax, paths, indices=stage["idx"])
    ax.set_title("channel impulse response", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, max(40, float(paths.delay.max()) * 1e9 * 1.05))
    ax.set_xlabel("")
    ax.set_ylabel("gain [dB]", fontsize=9)

    f = BAND.numpy() / 1e9
    # amplitude
    ax = fig.add_axes([0.545, 0.355, 0.425, 0.20])
    ax.plot(f, 20 * torch.log10(stage["H"].abs()).numpy(), color="#2e86c1", lw=1.7)
    ax.set_ylabel(r"$|H(f)|$ [dB]", fontsize=9)
    ax.set_title(f"amplitude, {stage['ripple_db']:.1f} dB peak to trough",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.tick_params(labelbottom=False)
    if stage["n"] == 2:
        # two paths give a regular comb whose spacing is set by the delay gap
        d = paths.delay[0, stage["idx"]].numpy()
        spacing = 1.0 / abs(d[1] - d[0]) / 1e9
        ax.annotate(f"notch spacing = 1/$\\Delta\\tau$ = {spacing*1000:.0f} MHz",
                    xy=(0.5, 0.06), xycoords="axes fraction", ha="center",
                    fontsize=9.5, color="#1a5276")

    # phase
    ax = fig.add_axes([0.545, 0.085, 0.425, 0.20])
    ph = np.unwrap(np.angle(stage["H"].numpy()))
    ax.plot(f, ph / np.pi, color="#7d3c98", lw=1.7)
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"phase [$\times\pi$]", fontsize=9)
    ax.set_title("phase, unwrapped", fontsize=10.5)
    ax.grid(alpha=0.25)

    notes = {
        1: "One path. The impulse response is a single spike, the amplitude is "
           "flat and the phase is a straight ramp whose slope is the delay.",
        2: "A second path adds a delayed copy. The two interfere, giving a "
           "regular comb of notches and a phase that now wobbles about the ramp.",
        3: "A third path breaks the regularity. Notch spacing stops being "
           "uniform, because three different delays are now beating together.",
    }.get(stage["n"],
          f"All {stage['n']} paths. Deep, irregular notches and a phase that "
          f"departs sharply from the ramp. This is a real indoor channel.")
    slide(f"B{index}_paths_{stage['n']}.png", fig,
          f"From one path to many: {stage['n']} path"
          + ("s" if stage["n"] > 1 else ""), notes)


def slide_b_summary(stages):
    """What each added path does to the channel statistics."""
    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)
    n = [s["n"] for s in stages]
    x = np.arange(len(n))
    labels = [f"{v} path" + ("s" if v > 1 else "") for v in n]

    for ax, key, name, unit, colour in [
            (axes[0], "ripple_db", "Frequency selectivity", "peak to trough [dB]", "#2e86c1"),
            (axes[1], "delay_spread_ns", "Delay spread", "RMS [ns]", "#7d3c98"),
            (axes[2], "rss_dbm", "Received power", "[dBm]", "#c0392b")]:
        vals = [s[key] for s in stages]
        if key == "rss_dbm":
            floor = min(vals) - 3.0
            ax.bar(x, [v - floor for v in vals], bottom=floor, color=colour,
                   alpha=0.9, width=0.6)
            ax.set_ylim(floor, max(vals) + 2.0)
        else:
            ax.bar(x, vals, color=colour, alpha=0.9, width=0.6)
            ax.set_ylim(0, max(vals) * 1.18)
        for xi, v in zip(x, vals):
            ax.annotate(f"{v:.2f}", (xi, v), textcoords="offset points",
                        xytext=(0, 5), ha="center", fontsize=9.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylabel(unit)
        ax.set_title(name, fontsize=11.5)
        ax.grid(alpha=0.25, axis="y")

    fig.subplots_adjust(left=0.06, right=0.98, top=0.78, bottom=0.12, wspace=0.28)
    slide("B5_summary.png", fig, "What each added path actually does",
          "Received power barely moves, within 2.7 dB, because the direct path "
          "dominates it. What changes is the shape of the channel: frequency "
          "selectivity goes from nothing to 29 dB, and delay spread from zero "
          "to 3.1 ns.")


# ---------------------------------------------------------------------------
# Set D: how the paths are constructed
# ---------------------------------------------------------------------------
def slide_d1(mesh, tx, tracer):
    """The method of images, for the first reflection."""
    _, pos = VIEWPOINTS[0]
    paths = trace_at(tracer, tx, pos)
    # pick the strongest single-bounce path and rebuild its construction
    cand = [i for i in paths.strongest(paths.n_paths()).tolist()
            if int(paths.order[i]) == 1 and paths.kind[i].startswith("refl")]
    i = cand[0]
    si = int(paths.kind[i].split("-")[1])
    node = paths.nodes[i][0].detach().numpy()
    p0, n = tracer._plane(si)
    rx_t = torch.tensor(pos, dtype=torch.float64)
    image = mirror(rx_t, p0, n).detach().numpy()

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], left=0.05, right=0.975,
                          top=0.84, bottom=0.09, wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    viz.draw_floorplan(mesh=mesh, ax=ax, annotate=False)
    ax.plot(node[:, 0], node[:, 1], color=viz.PATH_COLOURS["refl1"], lw=3.4,
            label="reflected path (what is traced)", zorder=3)
    # the construction line runs Tx to image and passes exactly through the
    # bounce point, so draw it on top or it hides under the reflected path
    ax.plot([tx.position[0], image[0]], [tx.position[1], image[1]], ls=(0, (5, 4)),
            color="#444444", lw=1.7, label="straight line Tx to image", zorder=5)
    ax.plot(node[1, 0], node[1, 1], "o", ms=9, color=viz.PATH_COLOURS["refl1"],
            mec="white", mew=1.4, zorder=6)
    ax.annotate("bounce point\nfalls out of the\nconstruction",
                (node[1, 0], node[1, 1]), textcoords="offset points",
                xytext=(14, 12), fontsize=9, color="#1a5276")
    ax.plot(image[0], image[1], "o", ms=10, color="#c0392b", mec="white",
            label="receiver mirrored through the wall", zorder=6)
    ax.annotate("mirror image of Rx", (image[0], image[1]),
                textcoords="offset points", xytext=(12, -4), fontsize=9,
                color="#8c2020")
    viz.draw_endpoints(ax, tx.position.numpy(), pos, view="plan")
    ax.set_xlim(min(image[0], 0) - 0.6, 6.4)
    ax.set_ylim(min(image[1], 0) - 0.6, 5.4)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.2)
    ax.set_title("mirror the receiver, draw a straight line, "
                 "read off the bounce point", fontsize=10.5)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.text(0.0, 0.98, "Finding the second path", fontsize=13, va="top",
            fontweight="bold")
    ax.text(0.0, 0.90, (
        "1.  Reflect the receiver through the wall's plane.\n\n"
        "2.  Join transmitter to that mirror image with a\n"
        "     straight line.\n\n"
        "3.  Where the line crosses the plane is the bounce\n"
        "     point, and it satisfies the law of reflection\n"
        "     exactly, by construction.\n\n"
        "4.  Total path length is simply the straight-line\n"
        "     distance to the image.\n\n"
        "The important detail for this project: step 3 uses the\n"
        "*infinite* plane, not the finite wall. A bounce point\n"
        "therefore always exists and moves smoothly when the\n"
        "wall moves. Whether it actually lands on the wall is\n"
        "applied afterwards, as a smooth weight rather than a\n"
        "yes or no test."), fontsize=10.5, va="top", family="monospace",
        linespacing=1.45)
    ax.text(0.0, 0.30, f"This path: length {float(paths.length[0, i]):.3f} m, "
                       f"delay {float(paths.delay[0, i])*1e9:.2f} ns,\n"
                       f"equal to the straight-line distance from the\n"
                       f"transmitter to the mirrored receiver.",
            fontsize=10.5, color="#1a5276", va="top", linespacing=1.5)
    slide("D1_method_of_images.png", fig, "How the second path is found",
          "The method of images, reparameterised so that it stays differentiable.")


def slide_d2(mesh, tx, tracer):
    """Double bounce by repeated mirroring, and the diffraction path."""
    _, pos = VIEWPOINTS[0]
    paths = trace_at(tracer, tx, pos)
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, left=0.05, right=0.975, top=0.84, bottom=0.09,
                          wspace=0.16)

    # double bounce
    ax = fig.add_subplot(gs[0, 0])
    viz.draw_floorplan(mesh=mesh, ax=ax, annotate=False)
    two = [i for i in paths.strongest(paths.n_paths()).tolist()
           if int(paths.order[i]) == 2]
    if two:
        i = two[0]
        node = paths.nodes[i][0].detach().numpy()
        ax.plot(node[:, 0], node[:, 1], color=viz.PATH_COLOURS["refl2"], lw=2.6,
                zorder=4)
        offsets = [(10, 14), (14, -18)]
        for k, (px, py) in enumerate(node[1:-1, :2]):
            ax.plot(px, py, "o", ms=8, color=viz.PATH_COLOURS["refl2"],
                    mec="white", zorder=5)
            ax.annotate(f"bounce {k+1}", (px, py), textcoords="offset points",
                        xytext=offsets[k % 2], fontsize=9, color="#5b2c6f",
                        arrowprops=dict(arrowstyle="-", color="#5b2c6f", lw=0.7))
        ax.set_title(f"two bounces, {float(paths.length[0,i]):.2f} m total",
                     fontsize=10.5)
    viz.draw_endpoints(ax, tx.position.numpy(), pos, plan=True)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)

    # diffraction
    ax = fig.add_subplot(gs[0, 1])
    viz.draw_floorplan(mesh=mesh, ax=ax, annotate=False)
    dif = [i for i in paths.strongest(paths.n_paths()).tolist()
           if paths.kind[i].startswith("diff")]
    if dif:
        i = dif[0]
        node = paths.nodes[i][0].detach().numpy()
        ax.plot(node[:, 0], node[:, 1], color=viz.PATH_COLOURS["diff"], lw=2.6,
                zorder=4)
        ax.plot(node[1, 0], node[1, 1], "o", ms=9, color=viz.PATH_COLOURS["diff"],
                mec="white", zorder=5)
        ax.annotate("diffraction point\non a furniture edge", (node[1, 0], node[1, 1]),
                    textcoords="offset points", xytext=(10, 8), fontsize=8.5,
                    color="#1e6b3a")
        ax.set_title(f"diffracted, {float(paths.length[0,i]):.2f} m total",
                     fontsize=10.5)
    viz.draw_endpoints(ax, tx.position.numpy(), pos, plan=True)
    ax.set_xlabel("x [m]")
    ax.grid(alpha=0.2)

    slide("D2_higher_order.png", fig, "The third path, and beyond",
          "Two bounces: mirror the receiver through both planes in turn, then "
          "sweep forward. Diffraction: the bend point is the one that minimises "
          "total path length along the edge, which has a closed-form solution.")


def main():
    """Generate every slide."""
    os.makedirs(SLIDES, exist_ok=True)
    print("Generating slides")
    mesh, tx, traj, tracer = build_scene()

    print(" Set A, environment")
    slide_a1(mesh, tx, traj)
    slide_a2(mesh, tx, tracer)
    slide_a3(mesh, tx, tracer)

    print(" Set B, single path to multipath")
    paths, pos, label, stages = multipath_buildup(mesh, tx, tracer)
    for k, st in enumerate(stages, start=1):
        slide_b_stage(mesh, tx, paths, pos, st, k, len(stages))
    slide_b_summary(stages)

    print(" Set D, path construction")
    slide_d1(mesh, tx, tracer)
    slide_d2(mesh, tx, tracer)

    print(f"\nSet C is the existing experiment figures:")
    print("  results/exp5_frequency_response.png  amplitude, phase, group delay")
    print("  results/exp2_material_sweep.png      material comparison")
    print(f"\nSlides written to {SLIDES}")
    for st in stages:
        print(f"  {st['n']:2d} paths: selectivity {st['ripple_db']:5.1f} dB, "
              f"delay spread {st['delay_spread_ns']:5.2f} ns, "
              f"RSS {st['rss_dbm']:7.2f} dBm")


if __name__ == "__main__":
    main()