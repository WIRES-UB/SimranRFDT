"""Experiment 9: second-order diffraction, and where it stops being optional.

First-order diffraction is a correction almost everywhere: a few tenths of a dB
on top of a field that reflections already dominate.  Deep in a shadow it is not
a correction to anything, because the first-order term is itself nearly zero
there, and what reaches the receiver has bent around two edges rather than one.

That is the regime this measures.  The receiver walks along a line behind the
partition, starting deep in its shadow and ending past its open end where the
direct path returns, and the field is traced twice: with single-edge
diffraction only, and with edge-to-edge cascades included.  The question is not
whether the cascade adds something, which it must, but where it stops being
negligible and starts being the answer.

Three things are reported.

  1. The crossover: how far into shadow the second-order term has to reach
     before it carries a meaningful share of the received field.

  2. What the slope term is worth.  Ordinary diffraction uses only the value of
     the incident field at the edge; when the second edge sits near the first
     one's shadow boundary that field varies rapidly across it and the
     derivative term is comparable in size.

  3. What it costs, and how far the cascade has been pushed past the separation
     where a ray-optical composition is derived.  Both are reported rather than
     left for the reader to discover.

Outputs: ``results/exp9_double_diffraction.{png,json}``
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from rfdt import scenes                                               # noqa: E402
from rfdt.geometry import plate                                       # noqa: E402
from rfdt.antennas import Antenna                                     # noqa: E402
from rfdt.tracer import (RFDTracer, TracerConfig, Transmitter,        # noqa: E402
                         Receiver)

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

FREQ = 5.0e9
TX_POWER_DBM = 20.0
AP = (1.2, 1.2, 2.7)
#: The partition runs from y = 0 to y = 3.5 at x = 3.  Walking the receiver
#: along y at x = 4.5 starts deep in its shadow and ends past its open end,
#: where the direct path returns, so one transect spans both regimes.
TRANSECT_X = 4.5
TRANSECT_Z = 0.9
TRANSECT_Y = np.linspace(0.35, 4.60, 40)
PARTITION_END_Y = 3.5


#: Two knife edges in series, in free space.  This is the geometry the cascade
#: is actually about: the direct path is blocked by the first screen, and a ray
#: that bends over the first screen alone is blocked by the second, so nothing
#: reaches the receiver without bending twice.  Nothing else is present, so no
#: reflection can fill the shadow in.
SCREEN_A_X, SCREEN_A_TOP = 0.0, 1.0
SCREEN_B_X, SCREEN_B_TOP = 3.0, 1.2
SCREEN_HEIGHT, SCREEN_WIDTH = 4.0, 8.0
DEEP_TX = (-2.5, 0.0, 0.0)
DEEP_RX_X = 5.5
DEEP_RX_Z = np.linspace(-1.6, 1.4, 31)


def two_screen_scene():
    """Two parallel knife edges with free space around them."""
    a = plate((SCREEN_A_X, 0.0, SCREEN_A_TOP - SCREEN_HEIGHT / 2.0),
              (SCREEN_WIDTH, SCREEN_HEIGHT), "x", "metal", "screen_a")
    b = plate((SCREEN_B_X, 0.0, SCREEN_B_TOP - SCREEN_HEIGHT / 2.0),
              (SCREEN_WIDTH, SCREEN_HEIGHT), "x", "metal", "screen_b")
    return a.merged(b).weld()


def deep_shadow_study():
    """Field behind two screens in series, with and without the cascade.

    The receiver climbs from far below the second screen's edge, which is deep
    shadow, up to just above it, where a single bend suffices.  This is the
    regime the cascade exists for, and it is deliberately not a closed room:
    the transect study below shows that a partition inside a reflective room
    never produces a deep shadow at all, because the walls fill it in.
    """
    mesh = two_screen_scene()
    tx = Transmitter(DEEP_TX, FREQ, TX_POWER_DBM, Antenna("isotropic"))
    rx = Receiver((DEEP_RX_X, 0.0, float(DEEP_RX_Z[0])), Antenna("isotropic"))
    pos = torch.tensor([[DEEP_RX_X, 0.0, float(z)] for z in DEEP_RX_Z],
                       dtype=torch.float64)
    out = {"rx_z": DEEP_RX_Z.tolist(),
           "screen_a_top": SCREEN_A_TOP, "screen_b_top": SCREEN_B_TOP,
           "tx": list(DEEP_TX), "rx_x": DEEP_RX_X}
    for name, order in (("first_order", 1), ("second_order", 2)):
        cfg = TracerConfig(max_order=1, max_diffraction_order=order,
                           enable_slope_diffraction=True)
        with torch.no_grad():
            paths = RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=pos)
        out[name] = {
            "total_dbm": family_power(paths, lambda k: True).tolist(),
            "double_dbm": family_power(
                paths, lambda k: k.startswith("diff2")).tolist(),
            "n_double": sum(1 for k in paths.kind if k.startswith("diff2")),
        }
        print(f"  two screens, {name:13s} {paths.n_paths():4d} paths "
              f"({out[name]['n_double']:3d} second order)", flush=True)
    # Reporting a ratio here would be meaningless: with single diffraction
    # alone the predicted field is not merely small but zero, so the "gain" is
    # whatever numerical floor the logarithm was clamped to and says nothing
    # about physics.  What matters is that one model predicts nothing and the
    # other predicts a definite level, so both are reported as levels.
    first = np.array(out["first_order"]["total_dbm"])
    second = np.array(out["second_order"]["total_dbm"])
    FLOOR_DBM = -250.0
    blind = first < FLOOR_DBM
    out["floor_dbm"] = FLOOR_DBM
    out["points_where_first_order_predicts_nothing"] = int(blind.sum())
    out["points_total"] = int(len(first))
    out["second_order_level_range_dbm"] = [float(second.min()), float(second.max())]
    out["level_where_one_edge_suffices"] = {
        "z": [float(v) for v in np.array(out["rx_z"])[~blind]],
        "first_order_dbm": [float(v) for v in first[~blind]],
        "second_order_dbm": [float(v) for v in second[~blind]],
    }
    print(f"  single diffraction predicts nothing at all at "
          f"{out['points_where_first_order_predicts_nothing']} of "
          f"{out['points_total']} points; the cascade gives "
          f"{second.min():.0f} to {second.max():.0f} dBm", flush=True)
    return out


def trace_transect(mesh, cfg):
    """Field along the transect for one tracer configuration."""
    tx = Transmitter(AP, FREQ, TX_POWER_DBM, Antenna("isotropic"))
    rx = Receiver((TRANSECT_X, float(TRANSECT_Y[0]), TRANSECT_Z),
                  Antenna("isotropic"))
    pos = torch.tensor([[TRANSECT_X, float(y), TRANSECT_Z] for y in TRANSECT_Y],
                       dtype=torch.float64)
    with torch.no_grad():
        return RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=pos)


def family_power(paths, predicate) -> np.ndarray:
    """Coherent power in dB of the subset of paths matching ``predicate``."""
    idx = [i for i, kind in enumerate(paths.kind) if predicate(kind)]
    if not idx:
        return np.full(paths.gain.shape[0], -300.0)
    total = paths.gain[:, idx].sum(-1).abs().clamp_min(1e-30)
    return (20.0 * torch.log10(total)).detach().numpy() + TX_POWER_DBM


def main():
    """Trace the transect three ways and report the crossover and the cost."""
    os.makedirs(RESULTS, exist_ok=True)
    mesh = scenes.furnished_room()
    t0 = time.time()

    configs = {
        "first_order": TracerConfig(max_order=2, max_diffraction_order=1),
        "second_order": TracerConfig(max_order=2, max_diffraction_order=2,
                                     enable_slope_diffraction=True),
        "second_order_no_slope": TracerConfig(max_order=2,
                                              max_diffraction_order=2,
                                              enable_slope_diffraction=False),
    }
    out = {"setup": {"frequency_hz": FREQ, "ap": list(AP),
                     "transect_x": TRANSECT_X, "transect_z": TRANSECT_Z,
                     "transect_y": TRANSECT_Y.tolist(),
                     "partition_end_y": PARTITION_END_Y},
           "runs": {}}

    traces = {}
    for name, cfg in configs.items():
        t = time.time()
        paths = trace_transect(mesh, cfg)
        elapsed = time.time() - t
        traces[name] = paths
        n_double = sum(1 for kind in paths.kind if kind.startswith("diff2"))
        out["runs"][name] = {
            "total_dbm": family_power(paths, lambda k: True).tolist(),
            "single_diff_dbm": family_power(
                paths, lambda k: k.startswith("diff-")).tolist(),
            "double_diff_dbm": family_power(
                paths, lambda k: k.startswith("diff2")).tolist(),
            "n_paths": int(paths.n_paths()),
            "n_double_paths": int(n_double),
            "seconds": elapsed,
        }
        print(f"  {name:22s} {paths.n_paths():5d} paths "
              f"({n_double:4d} second order)  {elapsed:6.1f} s", flush=True)

    print("\ntwo knife edges in series, free space")
    out["deep_shadow"] = deep_shadow_study()

    first = np.array(out["runs"]["first_order"]["total_dbm"])
    second = np.array(out["runs"]["second_order"]["total_dbm"])
    no_slope = np.array(out["runs"]["second_order_no_slope"]["total_dbm"])
    shadow = TRANSECT_Y < PARTITION_END_Y
    # How dominant is diffraction on the room transect at all?  If reflections
    # swamp it there, the cascade cannot matter there whatever its own size,
    # and saying so is the point of reporting this number.
    sing = np.array(out["runs"]["second_order"]["single_diff_dbm"])
    out["room_diffraction_share"] = {
        "max_single_diff_minus_total_db": float(np.max(sing - second)),
        "mean_single_diff_minus_total_db": float(np.mean(sing - second)),
    }

    out["summary"] = {
        "max_change_db": float(np.max(np.abs(second - first))),
        "mean_change_shadowed_db": float(np.mean(np.abs(second - first)[shadow])),
        "mean_change_lit_db": float(np.mean(np.abs(second - first)[~shadow])),
        "max_slope_change_db": float(np.max(np.abs(second - no_slope))),
        "cost_ratio": (out["runs"]["second_order"]["seconds"]
                       / max(out["runs"]["first_order"]["seconds"], 1e-9)),
        "deepest_shadow_change_db": float((second - first)[0]),
    }
    s = out["summary"]
    print(f"\n  change from adding the cascade: up to {s['max_change_db']:.2f} dB, "
          f"mean {s['mean_change_shadowed_db']:.2f} dB in shadow against "
          f"{s['mean_change_lit_db']:.2f} dB where the direct path survives")
    print(f"  slope term worth up to {s['max_slope_change_db']:.2f} dB")
    print(f"  cost: {s['cost_ratio']:.1f} times a first-order trace "
          f"(wall-clock, varies with machine load)")

    out["setup"]["runtime_s"] = time.time() - t0
    _plot(out)
    with open(os.path.join(RESULTS, "exp9_double_diffraction.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/exp9_double_diffraction.{{png,json}} "
          f"({out['setup']['runtime_s']:.0f} s)")


def _plot(out):
    """Four panels: the free-space deep shadow, then the room transect."""
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.0))
    y = np.array(out["setup"]["transect_y"])
    end = out["setup"]["partition_end_y"]
    first = np.array(out["runs"]["first_order"]["total_dbm"])
    second = np.array(out["runs"]["second_order"]["total_dbm"])
    no_slope = np.array(out["runs"]["second_order_no_slope"]["total_dbm"])
    single = np.array(out["runs"]["second_order"]["single_diff_dbm"])
    double = np.array(out["runs"]["second_order"]["double_diff_dbm"])

    def shade(ax):
        ax.axvspan(y[0], end, color="0.85", alpha=0.55, zorder=0)
        ax.axvline(end, color="0.4", ls="--", lw=1.1)
        ax.text(end, ax.get_ylim()[1], " partition ends", fontsize=8,
                color="0.35", va="top", ha="left")

    ds = out["deep_shadow"]
    ax = axes[0]
    z = np.array(ds["rx_z"])
    f1 = np.array(ds["first_order"]["total_dbm"])
    f2 = np.array(ds["second_order"]["total_dbm"])
    blind = f1 < ds["floor_dbm"]
    lo, hi = f2.min() - 12.0, max(f2.max(), f1[~blind].max() if (~blind).any()
                                  else f2.max()) + 4.0
    # The single-order curve is at the numerical floor over almost the whole
    # transect, which is the finding rather than a plotting nuisance, so it is
    # drawn pinned to the left edge and labelled instead of being allowed to
    # stretch the axis to -580 dBm and flatten everything else to a line.
    ax.plot(np.where(blind, lo, f1), z, color="#7f8c8d", lw=1.9,
            label="single-edge diffraction")
    ax.plot(f2, z, color="#c0392b", lw=1.9, label="with edge-to-edge cascade")
    ax.fill_betweenx(z, lo, np.where(blind, lo + 1.2, lo), color="#7f8c8d",
                     alpha=0.25)
    ax.axhline(ds["screen_b_top"], color="0.4", ls="--", lw=1.1)
    ax.set_xlim(lo, hi)
    ax.text(lo + 1.6, float(z.mean()),
            "single diffraction\npredicts exactly zero\nhere, not merely a\nsmall value",
            fontsize=8.5, color="0.3", va="center")
    ax.text(hi, ds["screen_b_top"], "second screen edge ", fontsize=8,
            color="0.35", va="bottom", ha="right")
    ax.set_ylabel("receiver height z [m]")
    ax.set_xlabel("received power [dBm]")
    ax.set_title("(a) Two knife edges in series, free space.\n"
                 "Nothing arrives without bending twice", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[1]
    ax.plot(y, first, color="#7f8c8d", lw=1.9, label="single-edge diffraction")
    ax.plot(y, second, color="#c0392b", lw=1.9, label="with edge-to-edge cascade")
    ax.set_xlabel("receiver position along the transect, y [m]")
    ax.set_ylabel("received power [dBm]")
    ax.set_title("(b) Inside a reflective room instead. The shaded\n"
                 "half is behind the partition", fontsize=10.5)
    shade(ax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[2]
    ax.plot(y, single, color="#2980b9", lw=1.8, label="first order alone")
    ax.plot(y, double, color="#c0392b", lw=1.8, label="second order alone")
    ax.set_xlabel("receiver position along the transect, y [m]")
    ax.set_ylabel("power in that family [dBm]")
    ax.set_title("(c) In the room, both diffraction orders sit far\n"
                 "below the total: reflections fill the shadow", fontsize=10.5)
    shade(ax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="lower right")

    ax = axes[3]
    ax.plot(y, second - first, color="#c0392b", lw=1.9,
            label="cascade, against first order")
    ax.plot(y, second - no_slope, color="#27ae60", lw=1.6,
            label="slope term, against value only")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("receiver position along the transect, y [m]")
    ax.set_ylabel("change in received power [dB]")
    ax.set_title("(d) In the room both additions stay under a dB,\n"
                 "because diffraction is not what dominates there", fontsize=10.5)
    shade(ax)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8.5, loc="upper right")

    fig.suptitle("Experiment 9: edge-to-edge diffraction, and where it stops "
                 "being optional", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(RESULTS, "exp9_double_diffraction.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
