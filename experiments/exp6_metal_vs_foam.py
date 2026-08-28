"""Experiment 6: metal against foam, head to head.

The eleven-material sweep of experiment 2 establishes that the channel changes
monotonically with reflectivity.  This experiment drops everything except the
two extremes and compares them directly, because the contrast is what makes the
mechanism visible: a metal-walled room and a foam-walled room are the same
geometry, the same transmitter and the same route, differing only in how much
energy the walls send back.

Metal reflects essentially everything.  Foam reflects 36.7 dB less at normal
incidence, a factor of about 4,300 in power.  Two rooms could hardly differ
more, and the point of the comparison is that received power barely notices
while everything else about the channel changes completely.

Route statistics and profiles are reused from experiment 2 rather than
recomputed, so both experiments report the same numbers.  The impulse and
frequency responses are traced here, since experiment 2 does not store them.

Outputs: ``results/exp6_metal_vs_foam.{png,json}``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from rfdt import antennas, scenes                                     # noqa: E402
from rfdt.materials import get_material, reflection_coefficient       # noqa: E402
from rfdt.signal import ofdm_channel                                  # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
SWEEP_JSON = os.path.join(RESULTS, "exp2_material_sweep.json")

#: The two materials, and the colour each keeps in every panel.
PAIR = [("metal", "Metal", "#5d6d7e"), ("foam_board", "Foam board", "#27ae60")]

AP = (1.2, 1.2, 2.7)
TX_POWER_DBM = 20.0
FREQ = 5.0e9
BAND = torch.linspace(4.8e9, 5.2e9, 601, dtype=torch.float64)
#: A receiver position with a clear path to the access point, so the two rooms
#: differ only in their reflected energy rather than in whether the direct path
#: survives at all.
PROBE = (2.0, 1.6, 0.9)


def load_sweep():
    """Route statistics for the two materials, from experiment 2.

    Raises rather than silently recomputing, so the two experiments cannot
    drift apart and quote different numbers for the same quantity.
    """
    if not os.path.exists(SWEEP_JSON):
        raise FileNotFoundError(
            f"{SWEEP_JSON} not found. Run experiments/exp2_material_sweep.py "
            "first; this experiment reuses its route statistics so that both "
            "report identical numbers.")
    return json.load(open(SWEEP_JSON))


def trace_probe(material):
    """Trace every path at the probe position for one wall material."""
    mesh = scenes.furnished_room(wall=material)
    tracer = RFDTracer(mesh, TracerConfig(max_order=2, weighting="rfdt",
                                          enable_diffraction=True))
    tx = antennas.wifi_ap(AP, FREQ, TX_POWER_DBM)
    pos = torch.tensor([PROBE], dtype=torch.float64)
    return tracer.trace(tx, antennas.robot_client(PROBE), rx_positions=pos)


def main():
    """Run the head-to-head comparison and write the figure and summary."""
    os.makedirs(RESULTS, exist_ok=True)
    sweep = load_sweep()

    print("Experiment 6: metal against foam")
    print(f"  same room, same route, same transmitter; only the walls differ")
    print(f"  probe position for the response panels: {PROBE}\n")

    traced = {}
    for key, label, _ in PAIR:
        p = trace_probe(key)
        H = ofdm_channel(p.delay, p.gain, BAND).squeeze()
        mag = 20 * torch.log10(H.abs())
        traced[key] = {
            "n_paths": p.n_paths(),
            "delays_ns": (p.delay[0].numpy() * 1e9).tolist(),
            "gains_db": (20 * torch.log10(
                p.gain[0].abs() / p.gain[0].abs().max())).numpy().tolist(),
            "H_db": mag.numpy().tolist(),
            "selectivity_db": float(mag.max() - mag.min()),
        }
        print(f"  {label:12s} {p.n_paths():3d} paths at the probe, "
              f"frequency selectivity {traced[key]['selectivity_db']:5.1f} dB")

    print(f"\n{'quantity':34s} {'metal':>12s} {'foam':>12s} {'difference':>12s}")
    rows = [
        ("Reflectivity, normal incidence [dB]", "reflection_db_normal", "5 GHz"),
        ("Route-mean received power [dBm]", "rss_mean_dbm", "5 GHz"),
        ("RMS delay spread [ns]", "delay_spread_ns", "5 GHz"),
        ("Rice K-factor [dB]", "k_factor_db", "5 GHz"),
        ("Angular spread [deg]", "angular_spread_deg", "5 GHz"),
        ("Route-mean received power [dBm]", "rss_mean_dbm", "60 GHz"),
        ("Rice K-factor [dB]", "k_factor_db", "60 GHz"),
    ]
    summary = {}
    for label, key, band in rows:
        m = sweep["bands"][band]["materials"]["metal"][key]
        f = sweep["bands"][band]["materials"]["foam_board"][key]
        tag = f"{label} @ {band}"
        summary[tag] = {"metal": m, "foam": f, "difference": m - f}
        print(f"{label[:30]:30s} {band:>5s} {m:12.2f} {f:12.2f} {m - f:+12.2f}")

    _plot(sweep, traced)
    payload = {"probe_position_m": list(PROBE), "carrier_hz": FREQ,
               "comparison": summary,
               "selectivity_db": {k: traced[k]["selectivity_db"]
                                  for k, _, _ in PAIR},
               "n_paths_at_probe": {k: traced[k]["n_paths"] for k, _, _ in PAIR}}
    with open(os.path.join(RESULTS, "exp6_metal_vs_foam.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote results/exp6_metal_vs_foam.{png,json}")
    return payload


def _plot(sweep, traced):
    """Six panels: the cause, then five consequences."""
    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.28)

    # 1. the cause: reflectivity against angle
    ax = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0.0, 89.5, 200)
    cos_ti = torch.tensor(np.cos(np.deg2rad(theta)), dtype=torch.float64)
    for key, label, colour in PAIR:
        g = reflection_coefficient(FREQ, cos_ti, get_material(key), "perp")
        db = 20 * np.log10(g.abs().numpy() + 1e-12)
        # metal on top: the two curves meet at grazing and metal would
        # otherwise be hidden underneath foam for the last few degrees
        ax.plot(theta, db, color=colour, lw=3.0 if key == "metal" else 2.0,
                label=label, zorder=2 if key == "metal" else 3,
                alpha=0.95)
        ax.annotate(f"{db[0]:.1f} dB", (1.5, db[0]), textcoords="offset points",
                    xytext=(6, 7 if key == "metal" else 7), fontsize=9.5,
                    color=colour, fontweight="bold")
    ax.set_xlabel("incidence angle from normal [deg]")
    ax.set_ylabel(r"$20\log_{10}|\Gamma_\perp|$ [dB]")
    ax.set_title("1. The cause: how much the wall reflects", fontsize=11)
    ax.set_ylim(-40, 3)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.annotate("both converge at grazing:\neven foam reflects well\nat shallow angles",
                (78, -8), fontsize=8, color="#555555", ha="center")

    # 2. impulse response at the probe
    ax = fig.add_subplot(gs[0, 1])
    for key, label, colour in PAIR:
        d = np.asarray(traced[key]["delays_ns"])
        g = np.asarray(traced[key]["gains_db"])
        keep = g > -60
        off = 0.15 if key == "metal" else -0.15
        ax.vlines(d[keep] + off, -60, g[keep], color=colour, lw=1.3, alpha=0.85)
        ax.plot(d[keep] + off, g[keep], "o", color=colour, ms=3.5, label=label)
    ax.set_xlabel("delay [ns]")
    ax.set_ylabel("path gain, relative to strongest [dB]")
    ax.set_title("2. What arrives, and when", fontsize=11)
    ax.set_ylim(-60, 4)
    ax.set_xlim(0, 45)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # 3. frequency response at the probe
    ax = fig.add_subplot(gs[0, 2])
    f = BAND.numpy() / 1e9
    for key, label, colour in PAIR:
        ax.plot(f, traced[key]["H_db"], color=colour, lw=1.5,
                label=f"{label}, {traced[key]['selectivity_db']:.0f} dB deep")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"$|H(f)|$ [dB]")
    ax.set_title("3. The channel across frequency", fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # 4. received power along the route
    ax = fig.add_subplot(gs[1, 0])
    traj = scenes.survey_trajectory(n_samples=48)
    s = np.asarray(traj.arclength)
    for key, label, colour in PAIR:
        prof = sweep["bands"]["5 GHz"]["materials"][key]["rss_profile_dbm"]
        ax.plot(s, prof, color=colour, lw=1.7, label=label)
    diff = (np.asarray(sweep["bands"]["5 GHz"]["materials"]["metal"]["rss_profile_dbm"])
            - np.asarray(sweep["bands"]["5 GHz"]["materials"]["foam_board"]["rss_profile_dbm"]))
    ax.set_xlabel("distance along route [m]")
    ax.set_ylabel("received power [dBm]")
    ax.set_title(f"4. Along the route at 5 GHz\nmean gap only {diff.mean():.1f} dB",
                 fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    # 5. the channel statistics that do differ
    ax = fig.add_subplot(gs[1, 1])
    # angular spread is divided by five so the three metrics share one axis;
    # the printed label must still be the true value, not the divided one
    metrics = [("delay_spread_ns", "Delay spread\n[ns]", 1.0),
               ("k_factor_db", "Rice K-factor\n[dB]", 1.0),
               ("angular_spread_deg", "Angular spread\n[deg, shown / 5]", 5.0)]
    x = np.arange(len(metrics))
    for i, (key, label, colour) in enumerate(PAIR):
        for j, (mk, _, div) in enumerate(metrics):
            true_v = sweep["bands"]["5 GHz"]["materials"][key][mk]
            shown = true_v / div
            xi = x[j] + (i - 0.5) * 0.36
            ax.bar(xi, shown, width=0.34, color=colour,
                   label=label if j == 0 else None)
            ax.annotate(f"{true_v:.1f}", (xi, shown), textcoords="offset points",
                        xytext=(0, 4 if shown >= 0 else -13), ha="center",
                        fontsize=9.5)
    ax.axhline(0, color="#333333", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics], fontsize=9)
    ax.set_title("5. What actually changes, at 5 GHz", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)

    # 6. the inversion at 60 GHz
    ax = fig.add_subplot(gs[1, 2])
    bands = ["5 GHz", "60 GHz"]
    x = np.arange(len(bands))
    for i, (key, label, colour) in enumerate(PAIR):
        vals = [sweep["bands"][b]["materials"][key]["rss_mean_dbm"] for b in bands]
        floor = -85
        ax.bar(x + (i - 0.5) * 0.36, [v - floor for v in vals], bottom=floor,
               width=0.34, color=colour, label=label)
        for xi, v in zip(x + (i - 0.5) * 0.36, vals):
            ax.annotate(f"{v:.1f}", (xi, v), textcoords="offset points",
                        xytext=(0, 4), ha="center", fontsize=9)
    for i, b in enumerate(bands):
        m = sweep["bands"][b]["materials"]["metal"]["rss_mean_dbm"]
        fo = sweep["bands"][b]["materials"]["foam_board"]["rss_mean_dbm"]
        ax.annotate(f"gap {m - fo:.1f} dB",
                    (i, min(m, fo) - 3.5), ha="center",
                    fontsize=10, color="#8c2020", fontweight="bold")
    ax.set_ylim(floor, -32)
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.set_ylabel("route-mean received power [dBm]")
    ax.set_title("6. At 60 GHz the walls start to matter", fontsize=11)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    fig.suptitle("Experiment 6: metal walls against foam walls, "
                 "same room and same route", fontsize=13.5)
    fig.savefig(os.path.join(RESULTS, "exp6_metal_vs_foam.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()