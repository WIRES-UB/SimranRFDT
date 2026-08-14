"""Experiment 2: indoor channel versus wall material (the headline result).

A mobile robot drives a fixed route through a furnished 6 x 5 x 2.8 m room
while a ceiling-mounted access point transmits.  Everything is held constant
except the material of one surface class, which is swept over the ITU-R
P.2040-1 database.  For each material we report what a robot's radio would
actually experience along the route:

  * received power (RSS) and its variability,
  * RMS delay spread and Rice K-factor,
  * coherence bandwidth and angular spread,
  * the deterministic reflection and penetration characteristics of the
    material itself, so the channel results can be read against the physics.

Two carrier frequencies are used, 5 GHz and 60 GHz, the two representative
bands of App. C.2.

A note on methodology.  At a single point and a single frequency the received
power is dominated by multipath fading, which can swing tens of dB for reasons
that have nothing to do with the material.  Every channel statistic here is
therefore averaged over the route *and* over a small frequency band, which is
what a real measurement campaign does.  Comparing single points would mostly
measure fading.

Outputs: ``results/exp2_material_sweep.{png,json,csv}``
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from rfdt import antennas, scenes                                     # noqa: E402
from rfdt.materials import (C0, MATERIALS, get_material,              # noqa: E402
                            reflection_coefficient)
from rfdt.metrics import (angular_spread, rice_k_factor_db,           # noqa: E402
                          rms_delay_spread, coherence_bandwidth)
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

#: Materials swept over. All ITU-R P.2040-1 entries valid in both bands, plus
#: two clearly-labelled approximate entries for contrast.
SWEEP = ["metal", "concrete", "brick", "marble", "glass", "plasterboard",
         "chipboard", "wood", "ceiling_board", "foam_board", "human_body"]

#: Carrier frequencies [Hz]: the two bands evaluated in App. C.2.
BANDS = {"5 GHz": 5.0e9, "60 GHz": 60.0e9}

#: Frequency samples per band used to average out small-scale fading.
N_FREQ = 5
FREQ_SPAN = 0.02          # +-1 % around the carrier

AP_POSITION = (1.2, 1.2, 2.7)
TX_POWER_DBM = 20.0
N_ROUTE = 48


def material_physics(name, f_hz):
    """Deterministic material characteristics, independent of the scene.

    Returns the normal-incidence and 45-degree reflection magnitudes and the
    one-way penetration loss through 10 cm, so the channel statistics can be
    interpreted against the underlying electromagnetics (Eq. 56-59).
    """
    mat = get_material(name)
    d = mat.describe(f_hz)
    g0 = reflection_coefficient(f_hz, torch.tensor(1.0, dtype=torch.float64),
                                mat, "perp")
    g45 = reflection_coefficient(f_hz, torch.tensor(np.cos(np.pi / 4),
                                                    dtype=torch.float64), mat, "perp")
    return {
        "eps_real": d["eps_real"],
        "sigma_S_per_m": d["sigma_S_per_m"],
        "loss_tangent": d["loss_tangent"],
        "reflection_db_normal": float(20 * np.log10(abs(complex(g0)))),
        "reflection_db_45deg": float(20 * np.log10(abs(complex(g45)))),
        "penetration_loss_db_10cm": float(mat.penetration_loss_db(f_hz, 0.10)),
        "source": d["source"],
        "f_range_ghz": d["f_range_ghz"],
        "in_validity_range": d["in_validity_range"],
    }


def run_route(material, f_centre, traj, cfg):
    """Trace the whole robot route for one wall material and one band.

    The wall, floor and ceiling classes keep their own materials; only the
    four vertical walls are swept, since those dominate the reflected
    multipath a floor-level robot sees.

    Returns a dictionary of route-averaged channel statistics.
    """
    mesh = scenes.furnished_room(wall=material)
    tracer = RFDTracer(mesh, cfg)
    rx = antennas.robot_client(traj.positions[0])

    # frequency samples across a narrow band, to average out small-scale fading
    freqs = np.linspace(f_centre * (1 - FREQ_SPAN / 2),
                        f_centre * (1 + FREQ_SPAN / 2), N_FREQ)
    rss, ds, kf, cb, asp, npath = [], [], [], [], [], []
    for f in freqs:
        tx = antennas.wifi_ap(AP_POSITION, float(f), TX_POWER_DBM)
        paths = tracer.trace(tx, rx, rx_positions=traj.positions)
        rss.append(paths.power_dbm(TX_POWER_DBM))
        ds.append(rms_delay_spread(paths.delay, paths.gain) * 1e9)
        kf.append(rice_k_factor_db(paths.gain))
        cb.append(coherence_bandwidth(paths.delay, paths.gain) / 1e6)
        asp.append(angular_spread(paths.arr_dir, paths.gain) * 180.0 / np.pi)
        npath.append(paths.n_paths())

    rss = torch.stack(rss)                 # (F, R)
    # band-average in the power domain, which is the physically meaningful mean
    rss_lin = 10.0 ** (rss / 10.0)
    rss_avg = 10.0 * torch.log10(rss_lin.mean(dim=0))
    return {
        "rss_mean_dbm": float(rss_avg.mean()),
        "rss_std_db": float(rss_avg.std()),
        "rss_min_dbm": float(rss_avg.min()),
        "rss_max_dbm": float(rss_avg.max()),
        "fade_depth_db": float(rss_avg.max() - rss_avg.min()),
        "delay_spread_ns": float(torch.stack(ds).mean()),
        "k_factor_db": float(torch.stack(kf).mean()),
        "coherence_bw_mhz": float(torch.stack(cb).mean()),
        "angular_spread_deg": float(torch.stack(asp).mean()),
        "n_paths": int(np.mean(npath)),
        "rss_profile_dbm": rss_avg.tolist(),
    }


def main():
    """Sweep every material in both bands, then write the figure and tables."""
    os.makedirs(RESULTS, exist_ok=True)
    traj = scenes.survey_trajectory(n_samples=N_ROUTE)
    cfg = TracerConfig(max_order=2, weighting="rfdt", enable_diffraction=True)

    out = {"setup": {"room_m": [6.0, 5.0, 2.8], "ap_position": list(AP_POSITION),
                     "tx_power_dbm": TX_POWER_DBM, "route_samples": N_ROUTE,
                     "route_length_m": float(traj.arclength[-1]),
                     "freq_samples_per_band": N_FREQ, "max_order": 2},
           "bands": {}}

    t0 = time.time()
    for band, f_c in BANDS.items():
        print(f"\n=== {band} ===")
        print(f"{'material':15s} {'RSS[dBm]':>9s} {'sd':>6s} {'fade':>6s} "
              f"{'tau[ns]':>8s} {'K[dB]':>7s} {'Bc[MHz]':>8s} {'AS[deg]':>8s} "
              f"{'|G|0[dB]':>9s} {'pen10cm':>8s}")
        out["bands"][band] = {"frequency_hz": f_c, "materials": {}}
        for name in SWEEP:
            phys = material_physics(name, f_c)
            chan = run_route(name, f_c, traj, cfg)
            rec = {**chan, **phys}
            out["bands"][band]["materials"][name] = rec
            print(f"{name:15s} {rec['rss_mean_dbm']:9.2f} {rec['rss_std_db']:6.2f} "
                  f"{rec['fade_depth_db']:6.1f} {rec['delay_spread_ns']:8.2f} "
                  f"{rec['k_factor_db']:7.2f} {rec['coherence_bw_mhz']:8.1f} "
                  f"{rec['angular_spread_deg']:8.1f} "
                  f"{rec['reflection_db_normal']:9.2f} "
                  f"{rec['penetration_loss_db_10cm']:8.2f}"
                  f"{'' if rec['in_validity_range'] else '  (*)'}")
    out["setup"]["runtime_s"] = time.time() - t0
    print("\n(*) marks a material evaluated outside the frequency range over "
          "which its ITU-R P.2040-1 regression is specified; the value is an "
          "extrapolation.")

    _write_csv(out)
    _plot(out, traj)
    with open(os.path.join(RESULTS, "exp2_material_sweep.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/exp2_material_sweep.{{png,json,csv}} "
          f"({out['setup']['runtime_s']:.0f} s)")
    return out


def _write_csv(out):
    """Flat CSV of every material in every band, for spreadsheets."""
    path = os.path.join(RESULTS, "exp2_material_sweep.csv")
    cols = ["band", "frequency_ghz", "material", "source", "in_validity_range",
            "eps_real",
            "sigma_S_per_m", "loss_tangent", "reflection_db_normal",
            "reflection_db_45deg", "penetration_loss_db_10cm", "rss_mean_dbm",
            "rss_std_db", "fade_depth_db", "delay_spread_ns", "k_factor_db",
            "coherence_bw_mhz", "angular_spread_deg", "n_paths"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for band, b in out["bands"].items():
            for name, rec in b["materials"].items():
                w.writerow([band, b["frequency_hz"] / 1e9, name] +
                           [rec.get(c, "") for c in cols[3:]])


def _plot(out, traj):
    """Six-panel summary: RSS, delay spread, K-factor, reflectivity, route."""
    fig = plt.figure(figsize=(16.5, 9.5))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)
    bands = list(out["bands"])
    colors = {"5 GHz": "#2e86c1", "60 GHz": "#c0392b"}
    y = np.arange(len(SWEEP))
    labels = [get_material(m).label for m in SWEEP]

    def grouped(ax, key, xlabel, title, tight=False):
        """Horizontal grouped bars of one metric for both bands.

        ``tight`` clips the axis to the data range instead of anchoring it at
        zero.  For quantities like RSS, whose values sit tens of dB from zero
        but differ by a few dB, a zero-anchored axis hides the whole effect.
        """
        allv = []
        for i, band in enumerate(bands):
            vals = [out["bands"][band]["materials"][m][key] for m in SWEEP]
            allv += vals
            ax.barh(y + (i - 0.5) * 0.38, vals, height=0.36,
                    color=colors[band], label=band, alpha=0.9)
        if tight:
            lo, hi = min(allv), max(allv)
            pad = 0.08 * (hi - lo) if hi > lo else 1.0
            ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.25, axis="x")
        ax.legend(fontsize=8, frameon=False)

    grouped(fig.add_subplot(gs[0, 0]), "rss_mean_dbm",
            "route-mean RSS [dBm]", "Received power along the route", tight=True)
    grouped(fig.add_subplot(gs[0, 1]), "delay_spread_ns",
            "RMS delay spread [ns]", "Multipath richness")
    grouped(fig.add_subplot(gs[0, 2]), "k_factor_db",
            "Rice K-factor [dB]", "Dominance of the direct path")

    # reflectivity vs incidence angle at 5 GHz
    ax = fig.add_subplot(gs[1, 0])
    theta = np.linspace(0.0, 89.0, 180)
    cos_ti = torch.tensor(np.cos(np.deg2rad(theta)), dtype=torch.float64)
    cmap = plt.get_cmap("viridis")
    for i, m in enumerate(SWEEP):
        g = reflection_coefficient(5e9, cos_ti, get_material(m), "perp")
        ax.plot(theta, 20 * np.log10(g.abs().numpy() + 1e-12),
                color=cmap(i / max(1, len(SWEEP) - 1)), lw=1.5,
                label=get_material(m).label)
    ax.set_xlabel("incidence angle from normal [deg]")
    ax.set_ylabel(r"$20\log_{10}|\Gamma_\perp|$ [dB]")
    ax.set_title("Fresnel reflectivity at 5 GHz (Eq. 56)", fontsize=10.5)
    ax.set_ylim(-30, 1)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, ncol=2, frameon=False, loc="lower right")

    # RSS along the route for a few contrasting materials
    ax = fig.add_subplot(gs[1, 1])
    s = np.asarray(traj.arclength)
    for m, c in [("metal", "#7f8c8d"), ("concrete", "#2c3e50"),
                 ("plasterboard", "#e08a1e"), ("foam_board", "#27ae60")]:
        ax.plot(s, out["bands"]["5 GHz"]["materials"][m]["rss_profile_dbm"],
                lw=1.5, color=c, label=get_material(m).label)
    ax.set_xlabel("distance along route [m]")
    ax.set_ylabel("RSS [dBm]")
    ax.set_title("Band-averaged RSS along the robot route (5 GHz)", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    # penetration loss, log scale over many decades
    ax = fig.add_subplot(gs[1, 2])
    for i, band in enumerate(bands):
        vals = [min(out["bands"][band]["materials"][m]["penetration_loss_db_10cm"],
                    1e4) for m in SWEEP]
        ax.barh(y + (i - 0.5) * 0.38, vals, height=0.36, color=colors[band],
                label=band, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xlabel("one-way loss through 10 cm [dB], log scale")
    ax.set_title("Penetration loss (Eq. 58, 59)", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Experiment 2: indoor robot channel versus wall material "
                 "(RFDT, order 2 + diffraction)", fontsize=13)
    fig.savefig(os.path.join(RESULTS, "exp2_material_sweep.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()