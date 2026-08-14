"""Experiment 3: the robot's own radar, and what materials do to it.

Experiment 2 looked at the communication link from a fixed access point.  This
one puts the sensor *on the robot*: a 77 GHz FMCW MIMO radar (the TI AWR1843
cascade configuration of App. C.1) that the robot uses to perceive its
surroundings while moving.  Four studies:

A. **Material identification from the echo.**  The robot faces a wall and the
   wall's material is swept.  The range-profile peak tells us how detectable
   each material is, and how far the echo sits above the receiver noise floor.
   This is the quantity that decides whether a robot can map a surface at all.

B. **Detection through an obstacle (NLOS).**  A thin board is placed between
   the radar and a metal target, reproducing the plastic / paper / foam
   occlusion study of Fig. 15(a).  The two-way transmission loss through the
   board sets how much of the target survives.

C. **Doppler from motion.**  The robot drives towards the wall, so every path
   acquires the Doppler shift of App. E.3.  The range-Doppler map shows the
   echo displaced in velocity, which is what lets a moving robot separate
   static structure from moving objects.

D. **Range resolution against material contrast.**  Two closely spaced
   surfaces of different materials are resolved or not depending on their
   relative echo strength, which sets a practical limit on mapping.

Outputs: ``results/exp3_robot_radar.{png,json,csv}``
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                     # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

from rfdt import scenes                                               # noqa: E402
from rfdt.antennas import mmwave_radar                                # noqa: E402
from rfdt.geometry import box, plate                                  # noqa: E402
from rfdt.materials import C0, get_material                           # noqa: E402
from rfdt.signal import (FMCWConfig, range_doppler_map,               # noqa: E402
                         range_profile_fft)
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

#: Wall materials probed by the robot's radar in study A.
WALL_MATERIALS = ["metal", "concrete", "marble", "brick", "glass", "plasterboard",
                  "chipboard", "wood", "ceiling_board", "human_body", "foam_board"]

#: Obstacle boards of study B, matching the three used in Fig. 15(a).
BOARDS = ["plastic_board", "paper_board", "foam_board"]

RADAR_POS = (0.0, 0.0, 0.9)
WALL_RANGE = 3.0            # distance from radar to wall [m]
TARGET_CENTER = (2.5, 0.0, 0.9)
TARGET_SIZE = (0.6, 0.6, 0.6)


def radar_pair(position=RADAR_POS, boresight=(1.0, 0.0, 0.0), frequency=77e9):
    """Monostatic radar with a small Tx/Rx offset.

    A truly co-located pair would give a zero-length direct path and a
    singular free-space factor, so the two antennas sit 2 cm apart, which is
    also physically what a real board does.  The direct coupling path is
    dropped in processing, exactly as a real radar removes it by calibration.
    """
    tx, rx = mmwave_radar(position, boresight, frequency)
    rx.position = rx.position + torch.tensor([0.0, 0.02, 0.0], dtype=torch.float64)
    return tx, rx


def echo_paths(mesh, tx, rx, cfg, rx_positions=None):
    """Trace and drop the direct Tx/Rx coupling, keeping only scene echoes."""
    paths = RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=rx_positions)
    keep = [i for i, k in enumerate(paths.kind) if k != "los"]
    return paths.select(keep)


def wall_scene(material, distance=WALL_RANGE):
    """A single large wall at ``distance`` metres, normal facing the radar.

    ``flip=True`` is essential: the radar sits at x = 0, so the wall's normal
    must point along -x.  A facet only reflects on the side its normal faces.
    """
    return plate((distance, 0.0, 0.9), (4.0, 3.0), "x", material, "wall",
                 flip=True).weld()


def study_a(cfg, fmcw):
    """Echo strength versus wall material (detectability of a surface)."""
    tx, rx = radar_pair()
    noise_dbm = rx.noise_floor_dbm
    out = {}
    for name in WALL_MATERIALS:
        mesh = wall_scene(name)
        p = echo_paths(mesh, tx, rx, cfg)
        prof = range_profile_fft(p.delay, p.gain, fmcw).abs().squeeze()
        pk = int(prof.argmax())
        # coherently combined two-way echo power, the link-budget quantity
        echo_dbm = float(tx.power_dbm + 10 * np.log10(
            float(p.field().abs() ** 2) + 1e-30))
        mat = get_material(name)
        out[name] = {
            "peak_range_m": float(fmcw.range_axis()[pk]),
            "range_error_m": float(fmcw.range_axis()[pk]) - WALL_RANGE,
            "echo_dbm": echo_dbm,
            "echo_snr_db": echo_dbm - noise_dbm,
            "profile_db": (20 * torch.log10(prof / prof.max().clamp_min(1e-30)
                                            + 1e-12)).tolist(),
            "reflection_db": float(20 * np.log10(abs(complex(
                __import__("rfdt.materials", fromlist=["x"]).reflection_coefficient(
                    77e9, torch.tensor(1.0, dtype=torch.float64), mat, "perp"))))),
            "in_validity_range": mat.in_validity_range(77e9),
            "source": mat.source,
        }
    return out, noise_dbm


def study_b(cfg, fmcw):
    """Detection of a metal target through a thin board of each material."""
    tx, rx = radar_pair()
    noise_dbm = rx.noise_floor_dbm
    out = {}
    ref = None
    for board in [None] + BOARDS:
        if board is None:
            mesh = box(TARGET_CENTER, TARGET_SIZE, "metal", "target").weld()
        else:
            mesh = scenes.obstacle_scene(board_material=board, board_x=1.2,
                                         target_center=TARGET_CENTER,
                                         target_size=TARGET_SIZE)
        p = echo_paths(mesh, tx, rx, cfg)
        # isolate the target return: the echo comes off the target's *front
        # face*, not its centre, so the round trip is to that face
        front = TARGET_CENTER[0] - TARGET_SIZE[0] / 2.0
        rt = 2.0 * (front - RADAR_POS[0])
        sel = [i for i in range(p.n_paths())
               if abs(float(p.length[0, i]) - rt) < 0.30]
        amp = p.gain[0, sel].sum() if sel else torch.zeros((), dtype=torch.complex128)
        dbm = tx.power_dbm + 10 * np.log10(float(amp.abs() ** 2) + 1e-30)
        if ref is None:
            ref = dbm
        mat = get_material(board) if board else None
        out[board or "no board"] = {
            "target_echo_dbm": dbm,
            "target_snr_db": dbm - noise_dbm,
            "excess_loss_db": dbm - ref,
            "board_thickness_m": mat.thickness if mat else 0.0,
            "two_way_slab_loss_db": (
                float(2 * mat.penetration_loss_db(77e9, mat.thickness))
                if mat else 0.0),
            "source": mat.source if mat else "",
        }
    return out


def study_c(cfg, fmcw, speed=1.0):
    """Range-Doppler map for a robot driving towards a concrete wall."""
    tx, rx = radar_pair()
    mesh = wall_scene("concrete")
    p = echo_paths(mesh, tx, rx, cfg)
    # both antennas ride on the robot, so both move: for a monostatic radar
    # closing at v the shift is 2v/lambda, not v/lambda
    v_robot = torch.tensor([[speed, 0.0, 0.0]], dtype=torch.float64)
    dop = RFDTracer.doppler(p, C0 / tx.frequency, v_tx=v_robot, v_rx=v_robot)
    rd = range_doppler_map(p.delay, p.gain, dop, fmcw).abs().squeeze()
    rd_db = 20 * torch.log10(rd / rd.max().clamp_min(1e-30) + 1e-12)
    # expected shift for a monostatic radar closing at `speed`
    expected = 2.0 * speed / (C0 / tx.frequency)
    v_axis = fmcw.velocity_axis()
    peak = np.unravel_index(int(rd_db.argmax()), rd_db.shape)
    return {
        "speed_m_s": speed,
        "expected_doppler_hz": float(expected),
        "measured_doppler_hz": float(dop.abs().max()),
        "peak_velocity_m_s": float(v_axis[peak[0]]),
        "peak_range_m": float(fmcw.range_axis()[peak[1]]),
        "map_db": rd_db.numpy(),
    }


def study_d(cfg, fmcw, gap=0.25):
    """Two surfaces one range cell apart, one metal and one dielectric.

    Whether the weaker surface survives next to the stronger one is set by
    their reflectivity contrast and by the sidelobes of the range transform.
    """
    tx, rx = radar_pair()
    out = {}
    for second in ["metal", "concrete", "wood", "foam_board"]:
        # a partially transparent front sheet, then a second surface behind it
        mesh = plate((WALL_RANGE, 0.0, 0.9), (4.0, 3.0), "x", "plasterboard",
                     "wall1", flip=True)
        mesh = mesh.merged(plate((WALL_RANGE + gap, 0.0, 0.9), (4.0, 3.0), "x",
                                 second, "wall2", flip=True)).weld()
        p = echo_paths(mesh, tx, rx, cfg)
        prof = range_profile_fft(p.delay, p.gain, fmcw).abs().squeeze()
        prof_db = 20 * torch.log10(prof / prof.max().clamp_min(1e-30) + 1e-12)
        axis = fmcw.range_axis().numpy()
        near = np.abs(axis - WALL_RANGE) < fmcw.range_resolution
        far = np.abs(axis - (WALL_RANGE + gap)) < fmcw.range_resolution
        out[second] = {
            "second_material": second,
            "front_peak_db": float(prof_db.numpy()[near].max()),
            "second_peak_db": float(prof_db.numpy()[far].max()),
            "contrast_db": float(prof_db.numpy()[far].max()
                                 - prof_db.numpy()[near].max()),
            "profile_db": prof_db.tolist(),
        }
    return out, fmcw.range_axis().numpy()


def main():
    """Run the four radar studies and write figure, JSON and CSV."""
    os.makedirs(RESULTS, exist_ok=True)
    fmcw = FMCWConfig()
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=True)

    print("Experiment 3: robot-mounted 77 GHz FMCW radar")
    print(f"  bandwidth {fmcw.bandwidth/1e9:.2f} GHz, range resolution "
          f"{fmcw.range_resolution*100:.1f} cm, {fmcw.n_chirps} chirps")

    a, noise_dbm = study_a(cfg, fmcw)
    print(f"\nA. wall echo vs material (wall at {WALL_RANGE} m, "
          f"noise floor {noise_dbm:.1f} dBm)")
    print(f"{'material':15s} {'|G|[dB]':>8s} {'echo[dBm]':>10s} {'SNR[dB]':>8s} "
          f"{'range[m]':>9s} {'err[cm]':>8s}")
    for k, v in a.items():
        flag = "" if v["in_validity_range"] else "  (*)"
        print(f"{k:15s} {v['reflection_db']:8.2f} {v['echo_dbm']:10.2f} "
              f"{v['echo_snr_db']:8.1f} {v['peak_range_m']:9.3f} "
              f"{v['range_error_m']*100:8.2f}{flag}")

    b = study_b(cfg, fmcw)
    print("\nB. metal target seen through a board (Fig. 15a analogue)")
    print(f"{'board':16s} {'thick[cm]':>10s} {'echo[dBm]':>10s} {'SNR[dB]':>8s} "
          f"{'excess[dB]':>11s} {'2-way slab[dB]':>15s}")
    for k, v in b.items():
        print(f"{k:16s} {v['board_thickness_m']*100:10.1f} "
              f"{v['target_echo_dbm']:10.2f} {v['target_snr_db']:8.1f} "
              f"{v['excess_loss_db']:11.2f} {v['two_way_slab_loss_db']:15.2f}")

    c = study_c(cfg, fmcw)
    print(f"\nC. Doppler at {c['speed_m_s']} m/s closing speed: expected "
          f"{c['expected_doppler_hz']/1e3:.2f} kHz, simulated "
          f"{c['measured_doppler_hz']/1e3:.2f} kHz, "
          f"map peak at {c['peak_velocity_m_s']:.2f} m/s / "
          f"{c['peak_range_m']:.2f} m")

    d, axis = study_d(cfg, fmcw)
    print("\nD. second surface 25 cm behind a plasterboard sheet")
    print(f"  {'material':14s} {'front[dB]':>10s} {'second[dB]':>11s} "
          f"{'contrast[dB]':>13s}")
    for k, v in d.items():
        print(f"  {k:14s} {v['front_peak_db']:10.2f} {v['second_peak_db']:11.2f} "
              f"{v['contrast_db']:13.2f}")

    _plot(a, b, c, d, axis, fmcw, noise_dbm)
    payload = {"config": {"f_c": fmcw.f_c, "bandwidth_hz": fmcw.bandwidth,
                          "range_resolution_m": fmcw.range_resolution,
                          "noise_floor_dbm": noise_dbm,
                          "wall_range_m": WALL_RANGE},
               "A_wall_material": a, "B_nlos_board": b,
               "C_doppler": {k: v for k, v in c.items() if k != "map_db"},
               "D_resolution": {k: {kk: vv for kk, vv in v.items()
                                    if kk != "profile_db"} for k, v in d.items()}}
    with open(os.path.join(RESULTS, "exp3_robot_radar.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(RESULTS, "exp3_robot_radar.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["study", "case", "metric", "value"])
        for k, v in a.items():
            for m in ["reflection_db", "echo_dbm", "echo_snr_db", "peak_range_m"]:
                w.writerow(["A_wall_material", k, m, v[m]])
        for k, v in b.items():
            for m in ["target_echo_dbm", "target_snr_db", "excess_loss_db"]:
                w.writerow(["B_nlos_board", k, m, v[m]])
    print("\n(*) evaluated outside the ITU-R P.2040-1 validity range for that "
          "material.\nwrote results/exp3_robot_radar.{png,json,csv}")
    return payload


def _plot(a, b, c, d, axis, fmcw, noise_dbm):
    """Four-panel figure summarising the radar studies."""
    fig = plt.figure(figsize=(16.5, 9.0))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)

    # A: echo SNR by material, plus a few range profiles
    ax = fig.add_subplot(gs[0, 0])
    names = list(a)
    snr = [a[n]["echo_snr_db"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, snr, color=["#7f8c8d" if a[n]["in_validity_range"] else "#b7950b"
                           for n in names])
    ax.set_yticks(y)
    ax.set_yticklabels([get_material(n).label for n in names], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("echo SNR [dB]")
    ax.set_title(f"A. Wall echo vs material, wall at {WALL_RANGE} m\n"
                 f"(77 GHz, {fmcw.bandwidth/1e9:.1f} GHz BW, "
                 f"noise floor {noise_dbm:.0f} dBm)", fontsize=10.5)
    ax.grid(alpha=0.25, axis="x")

    ax = fig.add_subplot(gs[0, 1])
    for n, col in [("metal", "#7f8c8d"), ("concrete", "#2c3e50"),
                   ("wood", "#a0522d"), ("foam_board", "#27ae60")]:
        ax.plot(fmcw.range_axis().numpy(), a[n]["profile_db"], lw=1.4,
                color=col, label=get_material(n).label)
    ax.set_xlim(0, 6)
    ax.set_ylim(-70, 3)
    ax.axvline(WALL_RANGE, color="0.5", ls=":", lw=1.2)
    ax.set_xlabel("range [m]")
    ax.set_ylabel("normalised range profile [dB]")
    ax.set_title("A. Range profiles (each normalised to its own peak)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    # B: NLOS boards
    ax = fig.add_subplot(gs[1, 0])
    keys = list(b)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, [b[k]["target_snr_db"] for k in keys], width=0.4,
           color="#2e86c1", label="target SNR")
    ax.bar(x + 0.2, [-b[k]["excess_loss_db"] for k in keys], width=0.4,
           color="#c0392b", label="excess two-way loss")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    ax.set_ylabel("dB")
    ax.set_title("B. Metal target seen through a board (NLOS)", fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, frameon=False)

    # C: range-Doppler map
    ax = fig.add_subplot(gs[1, 1])
    m = c["map_db"]
    v = fmcw.velocity_axis().numpy()
    r = fmcw.range_axis().numpy()
    keep = r < 6.0
    im = ax.pcolormesh(r[keep], v, m[:, keep], cmap="magma", vmin=-60, vmax=0,
                       shading="auto")
    ax.axhline(0.0, color="w", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("range [m]")
    ax.set_ylabel("radial velocity [m/s]")
    ax.set_title(f"C. Range-Doppler, robot closing at {c['speed_m_s']} m/s\n"
                 f"(concrete wall; peak at {c['peak_velocity_m_s']:.2f} m/s)",
                 fontsize=10.5)
    fig.colorbar(im, ax=ax, label="normalised [dB]")

    fig.suptitle("Experiment 3: robot-mounted FMCW radar versus material",
                 fontsize=13)
    fig.savefig(os.path.join(RESULTS, "exp3_robot_radar.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()