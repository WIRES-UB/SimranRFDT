"""Experiment 8: stratified walls, and the oblique-incidence phase they fixed.

A real interior partition is not a homogeneous block.  A stud wall is
plasterboard, an air cavity, then plasterboard; a window is glass, a gap, then
glass.  Each internal boundary reflects, and those reflections interfere, so
the wall has a structure in frequency and angle that no single effective
permittivity can reproduce.  Replacing the stack with one averaged slab does
not blur that structure, it deletes it.

This experiment replaces the single-slab model with the exact transfer-matrix
solution for a stratified medium and reports three things.

  1. What the layering itself is worth: how far a stud partition departs from
     the equivalent solid slab, in frequency and in angle.

  2. A defect the replacement exposed in the previous single-slab formula.  It
     took the phase across the slab along the slanted ray path,
     d / cos(theta_t), where the quantity that actually interferes is the
     normal component, d * cos(theta_t).  The two agree at normal incidence and
     diverge as the square of the cosine, so nothing at normal incidence could
     have caught it, and it moved the Fabry-Perot resonances the wrong way with
     angle.  Panel (c) is the before and after.

  3. Whether any of it reaches the robot.  A wall's transmission matters less
     indoors than its reflection, and the transfer matrix changes both, so the
     route is run with layered and with homogeneous walls.

Outputs: ``results/exp8_layered_walls.{png,json}``
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
from rfdt.materials import (C0, Layer, fresnel, get_material,          # noqa: E402
                            multilayer_coefficients, slab_transmission)
from rfdt.tracer import TracerConfig                                  # noqa: E402

from exp2_material_sweep import run_route                             # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

#: The stacks compared against their homogeneous equivalents.  In each case the
#: equivalent is the stack's dominant solid material at the stack's own total
#: thickness, which is the substitution a single-slab model forces on you.
PAIRS = [("drywall_partition", "plasterboard", "#c0392b"),
         ("double_glazed_window", "glass", "#2980b9")]

#: Materials used for the oblique-phase comparison of panel (c): the ones a
#: ray actually transmits through in these scenes.
OBLIQUE = ["plasterboard", "wood", "glass", "concrete"]

FREQ = 5.0e9
BAND = np.linspace(1e9, 12e9, 1101)
ANGLES = np.linspace(0.0, 88.0, 441)
N_ROUTE = 48


def previous_slab_formula(f_hz, cos_ti, mat, thickness):
    """The closed form this experiment replaced, kept only to measure it.

    Identical to the transfer matrix at normal incidence and wrong away from
    it, because the phase across the layer is taken along the slanted ray path
    rather than normal to the faces.  Reproduced here rather than described so
    that panel (c) compares against the real thing.
    """
    ct = torch.as_tensor(cos_ti, dtype=torch.float64)
    fr = fresnel(f_hz, ct, mat)
    g, t_in, cos_tt = fr["gamma_perp"], fr["tau_perp"], fr["cos_tt"]
    phi = mat.propagation_constant(f_hz) * torch.as_tensor(
        thickness, dtype=torch.float64) / cos_tt
    single = t_in * (1.0 - g) * torch.exp(-phi)
    return single / (1.0 - g ** 2 * torch.exp(-2.0 * phi))


def stack_against_slab():
    """Transmission and reflection of each stack beside its solid equivalent."""
    out = {}
    one = torch.tensor(1.0, dtype=torch.float64)
    for stack_name, solid_name, _ in PAIRS:
        stack = get_material(stack_name)
        solid = get_material(solid_name)
        rec = {"total_thickness_m": stack.thickness,
               "layers": [(lay.material.name, lay.thickness)
                          for lay in stack.layers],
               "equivalent": solid_name}
        # against frequency, normal incidence
        rec["frequency_hz"] = BAND.tolist()
        rec["stack_t_db"] = [float(20 * np.log10(abs(complex(
            multilayer_coefficients(float(f), one, stack.layers)["tau"])) + 1e-30))
            for f in BAND]
        rec["stack_r_db"] = [float(20 * np.log10(abs(complex(
            multilayer_coefficients(float(f), one, stack.layers)["gamma"])) + 1e-30))
            for f in BAND]
        rec["slab_t_db"] = [float(20 * np.log10(abs(complex(
            slab_transmission(float(f), one, solid,
                              thickness=stack.thickness))) + 1e-30))
            for f in BAND]
        # against angle, at the carrier
        rec["angles_deg"] = ANGLES.tolist()
        cts = [torch.tensor(float(np.cos(np.radians(a))), dtype=torch.float64)
               for a in ANGLES]
        rec["stack_r_db_vs_angle"] = [float(20 * np.log10(abs(complex(
            multilayer_coefficients(FREQ, c, stack.layers)["gamma"])) + 1e-30))
            for c in cts]
        rec["slab_r_db_vs_angle"] = [float(20 * np.log10(abs(complex(
            multilayer_coefficients(FREQ, c, [Layer(solid, stack.thickness)])["gamma"]
        )) + 1e-30)) for c in cts]
        span = np.array(rec["stack_t_db"]) - np.array(rec["slab_t_db"])
        rec["max_transmission_gap_db"] = float(np.max(np.abs(span)))
        rec["rms_transmission_gap_db"] = float(np.sqrt(np.mean(span ** 2)))
        out[stack_name] = rec
        print(f"  {stack_name:22s} against a solid {solid_name}: "
              f"transmission differs by up to "
              f"{rec['max_transmission_gap_db']:.1f} dB "
              f"(rms {rec['rms_transmission_gap_db']:.1f} dB)", flush=True)
    return out


def oblique_phase_correction():
    """The previous formula against the transfer matrix, versus angle."""
    out = {"angles_deg": ANGLES.tolist(), "materials": {}}
    for name in OBLIQUE:
        mat = get_material(name)
        old, new, dphase = [], [], []
        for a in ANGLES:
            ct = torch.tensor(float(np.cos(np.radians(a))), dtype=torch.float64)
            o = complex(previous_slab_formula(FREQ, ct, mat, mat.thickness))
            n = complex(slab_transmission(FREQ, ct, mat, thickness=mat.thickness))
            old.append(float(20 * np.log10(abs(o) + 1e-30)))
            new.append(float(20 * np.log10(abs(n) + 1e-30)))
            dphase.append(float(np.degrees(np.angle(n / o))))
        out["materials"][name] = {
            "thickness_m": mat.thickness, "old_db": old, "new_db": new,
            "phase_change_deg": dphase,
            "max_magnitude_change_db": float(np.max(np.abs(
                np.array(new) - np.array(old)))),
            "max_phase_change_deg": float(np.max(np.abs(dphase))),
        }
        r = out["materials"][name]
        print(f"  {name:14s} ({mat.thickness*100:5.1f} cm): up to "
              f"{r['max_magnitude_change_db']:6.2f} dB and "
              f"{r['max_phase_change_deg']:6.1f} deg of phase", flush=True)
    return out


def route_effect(traj):
    """Route statistics with layered walls against homogeneous ones.

    Indoors a wall's reflection matters more than its transmission, and the
    transfer matrix changes both, so this is where the layering either reaches
    the robot or does not.

    The comparison is against a solid wall of the *same total thickness*, not
    against the library's 12.7 mm plasterboard.  Comparing a 115 mm stack with
    a 13 mm board would vary thickness and layering together and then credit
    the whole difference to layering.  Registering the equivalent is the only
    way to hold thickness fixed, since the scene builder looks materials up by
    name.

    Worth stating plainly: for a homogeneous wall this simulator's reflection
    is the single-interface Fresnel value and does not depend on thickness at
    all, so the solid reference is the same wall the earlier experiments used.
    The stack's reflection depends on the entire structure.  That difference is
    the mechanism under test here.
    """
    from rfdt.materials import MATERIALS, Material
    stack = get_material("drywall_partition")
    base = get_material("plasterboard")
    equiv_name = "plasterboard_same_thickness"
    MATERIALS[equiv_name] = Material(
        equiv_name, base.a, base.b, base.c, base.d,
        thickness=stack.thickness, f_range=base.f_range,
        label=f"Solid plasterboard, {stack.thickness*1000:.1f} mm",
        roughness_sigma=base.roughness_sigma)

    cfg = TracerConfig(max_order=2, weighting="rfdt", enable_diffraction=True)
    out = {}
    for name in ("drywall_partition", equiv_name):
        out[name] = run_route(name, FREQ, traj, cfg)
        r = out[name]
        print(f"  walls of {name:28s} RSS {r['rss_mean_dbm']:7.2f} dBm  "
              f"tau {r['delay_spread_ns']:5.2f} ns  K {r['k_factor_db']:6.2f} dB",
              flush=True)
    out["_equivalent_name"] = equiv_name
    out["_total_thickness_m"] = stack.thickness
    return out


def main():
    """Run the three studies, then write the figure and the JSON record."""
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    print("stacks against their homogeneous equivalents")
    stacks = stack_against_slab()
    print("\noblique-incidence phase: previous closed form against the "
          "transfer matrix")
    oblique = oblique_phase_correction()
    print("\neffect on the robot route at 5 GHz")
    traj = scenes.survey_trajectory(n_samples=N_ROUTE)
    route = route_effect(traj)

    out = {"setup": {"frequency_hz": FREQ, "route_samples": N_ROUTE,
                     "method": "Abeles characteristic matrix, exact for a "
                               "stratified medium",
                     "known_limitation": (
                         "layering applies to plate facets.  A closed solid "
                         "still uses the two-interface volume model, so the "
                         "partition box in the furnished room is homogeneous "
                         "however its material is defined")},
           "stacks": stacks, "oblique_phase": oblique, "route": route}
    out["setup"]["runtime_s"] = time.time() - t0
    _plot(out)
    with open(os.path.join(RESULTS, "exp8_layered_walls.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/exp8_layered_walls.{{png,json}} "
          f"({out['setup']['runtime_s']:.0f} s)")


def _plot(out):
    """Four panels: the stack in frequency, the stack in angle, the phase
    correction, and whether any of it reaches the robot."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))

    # (a) frequency response of a stack against its solid equivalent
    ax = axes[0][0]
    for stack_name, solid_name, colour in PAIRS:
        rec = out["stacks"][stack_name]
        f = np.array(rec["frequency_hz"]) / 1e9
        ax.plot(f, rec["stack_t_db"], color=colour, lw=1.7,
                label=f"{stack_name.replace('_', ' ')} (layered)")
        ax.plot(f, rec["slab_t_db"], color=colour, lw=1.2, ls="--", alpha=0.8,
                label=f"solid {solid_name}, same total thickness")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("transmission [dB]")
    ax.set_title("(a) A stack is not a slab. The cavity resonances are the\n"
                 "structure a single effective permittivity deletes",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    # (b) reflection against angle
    ax = axes[0][1]
    for stack_name, solid_name, colour in PAIRS:
        rec = out["stacks"][stack_name]
        a = np.array(rec["angles_deg"])
        ax.plot(a, rec["stack_r_db_vs_angle"], color=colour, lw=1.7,
                label=f"{stack_name.replace('_', ' ')} (layered)")
        ax.plot(a, rec["slab_r_db_vs_angle"], color=colour, lw=1.2, ls="--",
                alpha=0.8, label=f"solid {solid_name}, same thickness")
    ax.set_xlabel("incidence angle [degrees]")
    ax.set_ylabel("reflection [dB]")
    ax.set_title(f"(b) Reflection at {FREQ/1e9:.0f} GHz. Indoors this matters\n"
                 "more than transmission, and it changes too", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # (c) the oblique-incidence phase defect
    ax = axes[1][0]
    colours = {"plasterboard": "#2980b9", "wood": "#27ae60",
               "glass": "#8e44ad", "concrete": "#c0392b"}
    a = np.array(out["oblique_phase"]["angles_deg"])
    for name, rec in out["oblique_phase"]["materials"].items():
        ax.plot(a, np.array(rec["new_db"]) - np.array(rec["old_db"]),
                color=colours[name], lw=1.7,
                label=f"{name} ({rec['thickness_m']*100:.1f} cm), "
                      f"up to {rec['max_magnitude_change_db']:.1f} dB")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xlabel("incidence angle [degrees]")
    ax.set_ylabel("correction to transmission [dB]")
    ax.set_title("(c) The defect this exposed: the previous formula took the\n"
                 "phase along the ray path, not normal to the faces",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower left")

    # (d) does any of it reach the robot
    # Plotted as the change rather than the raw values: received power in dBm
    # is a large negative number and would flatten nanoseconds and degrees to
    # invisibility on a shared axis.  The absolute pair is printed on each bar
    # so nothing is hidden by the choice.
    ax = axes[1][1]
    equiv = out["route"]["_equivalent_name"]
    thick_mm = out["route"]["_total_thickness_m"] * 1000.0
    keys = [("rss_mean_dbm", "route-mean\npower [dB]"),
            ("delay_spread_ns", "delay spread\n[ns]"),
            ("k_factor_db", "Rice K-factor\n[dB]"),
            ("angular_spread_deg", "angular spread\n[deg]")]
    lay = out["route"]["drywall_partition"]
    sol = out["route"][equiv]
    deltas = [lay[k] - sol[k] for k, _ in keys]
    x = np.arange(len(keys))
    ax.bar(x, deltas, 0.55,
           color=["#c0392b" if d >= 0 else "#2980b9" for d in deltas])
    for xi, (k, _), d in zip(x, keys, deltas):
        top = d >= 0
        ax.text(xi, d + (0.35 if top else -0.35), f"{d:+.2f}",
                ha="center", va="bottom" if top else "top",
                fontsize=9, fontweight="bold")
        ax.text(xi, 0.0, f"\n{sol[k]:.1f} to {lay[k]:.1f}", ha="center",
                va="top" if top else "bottom", fontsize=7.5, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels([lbl for _, lbl in keys], fontsize=8.5)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_ylabel("change from solid to layered wall")
    lo, hi = min(deltas), max(deltas)
    ax.set_ylim(lo - 0.25 * (hi - lo) - 1.2, hi + 0.25 * (hi - lo) + 1.2)
    ax.set_title(f"(d) Whether it reaches the robot: same room, same {thick_mm:.0f} mm\n"
                 "of wall, layered against solid. Grey is solid to layered",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Experiment 8: stratified walls by transfer matrix, "
                 "and the oblique-incidence phase they exposed", fontsize=13.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.subplots_adjust(hspace=0.42, wspace=0.22)
    fig.savefig(os.path.join(RESULTS, "exp8_layered_walls.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
