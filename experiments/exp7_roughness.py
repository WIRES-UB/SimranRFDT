"""Experiment 7: surface roughness, and how much it actually changes.

The Fresnel coefficients of Eq. 56 are derived for an ideally smooth interface.
Real building surfaces are not smooth, and the resulting loss of coherent
specular energy is set by the surface height measured in wavelengths, so it is
invisible at 5 GHz and potentially severe at 60 GHz.  RFDT does not model it.
This experiment adds it and then measures, rather than assumes, what difference
it makes to every number this study has reported.

Three questions, in order:

  1. Where do the two roughness models differ?  The Rayleigh form assumes a
     small two-way phase variance; Miller-Brown adds back the energy a rough
     surface still returns near the specular direction.  They must agree while
     the roughness parameter is small and diverge only outside that regime,
     which is the stated reason the tracer defaults to Miller-Brown.

  2. How much does the correction move the reported route statistics?  This is
     the question that decides whether previous results need restating.

  3. How much of the answer rests on the surface heights themselves?  Those are
     order-of-magnitude literature estimates carrying roughly a factor-of-two
     uncertainty, unlike the permittivities, which come from ITU-R P.2040-1.
     If the estimate dominates the result, the honest conclusion is that the
     surface height should be fitted from data rather than tabulated, and the
     tracer supports exactly that.

A bias this experiment does not remove: energy taken out of the specular
direction is scattered, not absorbed, and nothing here puts it back, because
diffuse scattering is not modelled.  Every roughness-enabled number below is
therefore a lower bound on received power, by a knowable amount, and that is
stated rather than buried.

Outputs: ``results/exp7_roughness.{png,json}``
"""

from __future__ import annotations

import contextlib
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
from rfdt.materials import (C0, get_material, roughness_factor)       # noqa: E402
from rfdt.tracer import TracerConfig                                  # noqa: E402

from exp2_material_sweep import run_route                             # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

#: Materials carried through the route ablation, spanning the full roughness
#: range in the library: two effectively smooth, one intermediate, two rough.
ABLATION = ["metal", "foam_board", "plasterboard", "concrete", "brick"]

#: Materials whose surface height is swept over its uncertainty band.  One from
#: the pair this study compares and one genuinely rough surface.
SENSITIVITY = ["foam_board", "concrete"]

#: Multiplicative uncertainty on the tabulated surface heights.  They are
#: literature estimates, not measurements of the simulated surfaces, so a
#: factor of two either way is the honest span.
SIGMA_FACTORS = [0.5, 0.707, 1.0, 1.414, 2.0]

MODELS = ["none", "miller_brown", "rayleigh"]
BANDS = {"5 GHz": 5.0e9, "60 GHz": 60.0e9}
N_ROUTE = 48


@contextlib.contextmanager
def surface_height(name, sigma):
    """Temporarily override one material's surface height, then restore it.

    The material library holds shared objects, so a sweep has to put the
    tabulated value back afterwards or it would leak into whatever runs next.
    """
    mat = get_material(name)
    original = mat.roughness_sigma
    mat.roughness_sigma = sigma
    try:
        yield mat
    finally:
        mat.roughness_sigma = original


def model_curves():
    """Roughness factor against incidence angle and against frequency.

    Deterministic material physics with no scene involved, so this panel says
    exactly what the correction does before any propagation is layered on top.
    """
    cos_grid = torch.linspace(0.001, 1.0, 400, dtype=torch.float64)
    freq_grid = torch.logspace(np.log10(1e9), np.log10(1e11), 400,
                               dtype=torch.float64)
    out = {"cos_theta": cos_grid.tolist(),
           "frequency_hz": freq_grid.tolist(), "materials": {}}
    for name in ABLATION:
        sig = get_material(name).roughness_sigma
        out["materials"][name] = {
            "sigma_m": sig,
            "vs_angle_60ghz": {
                m: roughness_factor(60e9, cos_grid, sig, m).tolist()
                for m in ("rayleigh", "miller_brown")},
            "vs_frequency_normal": {
                m: roughness_factor(freq_grid,
                                    torch.ones_like(freq_grid), sig, m).tolist()
                for m in ("rayleigh", "miller_brown")},
            # the roughness parameter itself, which is what decides whether the
            # Rayleigh derivation still applies
            "g_at_60ghz_normal": float(2 * np.pi * 60e9 / C0 * sig),
        }
    return out


def route_ablation(traj):
    """Route statistics for each material under each roughness model.

    Same scene, same route, same transmitter; only the roughness model changes,
    so any difference is attributable to it alone.
    """
    out = {}
    for band, f_c in BANDS.items():
        out[band] = {}
        for name in ABLATION:
            out[band][name] = {}
            for model in MODELS:
                cfg = TracerConfig(max_order=2, weighting="rfdt",
                                   enable_diffraction=True,
                                   surface_roughness=model)
                r = run_route(name, f_c, traj, cfg)
                out[band][name][model] = {
                    k: r[k] for k in ("rss_mean_dbm", "delay_spread_ns",
                                      "k_factor_db", "angular_spread_deg")}
            base = out[band][name]["none"]["rss_mean_dbm"]
            for model in MODELS:
                out[band][name][model]["delta_db"] = (
                    out[band][name][model]["rss_mean_dbm"] - base)
            print(f"  {band:7s} {name:13s} "
                  + "  ".join(f"{m}={out[band][name][m]['delta_db']:+6.2f} dB"
                              for m in MODELS[1:]), flush=True)
    return out


def sigma_sensitivity(traj):
    """Route power against the assumed surface height, over its uncertainty.

    The permittivities come from ITU-R P.2040-1; the surface heights do not.
    This quantifies how much of the 60 GHz answer rides on the weaker of the
    two inputs.
    """
    out = {}
    for name in SENSITIVITY:
        nominal = get_material(name).roughness_sigma
        out[name] = {"nominal_sigma_m": nominal, "points": []}
        for fac in SIGMA_FACTORS:
            with surface_height(name, nominal * fac):
                cfg = TracerConfig(max_order=2, weighting="rfdt",
                                   enable_diffraction=True,
                                   surface_roughness="miller_brown")
                r = run_route(name, 60e9, traj, cfg)
            out[name]["points"].append({
                "factor": fac, "sigma_m": nominal * fac,
                "rss_mean_dbm": r["rss_mean_dbm"],
                "delay_spread_ns": r["delay_spread_ns"]})
            print(f"  sigma x{fac:5.3f}  {name:13s} "
                  f"{r['rss_mean_dbm']:7.2f} dBm", flush=True)
        rss = [p["rss_mean_dbm"] for p in out[name]["points"]]
        out[name]["spread_db"] = max(rss) - min(rss)
    return out


def main():
    """Run the three studies, then write the figure and the JSON record."""
    os.makedirs(RESULTS, exist_ok=True)
    traj = scenes.survey_trajectory(n_samples=N_ROUTE)
    t0 = time.time()

    print("model curves (no scene)")
    curves = model_curves()
    for name, rec in curves["materials"].items():
        print(f"  {name:13s} sigma={rec['sigma_m']*1000:5.2f} mm  "
              f"roughness parameter at 60 GHz, normal incidence "
              f"g={rec['g_at_60ghz_normal']:.3f}")

    print("\nroute ablation (change in route-mean power against no roughness)")
    ablation = route_ablation(traj)

    print("\nsensitivity to the assumed surface height, 60 GHz")
    sens = sigma_sensitivity(traj)

    out = {
        "setup": {
            "route_samples": N_ROUTE,
            "models": MODELS,
            "default_model": "miller_brown",
            "sigma_provenance": (
                "order-of-magnitude literature estimates, NOT ITU-R P.2040-1 "
                "and NOT measurements of the simulated surfaces"),
            "known_bias": (
                "energy removed from the specular direction is not "
                "re-radiated, since diffuse scattering is not modelled, so "
                "every roughness-enabled power here is a lower bound"),
        },
        "curves": curves,
        "route_ablation": ablation,
        "sigma_sensitivity": sens,
    }
    out["setup"]["runtime_s"] = time.time() - t0
    _plot(out)
    with open(os.path.join(RESULTS, "exp7_roughness.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote results/exp7_roughness.{{png,json}} "
          f"({out['setup']['runtime_s']:.0f} s)")


def _plot(out):
    """Four panels: the two models, the band dependence, the route effect, the
    sensitivity to the surface height itself."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.6))
    colours = {"metal": "#5d6d7e", "foam_board": "#27ae60",
               "plasterboard": "#2980b9", "concrete": "#c0392b",
               "brick": "#8e44ad"}

    # (a) the two models against incidence angle at 60 GHz
    ax = axes[0][0]
    cos = np.array(out["curves"]["cos_theta"])
    for name, rec in out["curves"]["materials"].items():
        ax.plot(cos, 20 * np.log10(np.array(rec["vs_angle_60ghz"]["miller_brown"])),
                color=colours[name], lw=1.8,
                label=f"{name.replace('_', ' ')} ({rec['sigma_m']*1000:.2f} mm)")
        ax.plot(cos, 20 * np.log10(np.array(rec["vs_angle_60ghz"]["rayleigh"])),
                color=colours[name], lw=1.1, ls=":")
    ax.set_xlabel("cos(incidence angle), 1.0 is normal incidence")
    ax.set_ylabel("specular loss [dB]")
    ax.set_title("(a) 60 GHz: Miller-Brown (solid) against Rayleigh (dotted).\n"
                 "They separate only where the Rayleigh derivation fails",
                 fontsize=10.5)
    ax.set_ylim(-45, 2)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")

    # (b) band dependence at normal incidence
    ax = axes[0][1]
    f = np.array(out["curves"]["frequency_hz"]) / 1e9
    for name, rec in out["curves"]["materials"].items():
        ax.plot(f, 20 * np.log10(np.array(rec["vs_frequency_normal"]["miller_brown"])),
                color=colours[name], lw=1.8, label=name.replace("_", " "))
    for band, style in ((5.0, "-."), (60.0, "--")):
        ax.axvline(band, color="0.35", ls=style, lw=1.1)
        ax.text(band, 1.5, f"{band:.0f} GHz", ha="center", fontsize=8, color="0.3")
    ax.set_xscale("log")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel("specular loss at normal incidence [dB]")
    ax.set_title("(b) Why this is a millimetre-wave correction: under 0.2 dB\n"
                 "at 5 GHz for ordinary surfaces, 0.75 dB for the roughest",
                 fontsize=10.5)
    ax.set_ylim(-45, 4)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8, loc="lower left")

    # (c) effect on route-mean power
    ax = axes[1][0]
    names = ABLATION
    x = np.arange(len(names))
    w = 0.35
    for i, band in enumerate(BANDS):
        d = [out["route_ablation"][band][n]["miller_brown"]["delta_db"]
             for n in names]
        ax.bar(x + (i - 0.5) * w, d, w, label=band,
               color="#3d5a80" if i == 0 else "#ee6c4d")
        for xi, di in zip(x + (i - 0.5) * w, d):
            # The 5 GHz bars are far too small to see, which is the finding
            # rather than a plotting defect, so they are labelled above the
            # axis instead of underneath where they would sit on top of the
            # 60 GHz labels.
            if abs(di) < 0.05:
                ax.text(xi, 0.12, f"{di:+.2f}", ha="center", va="bottom",
                        fontsize=7.5, rotation=90, color="#3d5a80")
            else:
                ax.text(xi, di - 0.15, f"{di:+.2f}", ha="center", va="top",
                        fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=9)
    ax.set_ylabel("change in route-mean power [dB]")
    ax.axhline(0, color="0.3", lw=0.8)
    lowest = min(out["route_ablation"]["60 GHz"][n]["miller_brown"]["delta_db"]
                 for n in names)
    ax.set_ylim(lowest - 0.75, 1.05)
    ax.set_title("(c) Effect on the reported route statistics. Metal and foam\n"
                 "are both smooth, so the study's headline pair barely moves",
                 fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=9)

    # (d) sensitivity to the surface height estimate
    ax = axes[1][1]
    for name, rec in out["sigma_sensitivity"].items():
        fac = [p["factor"] for p in rec["points"]]
        rss = [p["rss_mean_dbm"] for p in rec["points"]]
        ax.plot(fac, rss, "o-", color=colours[name], lw=1.8,
                label=f"{name.replace('_', ' ')} "
                      f"(spread {rec['spread_db']:.1f} dB)")
    ax.axvline(1.0, color="0.35", ls="--", lw=1.0)
    ax.annotate("tabulated estimate", xy=(1.0, 0.97), xycoords=("data", "axes fraction"),
                fontsize=8, color="0.3", ha="left", va="top",
                xytext=(4, 0), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xticks(SIGMA_FACTORS)
    ax.set_xticklabels([f"x{v:g}" for v in SIGMA_FACTORS])
    # suppress the decade minor labels the log locator adds underneath
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("assumed surface height, relative to the tabulated estimate")
    ax.set_ylabel("route-mean power at 60 GHz [dBm]")
    ax.set_title("(d) How much rides on the surface height itself.\n"
                 "The estimates carry a factor-of-two uncertainty",
                 fontsize=10.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8.5)

    fig.suptitle("Experiment 7: surface roughness, an addition to RFDT, "
                 "and its measured effect", fontsize=13.5)
    # titles here are two lines tall, so the default spacing lets the upper
    # panels' x-labels land on the lower panels' titles
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.subplots_adjust(hspace=0.42, wspace=0.22)
    fig.savefig(os.path.join(RESULTS, "exp7_roughness.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
