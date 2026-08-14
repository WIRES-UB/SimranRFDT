"""Experiment 4: solving the inverse problem (RFDT Sec. 5.2).

Experiments 2 and 3 run the simulator forwards.  This one runs it *backwards*,
which is what the paper is actually about: given RF measurements, recover the
physical parameters of the scene by gradient descent through the simulator
(Eq. 23), updated with regularised Adam (Eq. 24).

Two studies:

A. **Material recovery from a robot's RSS survey.**  A ceiling access point
   transmits while the robot drives its route; the recorded RSS is the only
   observation.  Starting from a deliberately wrong material guess, the
   optimiser recovers the wall's permittivity and conductivity.  This is the
   digital-twin loop of Fig. 5 in miniature: the parameters are explicit and
   physically meaningful, so the result is interpretable rather than a set of
   opaque network weights (Fig. 6).

B. **Joint geometry and material recovery from radar, with and without the
   surrogate.**  The robot's radar observes a wall at an unknown distance made
   of an unknown material, and both are recovered from the range profile.
   Because the observation passes through an FFT, the loss landscape is the
   rugged one of Sec. 4, full of wavelength-scale local minima.  Running the
   same optimisation with and without the coarse-to-fine Dirichlet surrogate
   of Eq. 18-20 reproduces the ablation of Fig. 16(b).

Note that recovering the wall distance is only possible because the simulator
is differentiable *with respect to geometry*; experiment 1 shows that the
conventional visibility test gives exactly zero gradient there.

Outputs: ``results/exp4_inverse_material.{png,json}``
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                      # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                        # noqa: E402

from rfdt import antennas, scenes                                      # noqa: E402
from rfdt.geometry import plate                                        # noqa: E402
from rfdt.materials import C0, MaterialParams, get_material            # noqa: E402
from rfdt.optimize import multiscale_mse, optimize_digital_twin        # noqa: E402
from rfdt.signal import FMCWConfig, annealed_range_profile             # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                        # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

TRUE_WALL = "concrete"       # ground-truth material to recover
INIT_WALL = "wood"           # deliberately wrong starting guess
FREQ_A = 5.0e9
N_ROUTE = 32
#: RSS is recorded at several frequencies, as a real survey would be, which
#: adds independent constraints without adding measurement locations.
FREQS_A = [4.9e9, 5.0e9, 5.1e9]
EPOCHS_A = 150
NOISE_DB = 0.5               # measurement noise added to the synthetic RSS

TRUE_RANGE = 3.0             # ground-truth wall distance for study B [m]
#: Initial distance errors swept in study B [m].  The ablation is reported
#: across the whole sweep rather than at one hand-picked offset, so that the
#: regime where each method works, and where both fail, is visible.
INIT_OFFSETS = [0.02, 0.05, 0.10, 0.20, 0.40, 0.80]
EPOCHS_B = 200
SEED = 0


# ---------------------------------------------------------------------------
# study A: material from an RSS survey
# ---------------------------------------------------------------------------
def study_a(verbose=True):
    """Recover wall (eps', sigma) from the robot's RSS survey."""
    torch.manual_seed(SEED)
    traj = scenes.survey_trajectory(n_samples=N_ROUTE)
    tx = antennas.wifi_ap((1.2, 1.2, 2.7), FREQ_A, 20.0)
    rx = antennas.robot_client(traj.positions[0])
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=True)

    # the scene the optimiser works with; only the material changes, so the
    # candidate path search is done once and reused every iteration
    mesh = scenes.furnished_room(wall=TRUE_WALL)
    tracer = RFDTracer(mesh, cfg)
    seqs = tracer.candidate_sequences(tx.position, traj.positions, 1)

    txs = [antennas.wifi_ap((1.2, 1.2, 2.7), f, 20.0) for f in FREQS_A]

    def rss(params=None):
        """Route RSS in dBm at every survey frequency, concatenated."""
        return torch.cat([
            tracer.trace(t, rx, rx_positions=traj.positions,
                         mat_overrides=params, sequences=seqs
                         ).power_dbm(t.power_dbm) for t in txs])

    with torch.no_grad():
        clean = rss()
        target = clean + NOISE_DB * torch.randn_like(clean)

    truth = get_material(TRUE_WALL)
    start = get_material(INIT_WALL)
    params = {TRUE_WALL: MaterialParams.from_material(start, FREQ_A)}

    def readout():
        """Current parameter estimates, in physical units."""
        return {"eps_real": float(params[TRUE_WALL].eps_real),
                "sigma": float(params[TRUE_WALL].sigma)}

    def forward(epoch, lam):
        """One simulator evaluation with the current material parameters."""
        return rss(params)

    res = optimize_digital_twin(
        forward, target, params[TRUE_WALL].tensors(), epochs=EPOCHS_A, lr=0.08,
        loss_fn=multiscale_mse, use_surrogate=False, readout=readout,
        verbose=verbose)

    true_eps = float(truth.eps_real(FREQ_A))
    true_sig = float(truth.sigma(FREQ_A))
    got = res.best_params or readout()

    # eps' and sigma are not equally identifiable from RSS alone: what the
    # measurement really constrains is the reflection coefficient, and many
    # (eps', sigma) pairs give nearly the same |Gamma| at the incidence angles
    # this route happens to sample.  Report the identifiable quantity too,
    # rather than only the individual parameters.
    from rfdt.materials import Material, reflection_coefficient
    fitted = Material("fitted", a=got["eps_real"], b=0.0, c=got["sigma"], d=0.0)
    cos_ti = torch.tensor(np.cos(np.deg2rad([0.0, 30.0, 45.0, 60.0, 75.0])),
                          dtype=torch.float64)
    g_true = reflection_coefficient(FREQ_A, cos_ti, truth, "perp").abs()
    g_fit = reflection_coefficient(FREQ_A, cos_ti, fitted, "perp").abs()
    g_init = reflection_coefficient(FREQ_A, cos_ti, start, "perp").abs()
    refl_err = float((20 * torch.log10(g_fit / g_true)).abs().mean())
    refl_err_init = float((20 * torch.log10(g_init / g_true)).abs().mean())

    return {
        "reflection_error_db": refl_err,
        "reflection_error_db_initial": refl_err_init,
        "n_observations": int(target.numel()),
        "frequencies_hz": FREQS_A,
        "true_material": TRUE_WALL, "init_material": INIT_WALL,
        "true_eps_real": true_eps, "true_sigma": true_sig,
        "init_eps_real": float(start.eps_real(FREQ_A)),
        "init_sigma": float(start.sigma(FREQ_A)),
        "recovered_eps_real": got["eps_real"], "recovered_sigma": got["sigma"],
        "eps_error_pct": 100.0 * abs(got["eps_real"] - true_eps) / true_eps,
        "sigma_error_pct": 100.0 * abs(got["sigma"] - true_sig) / true_sig,
        "loss_history": res.loss_history,
        "eps_history": [h["eps_real"] for h in res.param_history],
        "sigma_history": [h["sigma"] for h in res.param_history],
        "seconds": res.seconds,
    }


# ---------------------------------------------------------------------------
# study B: geometry and material from radar, surrogate ablation
# ---------------------------------------------------------------------------
def _radar_setup():
    """Radar, scene template and FMCW configuration for study B."""
    tx, rx = antennas.mmwave_radar((0.0, 0.0, 0.9), (1.0, 0.0, 0.0), 77e9)
    rx.position = rx.position + torch.tensor([0.0, 0.02, 0.0], dtype=torch.float64)
    mesh = plate((TRUE_RANGE, 0.0, 0.9), (4.0, 3.0), "x", TRUE_WALL, "wall",
                 flip=True).weld()
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=True)
    return tx, rx, mesh, cfg, FMCWConfig()


def _profile(tracer, tx, rx, mesh, fmcw, distance, params, lam):
    """Range profile for a wall placed at ``distance``, at annealing level ``lam``.

    ``distance`` is a differentiable tensor: the wall's vertices are rebuilt
    from it, so the gradient reaches the geometry through the reparameterised
    reflection point of Eq. 52-53.
    """
    base = mesh.vertices.detach()
    verts = torch.stack([torch.full_like(base[:, 0], 0.0) + distance,
                         base[:, 1], base[:, 2]], dim=-1)
    p = tracer.trace(tx, rx, vertices=verts, mat_overrides=params)
    keep = [i for i, k in enumerate(p.kind) if k != "los"]
    p = p.select(keep)
    return annealed_range_profile(p.delay, p.gain, fmcw, lam)


def study_b(use_surrogate, init_offset, verbose=False):
    """Recover wall distance and material from a radar range profile.

    ``init_offset`` is how far the initial distance guess is from the truth.
    """
    torch.manual_seed(SEED)
    tx, rx, mesh, cfg, fmcw = _radar_setup()
    tracer = RFDTracer(mesh, cfg)

    with torch.no_grad():
        target = _profile(tracer, tx, rx, mesh, fmcw,
                          torch.tensor(TRUE_RANGE, dtype=torch.float64), None, 1.0)
        target = target + 0.002 * target.max() * torch.randn_like(target)

    init_range = TRUE_RANGE + init_offset
    distance = torch.tensor(init_range, dtype=torch.float64, requires_grad=True)
    params = {TRUE_WALL: MaterialParams.from_material(get_material(INIT_WALL), 77e9)}

    def readout():
        """Current distance and material estimates."""
        return {"distance_m": float(distance),
                "eps_real": float(params[TRUE_WALL].eps_real),
                "sigma": float(params[TRUE_WALL].sigma)}

    def forward(epoch, lam):
        """One radar simulation at the current annealing level."""
        return _profile(tracer, tx, rx, mesh, fmcw, distance, params, lam)

    res = optimize_digital_twin(
        forward, target, [distance] + params[TRUE_WALL].tensors(),
        epochs=EPOCHS_B, lr=0.03, loss_fn=multiscale_mse,
        use_surrogate=use_surrogate, warmup=0.45, readout=readout,
        verbose=verbose)

    got = res.best_params or readout()
    truth = get_material(TRUE_WALL)
    return {
        "use_surrogate": use_surrogate,
        "true_range_m": TRUE_RANGE, "init_range_m": init_range,
        "init_offset_m": init_offset,
        "init_offset_cells": init_offset / fmcw.range_resolution,
        "recovered_range_m": got["distance_m"],
        "range_error_cm": 100.0 * abs(got["distance_m"] - TRUE_RANGE),
        "true_eps_real": float(truth.eps_real(77e9)),
        "recovered_eps_real": got["eps_real"],
        "range_resolution_cm": 100.0 * fmcw.range_resolution,
        "loss_history": res.loss_history,
        "lambda_history": res.lambda_history,
        "range_history": [h["distance_m"] for h in res.param_history],
        "seconds": res.seconds,
    }


def main():
    """Run both inverse studies and write the figure and JSON summary."""
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()

    print("Experiment 4: inverse problem (digital-twin optimisation)")
    print(f"\nA. wall material from a {N_ROUTE}-point RSS survey at "
          f"{len(FREQS_A)} frequencies near {FREQ_A/1e9:.0f} GHz")
    print(f"   truth = {TRUE_WALL}, initial guess = {INIT_WALL}")
    a = study_a()
    print(f"   eps'  : true {a['true_eps_real']:.3f}  init "
          f"{a['init_eps_real']:.3f}  ->  recovered {a['recovered_eps_real']:.3f} "
          f"({a['eps_error_pct']:.1f} % error)")
    print(f"   sigma : true {a['true_sigma']:.5f}  init {a['init_sigma']:.5f}  "
          f"->  recovered {a['recovered_sigma']:.5f} "
          f"({a['sigma_error_pct']:.1f} % error)")
    print(f"   |Gamma| error, averaged over 0-75 deg incidence: "
          f"{a['reflection_error_db_initial']:.2f} dB at the initial guess "
          f"-> {a['reflection_error_db']:.2f} dB after fitting")
    print(f"   loss  : {a['loss_history'][0]:.4e} -> {a['loss_history'][-1]:.4e}"
          f"   ({a['seconds']:.0f} s, {a['n_observations']} observations)")

    print(f"\nB. wall distance and material from the robot's 77 GHz radar")
    print(f"   truth = {TRUE_RANGE} m of {TRUE_WALL}, material guess = {INIT_WALL}")
    print(f"   {'init error':>12s} {'cells':>7s} {'with surrogate':>16s} "
          f"{'without surrogate':>19s}")
    sweep = []
    for off in INIT_OFFSETS:
        w = study_b(True, off)
        n = study_b(False, off)
        sweep.append({"init_offset_m": off, "with": w, "without": n})
        print(f"   {off*100:9.0f} cm {w['init_offset_cells']:7.1f} "
              f"{w['range_error_cm']:13.2f} cm {n['range_error_cm']:16.2f} cm")

    payload = {"A_material_from_rss": a,
               "B_radar_sweep": [
                   {"init_offset_m": s_["init_offset_m"],
                    "init_offset_cells": s_["with"]["init_offset_cells"],
                    "with_surrogate_error_cm": s_["with"]["range_error_cm"],
                    "without_surrogate_error_cm": s_["without"]["range_error_cm"]}
                   for s_ in sweep],
               "B_detail_with": sweep[len(sweep) // 2]["with"],
               "B_detail_without": sweep[len(sweep) // 2]["without"],
               "runtime_s": time.time() - t0}
    with open(os.path.join(RESULTS, "exp4_inverse_material.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    _plot(a, sweep)
    print(f"\nwrote results/exp4_inverse_material.{{png,json}} "
          f"({payload['runtime_s']:.0f} s)")
    return payload


def _plot(a, sweep):
    """Four-panel figure: convergence of study A and the study B sweep."""
    mid = sweep[len(sweep) // 2]
    b_with, b_without = mid["with"], mid["without"]
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.3))

    ax = axes[0]
    ax.semilogy(a["loss_history"], color="#2e86c1", lw=1.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("multiscale MSE loss")
    ax.set_title("A. Material from RSS survey\n(loss)", fontsize=10.5)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(a["eps_history"], color="#2e86c1", lw=1.7, label=r"$\varepsilon'$")
    ax.axhline(a["true_eps_real"], color="#2e86c1", ls="--", lw=1.2,
               label=r"true $\varepsilon'$")
    ax2 = ax.twinx()
    ax2.plot(a["sigma_history"], color="#c0392b", lw=1.7, label=r"$\sigma$")
    ax2.axhline(a["true_sigma"], color="#c0392b", ls="--", lw=1.2)
    ax2.set_ylabel(r"$\sigma$ [S/m]", color="#c0392b")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\varepsilon'$", color="#2e86c1")
    ax.set_title(f"A. Recovering {TRUE_WALL}\nfrom a {INIT_WALL} initialisation",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, loc="lower right", frameon=False)

    ax = axes[2]
    ax.semilogy(b_without["loss_history"], color="#c0392b", lw=1.6,
                label="without surrogate")
    ax.semilogy(b_with["loss_history"], color="#2e86c1", lw=1.6,
                label="with surrogate")
    lam = np.asarray(b_with["lambda_history"])
    switch = int(np.argmax(lam > 0)) if (lam > 0).any() else 0
    ax.axvline(switch, color="0.5", ls=":", lw=1.2)
    ax.annotate("annealing starts", (switch, ax.get_ylim()[1]), fontsize=8,
                rotation=90, va="top", ha="right", color="0.4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("multiscale MSE loss")
    ax.set_title("B. Radar inverse problem\n(surrogate ablation, Fig. 16b)",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[3]
    off = np.asarray([s["init_offset_m"] for s in sweep]) * 100.0
    ew = np.asarray([s["with"]["range_error_cm"] for s in sweep])
    en = np.asarray([s["without"]["range_error_cm"] for s in sweep])
    ax.plot(off, en, "o-", color="#c0392b", lw=1.6, ms=4, label="without surrogate")
    ax.plot(off, ew, "o-", color="#2e86c1", lw=1.6, ms=4, label="with surrogate")
    ax.plot(off, off, ":", color="0.5", lw=1.2, label="no improvement")
    ax.axhline(b_with["range_resolution_cm"], color="0.3", ls="--", lw=1.1,
               label="one range cell")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("initial distance error [cm]")
    ax.set_ylabel("final distance error [cm]")
    ax.set_title("B. Recovery vs how wrong the start is\n"
                 "(swept, not a single chosen offset)", fontsize=10.5)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7.5, frameon=False)

    fig.suptitle("Experiment 4: inverse rendering, recovering scene parameters "
                 "from RF measurements", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(os.path.join(RESULTS, "exp4_inverse_material.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()