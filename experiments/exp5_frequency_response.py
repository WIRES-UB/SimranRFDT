"""Experiment 5: channel frequency response, amplitude and phase.

Experiments 1 to 4 work in the spatial and delay domains.  This one works in
the frequency domain, which is what a wideband receiver actually estimates: the
complex transfer function

    H(f) = sum_i alpha_i exp(-2 pi j f tau_i)

over a band of carriers.  Four studies:

A. **Two-ray frequency selectivity.**  A transmitter, a receiver and one
   conducting reflector, with the direct path compared against direct plus
   reflection.  A single reflector turns a flat channel into a comb of deep
   notches.  Their spacing is predicted in closed form by ``c / dL``, where
   ``dL`` is the path-length difference, so this doubles as a validation of the
   simulator against analysis.

B. **Phase and group delay.**  The unwrapped phase of ``H(f)`` and its
   derivative.  A single path gives a straight phase ramp whose slope is
   exactly the propagation delay; adding a reflector makes the group delay
   swing wildly near each notch, which is what distorts a wideband signal.

C. **Frequency selectivity versus wall material** in the furnished room.  This
   is the frequency-domain counterpart of the delay-spread result in
   experiment 2.

D. **Coherence bandwidth, measured rather than estimated.**  Experiment 2
   reports coherence bandwidth from the RMS delay spread using the standard
   ``1/(5 sigma_tau)`` rule of thumb.  Here it is measured directly from the
   frequency correlation of ``H(f)``, which is an independent check on that
   estimate rather than a restatement of it.

Path delays are exactly frequency independent, being pure geometry, but path
amplitudes are not: free-space spreading goes as ``1/f`` and the Fresnel
coefficients depend on frequency through the material model.  The scene is
therefore re-traced at every frequency point rather than traced once and
extrapolated.  The cost of that choice is reported, and the narrowband
approximation is quantified against it in study A.

Outputs: ``results/exp5_frequency_response.{png,json,csv}``
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
from rfdt.antennas import Antenna, Receiver, Transmitter              # noqa: E402
from rfdt.materials import C0, get_material                           # noqa: E402
from rfdt.metrics import rms_delay_spread                             # noqa: E402
from rfdt.signal import ofdm_channel                                  # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

# study A and B: two-ray geometry, chosen so the reflected path is 1.08 m longer
TX_A = (0.0, 0.0, 2.0)
RX_A = (3.0, 0.0, 1.0)
BAND_A = (4.0e9, 6.0e9)
N_FREQ_A = 1001

# study C and D: furnished room, one receiver, wide band for correlation
RX_C = (3.6, 2.2, 0.9)
AP_C = (1.2, 1.2, 2.7)
BAND_C = (4.5e9, 5.5e9)
N_FREQ_C = 201
MATERIALS_C = ["metal", "concrete", "foam_board"]


def sweep_frequency(tracer, tx_factory, rx, rx_pos, freqs, sequences=None):
    """Trace the scene at every frequency and return the transfer function.

    Returns ``(H, delays, gains)`` where ``H`` is the complex ``H(f)`` sampled
    on ``freqs``, and ``delays``/``gains`` are the path data at the band centre
    (kept so the narrowband approximation can be compared against this).
    """
    H = torch.zeros(len(freqs), dtype=torch.complex128)
    mid_delays = mid_gains = None
    for i, f in enumerate(freqs):
        tx = tx_factory(float(f))
        p = tracer.trace(tx, rx, rx_positions=rx_pos, sequences=sequences)
        H[i] = p.field().squeeze()
        if i == len(freqs) // 2:
            mid_delays, mid_gains = p.delay.detach(), p.gain.detach()
    return H, mid_delays, mid_gains


def study_ab():
    """Two-ray amplitude, phase and group delay against closed-form analysis."""
    freqs = torch.linspace(*BAND_A, N_FREQ_A, dtype=torch.float64)

    def make_tx(f):
        """Isotropic transmitter at frequency ``f``."""
        return Transmitter(TX_A, f, 20.0, Antenna("isotropic"))

    rx = Receiver(RX_A, Antenna("isotropic"))
    rx_pos = torch.tensor([RX_A], dtype=torch.float64)

    # line of sight only: a far-away facet, so the tracer finds no reflection
    los_mesh = scenes.plate_scene(plate_size=(1.0, 1.0), material="metal",
                                 center=(0.0, 0.0, -60.0))
    los_tr = RFDTracer(los_mesh, TracerConfig(max_order=0, enable_diffraction=False))
    H_los, _, _ = sweep_frequency(los_tr, make_tx, rx, rx_pos, freqs)

    # direct plus one reflection off a large conducting ground plane
    ref_mesh = scenes.plate_scene(plate_size=(30.0, 30.0), material="metal")
    ref_tr = RFDTracer(ref_mesh, TracerConfig(max_order=1, enable_diffraction=False))
    H_ref, d_mid, g_mid = sweep_frequency(ref_tr, make_tx, rx, rx_pos, freqs)

    # closed-form prediction of the notch spacing
    l_direct = float(np.linalg.norm(np.array(RX_A) - np.array(TX_A)))
    image = np.array([TX_A[0], TX_A[1], -TX_A[2]])
    l_reflect = float(np.linalg.norm(np.array(RX_A) - image))
    dl = l_reflect - l_direct
    spacing_analytic = C0 / dl

    def local_minima(x):
        """Indices of interior local minima of a 1-D array."""
        interior = np.arange(1, len(x) - 1)
        return interior[(x[1:-1] < x[:-2]) & (x[1:-1] < x[2:])]

    # measured notch spacing, from the minima of |H|
    mag = H_ref.abs().numpy()
    notch_f = freqs.numpy()[local_minima(mag)]
    spacing_measured = float(np.mean(np.diff(notch_f))) if len(notch_f) > 1 else float("nan")

    # the direct-only response is not flat: free-space spreading makes the
    # amplitude fall as 1/f, so 20log10 of it drops by 20log10(f_hi/f_lo) across
    # the band.  What distinguishes it from the two-ray case is that it has no
    # notches at all.  Check both, rather than quoting a standard deviation that
    # would misrepresent a smooth slope as ripple.
    los_mag_db = 20 * np.log10(H_los.abs().numpy())
    los_minima = local_minima(H_los.abs().numpy())
    los_slope_sim = float(los_mag_db[0] - los_mag_db[-1])
    los_slope_ana = float(20 * np.log10(BAND_A[1] / BAND_A[0]))

    # narrowband approximation: trace once at band centre, then apply only the
    # delay phase, which is what a per-path OFDM model assumes
    H_narrow = ofdm_channel(d_mid, g_mid, freqs).squeeze()
    nb_err_db = float((20 * torch.log10(H_narrow.abs() / H_ref.abs())).abs().max())

    # group delay from the unwrapped phase
    def group_delay(H):
        """Negative derivative of unwrapped phase with respect to angular frequency."""
        ph = np.unwrap(np.angle(H.numpy()))
        w = 2.0 * np.pi * freqs.numpy()
        return -np.gradient(ph, w)

    return {
        "freqs_hz": freqs.numpy(),
        "H_los": H_los, "H_ref": H_ref,
        "gd_los": group_delay(H_los), "gd_ref": group_delay(H_ref),
        "phase_los": np.unwrap(np.angle(H_los.numpy())),
        "phase_ref": np.unwrap(np.angle(H_ref.numpy())),
        "l_direct_m": l_direct, "l_reflect_m": l_reflect, "delta_l_m": dl,
        "notch_spacing_analytic_hz": spacing_analytic,
        "notch_spacing_measured_hz": spacing_measured,
        "notch_spacing_error_pct": 100.0 * abs(spacing_measured - spacing_analytic)
                                   / spacing_analytic,
        "n_notches_found": int(len(notch_f)),
        "los_n_notches": int(len(los_minima)),
        "los_slope_db_simulated": los_slope_sim,
        "los_slope_db_analytic": los_slope_ana,
        "ref_notch_depth_db": float(20 * np.log10(mag.max() / mag.min())),
        "los_group_delay_ns": float(np.mean(group_delay(H_los)) * 1e9),
        "expected_group_delay_ns": l_direct / C0 * 1e9,
        "narrowband_max_error_db": nb_err_db,
    }


def frequency_correlation(H, freqs):
    """Magnitude of the frequency correlation function ``R(df)``.

    ``R(df) = <H(f) H*(f + df)> / <|H|^2>``, averaged over ``f``.  The
    coherence bandwidth is the smallest ``df`` at which it drops below 0.5.
    """
    h = H.numpy()
    h = h - h.mean()
    n = len(h)
    lags = np.arange(n // 2)
    r = np.array([np.vdot(h[:n - k], h[k:]) for k in lags])
    r = np.abs(r) / np.abs(r[0])
    df = float(freqs[1] - freqs[0])
    below = np.nonzero(r < 0.5)[0]
    bc = float(lags[below[0]] * df) if len(below) else float("nan")
    return lags * df, r, bc


def study_cd():
    """Frequency selectivity and measured coherence bandwidth versus material."""
    freqs = torch.linspace(*BAND_C, N_FREQ_C, dtype=torch.float64)
    rx_pos = torch.tensor([RX_C], dtype=torch.float64)
    rx = antennas.robot_client(RX_C)
    out = {}
    for name in MATERIALS_C:
        mesh = scenes.furnished_room(wall=name)
        tr = RFDTracer(mesh, TracerConfig(max_order=1, weighting="rfdt",
                                          enable_diffraction=True))
        seqs = tr.candidate_sequences(torch.tensor(AP_C, dtype=torch.float64),
                                      rx_pos, 1)

        def make_tx(f):
            """Ceiling access point at frequency ``f``."""
            return antennas.wifi_ap(AP_C, f, 20.0)

        H, d_mid, g_mid = sweep_frequency(tr, make_tx, rx, rx_pos, freqs, seqs)
        lags, r, bc = frequency_correlation(H, freqs)
        tau = float(rms_delay_spread(d_mid, g_mid))
        out[name] = {
            "H": H, "corr_lags_hz": lags, "corr": r,
            "coherence_bw_measured_mhz": bc / 1e6,
            "delay_spread_ns": tau * 1e9,
            "coherence_bw_rule_of_thumb_mhz": 1.0 / (5.0 * tau) / 1e6 if tau > 0 else float("nan"),
            "fading_depth_db": float(20 * torch.log10(H.abs().max() / H.abs().min())),
            "mean_level_dbm": float(20.0 + 20 * torch.log10(H.abs().mean())),
        }
    return {"freqs_hz": freqs.numpy(), "materials": out}


def main():
    """Run the frequency-domain studies and write figure, JSON and CSV."""
    os.makedirs(RESULTS, exist_ok=True)
    t0 = time.time()
    print("Experiment 5: channel frequency response")

    ab = study_ab()
    print(f"\nA. Two-ray frequency selectivity, {BAND_A[0]/1e9:.1f} to "
          f"{BAND_A[1]/1e9:.1f} GHz, {N_FREQ_A} points")
    print(f"   direct path {ab['l_direct_m']:.4f} m, reflected path "
          f"{ab['l_reflect_m']:.4f} m, difference {ab['delta_l_m']:.4f} m")
    print(f"   notch spacing: analytic c/dL = "
          f"{ab['notch_spacing_analytic_hz']/1e6:.2f} MHz, "
          f"simulated {ab['notch_spacing_measured_hz']/1e6:.2f} MHz "
          f"({ab['notch_spacing_error_pct']:.3f} % error, "
          f"{ab['n_notches_found']} notches)")
    print(f"   deepest notch is {ab['ref_notch_depth_db']:.1f} dB below the peak")
    print(f"   direct path alone has {ab['los_n_notches']} notches and falls "
          f"{ab['los_slope_db_simulated']:.3f} dB across the band; free-space "
          f"1/f spreading predicts {ab['los_slope_db_analytic']:.3f} dB")

    print(f"\nB. Phase and group delay")
    print(f"   line-of-sight group delay {ab['los_group_delay_ns']:.4f} ns, "
          f"expected L/c = {ab['expected_group_delay_ns']:.4f} ns")
    print(f"   narrowband per-path approximation departs from the re-traced "
          f"result by up to {ab['narrowband_max_error_db']:.2f} dB over this "
          f"{(BAND_A[1]-BAND_A[0])/1e9:.0f} GHz span")

    cd = study_cd()
    print(f"\nC and D. Furnished room at {RX_C}, {BAND_C[0]/1e9:.1f} to "
          f"{BAND_C[1]/1e9:.1f} GHz")
    print(f"{'wall':14s} {'fade depth':>11s} {'tau_rms':>9s} "
          f"{'Bc measured':>12s} {'Bc rule':>9s} {'ratio':>7s}")
    for name, r in cd["materials"].items():
        ratio = r["coherence_bw_measured_mhz"] / r["coherence_bw_rule_of_thumb_mhz"]
        print(f"{name:14s} {r['fading_depth_db']:8.1f} dB {r['delay_spread_ns']:7.2f} ns "
              f"{r['coherence_bw_measured_mhz']:9.1f} MHz {r['coherence_bw_rule_of_thumb_mhz']:6.1f} MHz "
              f"{ratio:7.2f}")

    _plot(ab, cd)
    _write(ab, cd, time.time() - t0)
    print(f"\nwrote results/exp5_frequency_response.{{png,json,csv}} "
          f"({time.time()-t0:.0f} s)")
    return {"A_B": ab, "C_D": cd}


def _write(ab, cd, runtime):
    """Persist the numeric results as JSON and CSV."""
    payload = {
        "A_two_ray": {k: v for k, v in ab.items()
                      if not isinstance(v, (np.ndarray, torch.Tensor))},
        "C_room": {name: {k: v for k, v in r.items()
                          if not isinstance(v, (np.ndarray, torch.Tensor))}
                   for name, r in cd["materials"].items()},
        "band_a_hz": list(BAND_A), "band_c_hz": list(BAND_C),
        "n_freq_a": N_FREQ_A, "n_freq_c": N_FREQ_C,
        "receiver_position_m": list(RX_C),
        "runtime_s": runtime,
    }
    with open(os.path.join(RESULTS, "exp5_frequency_response.json"), "w") as fh:
        json.dump(payload, fh, indent=2)
    with open(os.path.join(RESULTS, "exp5_frequency_response.csv"), "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frequency_hz", "two_ray_los_db", "two_ray_with_reflector_db"]
                   + [f"room_{m}_db" for m in MATERIALS_C])
        f_a = ab["freqs_hz"]
        los = 20 * np.log10(ab["H_los"].abs().numpy())
        ref = 20 * np.log10(ab["H_ref"].abs().numpy())
        f_c = cd["freqs_hz"]
        room = {m: 20 * np.log10(cd["materials"][m]["H"].abs().numpy())
                for m in MATERIALS_C}
        for i, f in enumerate(f_a):
            # the two sweeps use different grids; room columns are interpolated
            row = [f, los[i], ref[i]] + [float(np.interp(f, f_c, room[m]))
                                         for m in MATERIALS_C]
            w.writerow(row)


def _plot(ab, cd):
    """Four-panel figure: amplitude, phase, group delay, material comparison."""
    fig = plt.figure(figsize=(16.5, 9.2))
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.24)
    f_ghz = ab["freqs_hz"] / 1e9

    # A: amplitude
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(f_ghz, 20 * np.log10(ab["H_los"].abs().numpy()), color="#2e86c1",
            lw=1.6, label="direct path only")
    ax.plot(f_ghz, 20 * np.log10(ab["H_ref"].abs().numpy()), color="#c0392b",
            lw=1.1, label="direct + one reflector")
    sp = ab["notch_spacing_analytic_hz"] / 1e9
    first = np.ceil(f_ghz[0] / sp) * sp
    for m, x in enumerate(np.arange(first, f_ghz[-1] + 1e-9, sp)):
        ax.axvline(x, color="0.6", ls=":", lw=0.9,
                   label="predicted notch, $c/\\Delta L$" if m == 0 else None)
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"$20\log_{10}|H(f)|$ [dB]")
    ax.set_title(f"A. Two-ray frequency selectivity\nnotch spacing "
                 f"{ab['notch_spacing_analytic_hz']/1e6:.1f} MHz predicted, "
                 f"{ab['notch_spacing_measured_hz']/1e6:.1f} MHz simulated",
                 fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, loc="lower left")

    # B: phase and group delay
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(f_ghz, ab["phase_los"] / np.pi, color="#2e86c1", lw=1.6,
            label="direct only")
    ax.plot(f_ghz, ab["phase_ref"] / np.pi, color="#c0392b", lw=1.1,
            label="direct + reflector")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"unwrapped phase of $H(f)$  [$\times\pi$ rad]")
    ax.set_title("B. Phase, and group delay (right axis)", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(f_ghz, ab["gd_ref"] * 1e9, color="#7d3c98", lw=0.9, alpha=0.75)
    ax2.axhline(ab["expected_group_delay_ns"], color="#2e86c1", ls="--", lw=1.1)
    ax2.set_ylabel("group delay [ns]", color="#7d3c98")
    ax2.set_ylim(-40, 60)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    # C: room, per material
    ax = fig.add_subplot(gs[1, 0])
    colors = {"metal": "#7f8c8d", "concrete": "#2c3e50", "foam_board": "#27ae60"}
    fc = cd["freqs_hz"] / 1e9
    for name, r in cd["materials"].items():
        ax.plot(fc, 20 * np.log10(r["H"].abs().numpy()), lw=1.3,
                color=colors[name], label=f"{get_material(name).label} "
                                          f"({r['fading_depth_db']:.0f} dB fade)")
    ax.set_xlabel("frequency [GHz]")
    ax.set_ylabel(r"$20\log_{10}|H(f)|$ [dB]")
    ax.set_title(f"C. Frequency selectivity vs wall material\n"
                 f"furnished room, receiver at {RX_C}", fontsize=10.5)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    # D: frequency correlation and coherence bandwidth
    ax = fig.add_subplot(gs[1, 1])
    for name, r in cd["materials"].items():
        ax.plot(r["corr_lags_hz"] / 1e6, r["corr"], lw=1.5, color=colors[name],
                label=f"{get_material(name).label}: "
                      f"$B_c$ = {r['coherence_bw_measured_mhz']:.0f} MHz")
        ax.axvline(r["coherence_bw_measured_mhz"], color=colors[name], ls=":",
                   lw=1.0)
    ax.axhline(0.5, color="0.4", ls="--", lw=1.1, label="0.5 correlation")
    ax.set_xlabel(r"frequency separation $\Delta f$ [MHz]")
    ax.set_ylabel(r"$|R(\Delta f)|$")
    ax.set_title("D. Coherence bandwidth measured from $H(f)$,\n"
                 "not inferred from delay spread", fontsize=10.5)
    ax.set_xlim(0, 300)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("Experiment 5: channel frequency response, amplitude and phase",
                 fontsize=13)
    fig.savefig(os.path.join(RESULTS, "exp5_frequency_response.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()