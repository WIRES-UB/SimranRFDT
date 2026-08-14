"""Experiment 1: verification of differentiability (RFDT Sec. 6.1).

Reproduces the controlled study behind Fig. 3 and Fig. 8 in the paper, using a
finite conducting plate and a transmitter that can be translated.

Three questions are answered:

1. **Is the field continuous?**  The receiver is swept so that the specular
   reflection point walks across a plate edge.  The conventional binary
   in-facet test (Eq. 3) steps discontinuously; the soften-triangle sigmoid
   (Eq. 4) is continuous but biased, leaking energy far outside the plate and
   suppressing it inside; the RFDT weight (Eq. 11) plus edge diffraction
   (Eq. 6) is continuous *and* unbiased.

2. **Are the gradients correct?**  Autograd through the whole pipeline is
   compared against central finite differences, the ground truth used in
   Sec. 6.1.

3. **Are the gradients useful?**  Correctness is not enough.  A gradient with
   respect to the *transmitter position* is easy: the path length varies
   smoothly however visibility is modelled.  The decisive quantity is the
   gradient with respect to the reflector's *extent*, because scaling a facet
   does not move its supporting plane and so changes the field only through
   the validity test.  A Heaviside has zero derivative almost everywhere, so
   the conventional model's gradient there is identically zero and no
   optimiser can ever learn a reflector's size from it.  This is the
   "d-Geometry" column of the paper's Table 1, measured directly.

Outputs: ``results/exp1_differentiability.{png,json}``
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from rfdt import scenes                                              # noqa: E402
from rfdt.antennas import Antenna, Receiver, Transmitter             # noqa: E402
from rfdt.metrics import continuity_jump, gradient_agreement         # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                      # noqa: E402

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")
FREQ = 5.0e9
PLATE = 1.0          # plate side length [m]
EDGE_X = 1.0         # receiver x at which the specular point crosses the edge

#: The three path-validity models under test, as (label, TracerConfig kwargs).
SCHEMES = [
    ("Conventional (Eq. 3)", dict(weighting="heaviside", enable_diffraction=False)),
    ("Soften triangle (Eq. 4)", dict(weighting="sigmoid", sigmoid_k=40.0,
                                     enable_diffraction=False)),
    ("RFDT (Eq. 11)", dict(weighting="rfdt", enable_diffraction=False)),
    ("RFDT + diffraction", dict(weighting="rfdt", enable_diffraction=True)),
]


def build_scene():
    """A single 1 m conducting plate with the transmitter 1 m above its centre."""
    mesh = scenes.plate_scene(plate_size=(PLATE, PLATE), material="metal")
    tx = Transmitter((0.0, 0.0, 1.0), FREQ, 20.0, Antenna("isotropic"))
    rx = Receiver((0.0, 0.0, 1.0), Antenna("isotropic"))
    return mesh, tx, rx


def sweep_field(mesh, tx, rx, cfg_kwargs, xs):
    """Scattered field magnitude as the receiver sweeps across the edge.

    The LoS path is excluded so that the reflected/diffracted contribution,
    which is the discontinuous one, is visible on its own.
    """
    cfg = TracerConfig(max_order=1, **cfg_kwargs)
    pos = torch.stack([xs, torch.zeros_like(xs), torch.ones_like(xs)], dim=-1)
    paths = RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=pos)
    keep = [i for i, k in enumerate(paths.kind) if k != "los"]
    return paths.gain[:, keep].sum(-1).abs()


def field_of_tx_height(mesh, rx, theta, cfg_kwargs):
    """Total field magnitude as a function of a vertical translation of the Tx.

    ``theta`` is the scene parameter differentiated in Sec. 6.1.
    """
    cfg = TracerConfig(max_order=1, **cfg_kwargs)
    z = torch.zeros((), dtype=torch.float64)
    tx = Transmitter(torch.stack([z, z, 1.0 + theta]), FREQ, 20.0, Antenna("isotropic"))
    return RFDTracer(mesh, cfg).trace(tx, rx).field().abs().squeeze()


def gradient_study(mesh, rx, cfg_kwargs, thetas, h=1e-6):
    """Autograd vs central finite differences over a range of Tx translations."""
    ad, fd = [], []
    for th0 in thetas:
        t = torch.tensor(float(th0), dtype=torch.float64, requires_grad=True)
        y = field_of_tx_height(mesh, rx, t, cfg_kwargs)
        y.backward()
        ad.append(float(t.grad))
        plus = float(field_of_tx_height(mesh, rx, torch.tensor(th0 + h,
                                                               dtype=torch.float64),
                                        cfg_kwargs))
        minus = float(field_of_tx_height(mesh, rx, torch.tensor(th0 - h,
                                                                dtype=torch.float64),
                                         cfg_kwargs))
        fd.append((plus - minus) / (2 * h))
    return np.asarray(ad), np.asarray(fd)


def geometry_gradient(cfg_kwargs, xs, delta=0.01):
    """Gradient of the field w.r.t. the *size of the plate* (Table 1, d-Geometry).

    This is the decisive test.  A correct gradient w.r.t. the transmitter
    position proves little: the path length varies smoothly no matter how
    visibility is modelled.  Extent is different.  Scaling the plate does not
    move the supporting plane, so it changes the field *only* through the
    validity test.  A Heaviside has zero derivative almost everywhere, so its
    analytic gradient w.r.t. the plate size is identically zero: the optimiser
    is blind to the reflector's extent however accurate its other gradients
    are.  The RFDT weight varies smoothly with the edge position and therefore
    produces a usable one.

    Returns ``(analytic, finite_difference)`` arrays over the receiver sweep.
    ``delta`` is a 1 cm perturbation, the scale an optimiser step would take.
    """
    mesh = scenes.plate_scene(plate_size=(PLATE, PLATE), material="metal")
    base = mesh.vertices.detach().clone()
    tx = Transmitter((0.0, 0.0, 1.0), FREQ, 20.0, Antenna("isotropic"))
    rx = Receiver((0.0, 0.0, 1.0), Antenna("isotropic"))
    cfg = TracerConfig(max_order=1, **cfg_kwargs)
    tracer = RFDTracer(mesh, cfg)
    pos = torch.stack([xs, torch.zeros_like(xs), torch.ones_like(xs)], dim=-1)
    seqs = tracer.candidate_sequences(tx.position, pos, 1)

    def field(scale):
        """Field magnitude with the plate's x extent scaled by ``scale``."""
        v = torch.stack([base[:, 0] * scale, base[:, 1], base[:, 2]], dim=-1)
        p = tracer.trace(tx, rx, rx_positions=pos, vertices=v, sequences=seqs)
        keep = [i for i, k in enumerate(p.kind) if k != "los"]
        return p.gain[:, keep].sum(-1).abs()

    # one backward pass per receiver, so the gradient is resolved per position
    # rather than summed over the sweep
    per = []
    for i in range(len(xs)):
        si = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        field(si)[i].backward()
        per.append(float(si.grad))
    analytic = np.asarray(per)

    with torch.no_grad():
        plus = field(torch.tensor(1.0 + delta, dtype=torch.float64)).numpy()
        minus = field(torch.tensor(1.0 - delta, dtype=torch.float64)).numpy()
    return analytic, (plus - minus) / (2 * delta)


def main():
    """Run the three studies, write the figure and the JSON summary."""
    os.makedirs(RESULTS, exist_ok=True)
    mesh, tx, rx = build_scene()
    summary = {"frequency_hz": FREQ, "plate_m": PLATE, "schemes": {}}

    # study 1: continuity across the edge -----------------------------------
    xs = torch.linspace(EDGE_X - 0.25, EDGE_X + 0.25, 401, dtype=torch.float64)
    curves = {}
    for label, kw in SCHEMES:
        e = sweep_field(mesh, tx, rx, kw, xs)
        curves[label] = e.detach().numpy()
        summary["schemes"].setdefault(label, {})["continuity"] = continuity_jump(e)

    # study 2 and 3: gradients ----------------------------------------------
    rx_grad = Receiver((1.1, 0.0, 1.0), Antenna("isotropic"))
    thetas = np.linspace(-0.3, 0.5, 33)
    grads, geo = {}, {}
    xs_geo = torch.linspace(EDGE_X - 0.20, EDGE_X + 0.20, 81, dtype=torch.float64)
    for label, kw in SCHEMES:
        ad, fd = gradient_study(mesh, rx_grad, kw, thetas)
        grads[label] = (ad, fd)
        summary["schemes"][label]["gradient_vs_fd"] = gradient_agreement(
            torch.as_tensor(ad), torch.as_tensor(fd))

        g_ad, g_fd = geometry_gradient(kw, xs_geo)
        geo[label] = (g_ad, g_fd)
        nonzero = float(np.mean(np.abs(g_ad) > 1e-14))
        denom = np.linalg.norm(g_ad) * np.linalg.norm(g_fd)
        summary["schemes"][label]["geometry_gradient"] = {
            "fraction_nonzero": nonzero,
            "cosine_similarity_with_fd": float(
                (g_ad * g_fd).sum() / denom) if denom > 0 else 0.0,
            "mean_abs": float(np.abs(g_ad).mean()),
        }

    # figure ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    colors = {"Conventional (Eq. 3)": "#c0392b", "Soften triangle (Eq. 4)": "#e08a1e",
              "RFDT (Eq. 11)": "#2e86c1", "RFDT + diffraction": "#1a5276"}

    ax = axes[0]
    for label, _ in SCHEMES:
        ax.plot(xs.numpy(), curves[label] * 1e3, label=label,
                color=colors[label], lw=1.9)
    ax.axvline(EDGE_X, color="0.4", ls=":", lw=1.2)
    ax.annotate("facet edge", (EDGE_X, ax.get_ylim()[1] * 0.94), ha="center",
                fontsize=9, color="0.35")
    ax.set_xlabel("receiver x [m]")
    ax.set_ylabel(r"scattered field $|E|$  [$\times 10^{-3}$]")
    ax.set_title("Field as the specular point crosses an edge")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for label, _ in SCHEMES:
        ad, fd = grads[label]
        ax.plot(thetas, ad, color=colors[label], lw=1.8, label=f"{label} (autograd)")
        ax.plot(thetas, fd, color=colors[label], lw=0, marker="o", ms=2.6, alpha=0.65)
    ax.set_xlabel(r"Tx vertical translation $\theta$ [m]")
    ax.set_ylabel(r"$\partial |E| / \partial \theta$")
    ax.set_title("Autograd (lines) vs finite differences (dots)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    ax = axes[2]
    for label, _ in SCHEMES:
        g_ad, g_fd = geo[label]
        ax.plot(xs_geo.numpy(), g_ad * 1e3, color=colors[label], lw=1.8, label=label)
    g_fd = geo["RFDT + diffraction"][1]
    ax.plot(xs_geo.numpy(), g_fd * 1e3, color="0.25", lw=0, marker="o", ms=2.6,
            alpha=0.6, label="finite difference (RFDT)")
    ax.axvline(EDGE_X, color="0.4", ls=":", lw=1.2)
    ax.set_xlabel("receiver x [m]")
    ax.set_ylabel(r"$\partial |E| / \partial(\mathrm{plate\ scale})$  [$\times 10^{-3}$]")
    ax.set_title("Gradient w.r.t. plate size\n(conventional test is identically zero)")
    ax.legend(fontsize=7, frameon=False)
    ax.grid(alpha=0.25)

    fig.suptitle("Experiment 1: differentiability of the RF visibility term "
                 f"(metal plate {PLATE} m, {FREQ/1e9:.0f} GHz)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png = os.path.join(RESULTS, "exp1_differentiability.png")
    fig.savefig(png, dpi=160)
    plt.close(fig)

    with open(os.path.join(RESULTS, "exp1_differentiability.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print("Experiment 1: differentiability")
    print(f"{'scheme':26s} {'max jump/range':>15s} {'dTx rel.err vs FD':>19s} "
          f"{'d(size) nonzero':>16s} {'d(size) cos vs FD':>18s}")
    for label, _ in SCHEMES:
        r = summary["schemes"][label]
        print(f"{label:26s} {r['continuity']['max_jump_normalised']:15.4f} "
              f"{r['gradient_vs_fd']['rel_l2_error']:19.2e} "
              f"{r['geometry_gradient']['fraction_nonzero']:16.3f} "
              f"{r['geometry_gradient']['cosine_similarity_with_fd']:18.4f}")
    print(f"\nwrote {png}")
    return summary


if __name__ == "__main__":
    main()