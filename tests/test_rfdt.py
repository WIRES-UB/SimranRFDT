"""Physics and differentiability regression tests for the RFDT simulator.

Each test pins down a property that was verified during development, so that a
later change cannot silently break it.  Run with::

    python3 tests/test_rfdt.py          # no pytest needed
    python3 -m pytest tests/ -q         # or with pytest, if installed
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfdt import antennas, scenes, transition as tf                    # noqa: E402
from rfdt.antennas import Antenna, Receiver, Transmitter                # noqa: E402
from rfdt.materials import (C0, get_material, reflection_coefficient,   # noqa: E402
                            slab_transmission)
from rfdt.tracer import RFDTracer, TracerConfig                         # noqa: E402

TOL = 1e-9


# ---------------------------------------------------------------------------
# transition function (Eq. 8, 44, 49)
# ---------------------------------------------------------------------------
def test_faddeeva_reference_values():
    """Weideman's approximation must reproduce known Faddeeva values."""
    assert abs(complex(tf.faddeeva(torch.tensor(0.0 + 0j))) - 1.0) < 1e-12
    ref = 0.3047442052569126 + 0.20821893820283163j
    assert abs(complex(tf.faddeeva(torch.tensor(1 + 1j))) - ref) < 1e-12


def test_transition_asymptotics():
    """F(x) -> 1 for large x and F(x) ~ sqrt(pi) e^{j pi/4} sqrt(x) for small x."""
    assert abs(complex(tf.transition_F(torch.tensor(1e6))) - 1.0) < 1e-5
    x = 1e-6
    # Eq. 49 keeps two terms: F ~ sqrt(pi) e^{j pi/4} sqrt(x) - 2 j x, with an
    # O(x^{3/2}) remainder, which at x = 1e-6 is of order 1e-9
    series = np.sqrt(np.pi) * np.exp(1j * np.pi / 4) * np.sqrt(x) - 2j * x
    assert abs(complex(tf.transition_F(torch.tensor(x))) - series) < 1e-8


def test_transition_custom_backward_matches_autograd():
    """The hand-written F'(x) must agree with autograd through the rational form."""
    for x0 in [1e-6, 1e-3, 0.05, 0.5, 3.0, 50.0]:
        xa = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        ya = tf._F_raw(xa)
        (ya.real + 0.3 * ya.imag).backward()
        xb = torch.tensor(x0, dtype=torch.float64, requires_grad=True)
        yb = tf.transition_F(xb)
        (yb.real + 0.3 * yb.imag).backward()
        assert abs(float(xa.grad) - float(xb.grad)) <= 1e-8 * abs(float(xa.grad))


def test_transition_gradient_is_zero_on_the_clamped_side():
    """F clamps x <= 0, so its derivative there must be exactly zero.

    Without this the (large) small-argument slope leaks into every candidate
    path whose reflection point lies outside its facet, corrupting the whole
    gradient.
    """
    x = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    y = tf.transition_F(x)
    (y.real + y.imag).backward()
    assert float(x.grad) == 0.0


def test_edge_argument_is_signed():
    """The transition argument must distinguish inside from outside a facet."""
    L = torch.tensor(1.0, dtype=torch.float64)
    inside = tf.edge_argument(torch.tensor(0.3, dtype=torch.float64), L, 100.0)
    outside = tf.edge_argument(torch.tensor(-0.3, dtype=torch.float64), L, 100.0)
    assert float(inside) > 0 and float(outside) < 0
    assert abs(complex(tf.transition_F(outside))) == 0.0


# ---------------------------------------------------------------------------
# materials (Eq. 56-59)
# ---------------------------------------------------------------------------
def test_conductor_reflects_almost_totally():
    """A good conductor gives Gamma -> -1 for the TE component."""
    g = reflection_coefficient(5e9, torch.tensor(0.7, dtype=torch.float64),
                               get_material("metal"), "perp")
    assert abs(complex(g) + 1.0) < 1e-3


def test_grazing_incidence_reflects_totally():
    """Every material tends to Gamma = -1 at grazing incidence."""
    for name in ["concrete", "wood", "glass", "foam_board"]:
        g = reflection_coefficient(5e9, torch.tensor(1e-6, dtype=torch.float64),
                                   get_material(name), "perp")
        assert abs(complex(g) + 1.0) < 1e-3, name


def test_vacuum_like_material_is_transparent():
    """Foam has eps ~ 1, so a slab of it must transmit essentially everything."""
    t = slab_transmission(5e9, torch.tensor(1.0, dtype=torch.float64),
                          get_material("foam_board"), thickness=0.02)
    assert abs(abs(complex(t)) - 1.0) < 0.02


def test_penetration_loss_ordering():
    """Lossier materials must attenuate more at equal thickness.

    The ordering is by loss tangent, not by permittivity: at 5 GHz wood
    attenuates slightly more per metre than brick, because brick's larger
    eps' lowers alpha even though its conductivity is a little higher.  Only
    unambiguous pairs are asserted here.
    """
    loss = {m: float(get_material(m).penetration_loss_db(5e9, 0.1))
            for m in ["foam_board", "ceiling_board", "wood", "brick", "concrete"]}
    assert loss["foam_board"] < loss["ceiling_board"] < loss["brick"]
    assert loss["brick"] < loss["concrete"]
    assert loss["wood"] < loss["concrete"]


def test_material_gradients_flow():
    """d|Gamma| / d eps' must be finite and non-zero (Eq. 60)."""
    from rfdt.materials import MaterialParams
    p = MaterialParams.from_material(get_material("concrete"), 5e9)
    g = reflection_coefficient(5e9, torch.tensor(0.6, dtype=torch.float64),
                               get_material("concrete"), "perp", params=p)
    g.abs().backward()
    assert p.log_eps_real.grad is not None and float(p.log_eps_real.grad) != 0.0


# ---------------------------------------------------------------------------
# surface roughness (an addition to RFDT, not part of the paper)
# ---------------------------------------------------------------------------
def test_roughness_matches_the_closed_form():
    """The factor must equal exp(-2 (k sigma cos)^2) with no fitted constant."""
    from rfdt.materials import roughness_factor
    for f in (5e9, 60e9):
        for sig in (0.0002, 0.001, 0.002):
            for c in (0.2, 0.7, 1.0):
                got = float(roughness_factor(
                    f, torch.tensor(c, dtype=torch.float64), sig, "rayleigh"))
                k = 2 * np.pi * f / C0
                assert abs(got - np.exp(-2 * (k * sig * c) ** 2)) < 1e-14


def test_roughness_is_identity_for_a_smooth_surface():
    """sigma = 0 must reproduce the ideal Fresnel coefficient bit for bit.

    This is what keeps every closed-form check in this file meaningful: they
    are all derived for smooth interfaces, so the roughness correction has to
    disappear exactly, not merely nearly, when the surface is smooth.
    """
    from rfdt.materials import Material, roughness_factor
    for model in ("rayleigh", "miller_brown"):
        assert float(roughness_factor(
            60e9, torch.tensor(1.0, dtype=torch.float64), 0.0, model)) == 1.0
    smooth = Material("smooth_test", 5.24, 0.0, 0.0462, 0.7822)
    assert smooth.roughness_sigma == 0.0
    c = torch.tensor(0.7, dtype=torch.float64)
    a = reflection_coefficient(60e9, c, smooth, "perp")
    b = reflection_coefficient(60e9, c, smooth, "perp", roughness="none")
    assert complex(a) == complex(b)


def test_roughness_vanishes_at_grazing_incidence():
    """A wave skimming the surface cannot resolve its height variation.

    cos(theta) -> 0 drives the two-way phase difference between a bump and a
    trough to zero, so however rough the surface is the coherent field must be
    left untouched.  Without this limit the roughness term would fight the
    Gamma -> -1 grazing behaviour that the tracer relies on.
    """
    from rfdt.materials import roughness_factor
    for model in ("rayleigh", "miller_brown"):
        r = float(roughness_factor(
            60e9, torch.tensor(1e-9, dtype=torch.float64), 0.01, model))
        assert abs(r - 1.0) < 1e-12, model


def test_roughness_is_monotone_in_height_and_frequency():
    """More roughness, or more frequency, can only cost coherent power."""
    from rfdt.materials import roughness_factor
    c = torch.tensor(0.8, dtype=torch.float64)
    for model in ("rayleigh", "miller_brown"):
        by_sigma = [float(roughness_factor(60e9, c, s, model))
                    for s in (0.0, 0.0005, 0.001, 0.002, 0.004)]
        by_freq = [float(roughness_factor(f, c, 0.001, model))
                   for f in (1e9, 5e9, 28e9, 60e9, 140e9)]
        assert all(x > y for x, y in zip(by_sigma, by_sigma[1:])), model
        assert all(x > y for x, y in zip(by_freq, by_freq[1:])), model


def test_miller_brown_is_never_more_severe_than_rayleigh():
    """The two models must agree where Rayleigh holds and diverge only outside.

    Rayleigh assumes a small two-way phase variance; Miller-Brown adds back the
    energy a rough surface returns near the specular direction from favourably
    tilted facets.  So Miller-Brown is always the weaker attenuation, and the
    gap is negligible while the roughness parameter is small.  That is the
    stated basis for making it the default, so it is asserted rather than
    assumed.
    """
    from rfdt.materials import roughness_factor
    for sig in (0.0002, 0.001, 0.002):
        for c in np.linspace(0.05, 1.0, 20):
            ct = torch.tensor(float(c), dtype=torch.float64)
            for f in (5e9, 28e9, 60e9):
                ray = float(roughness_factor(f, ct, sig, "rayleigh"))
                mb = float(roughness_factor(f, ct, sig, "miller_brown"))
                assert mb >= ray - 1e-15
                # Size of the gap, from the series I0(x) = 1 + x^2/4 + ... with
                # x = 2 g^2: the two models differ by e^{-x}(I0(x) - 1), which
                # is bounded above by x^2/4 = g^4.  Asserting the derived bound
                # rather than a hand-picked tolerance means this stays a test
                # of the physics and not of a constant chosen to make it pass.
                g = 2 * np.pi * f / C0 * sig * c
                assert mb - ray <= g ** 4 + 1e-15, (f, sig, c, g)


def test_roughness_gradient_matches_finite_differences():
    """d|Gamma| / d sigma_h must be exact, since sigma_h is a fitted parameter."""
    from rfdt.materials import MaterialParams
    mat = get_material("concrete")
    cos_ti = torch.tensor(0.8, dtype=torch.float64)

    def params(sig):
        return MaterialParams(
            torch.log(torch.tensor(5.24, dtype=torch.float64)),
            torch.log(torch.tensor(0.15, dtype=torch.float64)),
            torch.log(torch.as_tensor(sig, dtype=torch.float64)))

    sig = torch.tensor(1e-3, dtype=torch.float64, requires_grad=True)
    reflection_coefficient(60e9, cos_ti, mat, "perp",
                           params=params(sig)).abs().backward()
    analytic = float(sig.grad)

    h = 1e-9
    def mag(x):
        return float(reflection_coefficient(
            60e9, cos_ti, mat, "perp", params=params(x)).abs())
    numeric = (mag(1e-3 + h) - mag(1e-3 - h)) / (2.0 * h)
    assert abs(analytic - numeric) / abs(numeric) < 1e-6, (analytic, numeric)


def test_roughness_preserves_reciprocity():
    """Roughness must not break the swap-the-ends invariant.

    The factor depends on the incidence angle, which is shared by the two
    directions of travel, so reciprocity should survive.  It is checked rather
    than argued because the wedge-face coefficients receive the same treatment
    and those are the terms where reciprocity has broken before.
    """
    mesh = scenes.furnished_room()
    cfg = TracerConfig(max_order=1, surface_roughness="miller_brown")
    a, b = (1.2, 1.2, 2.7), (4.5, 1.0, 0.9)
    fwd = RFDTracer(mesh, cfg).trace(
        Transmitter(a, 60e9, 20.0, Antenna("isotropic")),
        Receiver(b, Antenna("isotropic")))
    rev = RFDTracer(mesh, cfg).trace(
        Transmitter(b, 60e9, 20.0, Antenna("isotropic")),
        Receiver(a, Antenna("isotropic")))
    assert abs(float(fwd.power_dbm(20.0)) - float(rev.power_dbm(20.0))) < 1e-9


def test_roughness_regime_flags_where_the_missing_diffuse_term_dominates():
    """The rough-surface flag must use the classical criterion, not a guess.

    Once a surface is rough the tracer removes most of the specular energy and
    nothing re-radiates it, so the returned field is a residue of a quantity
    the simulator no longer represents.  Reporting that is the difference
    between a documented limitation and a wrong number presented as a
    prediction, so the boundary is pinned here.
    """
    from rfdt.materials import (RAYLEIGH_SMOOTH_LIMIT, roughness_regime,
                                roughness_factor)
    assert abs(RAYLEIGH_SMOOTH_LIMIT - np.pi / 4) < 1e-12
    c = torch.tensor(1.0, dtype=torch.float64)

    # At 5 GHz nothing in the library crosses the criterion, and the largest
    # specular loss anywhere in it stays under 1 dB.  That is the bound worth
    # asserting: brick already scatters about 16 % of its specular power at
    # 5 GHz, so a tighter claim about the energy fraction would be false, while
    # the loss in dB is what decides whether a link budget notices.
    for name in ["metal", "foam_board", "concrete", "brick", "human_body"]:
        r = roughness_regime(5e9, c, get_material(name).roughness_sigma)
        assert not r["is_rough"], (name, r["g"])
        assert r["specular_loss_db"] < 1.0, (name, r)

    # at 77 GHz the rough surfaces are flagged and the smooth ones are not
    for name in ["metal", "glass"]:
        assert not roughness_regime(77e9, c, get_material(name).roughness_sigma)["is_rough"]
    for name in ["concrete", "brick"]:
        r = roughness_regime(77e9, c, get_material(name).roughness_sigma)
        assert r["is_rough"], name
        assert r["unmodelled_fraction"] > 0.9, (name, r)

    # the flag must track the criterion exactly, either side of it
    k = 2 * np.pi * 77e9 / C0
    just_smooth = (RAYLEIGH_SMOOTH_LIMIT * 0.99) / k
    just_rough = (RAYLEIGH_SMOOTH_LIMIT * 1.01) / k
    assert not roughness_regime(77e9, c, just_smooth)["is_rough"]
    assert roughness_regime(77e9, c, just_rough)["is_rough"]


    # The human body entry carries no roughness on purpose: the model assumes a
    # planar interface and a body is not one.  Pinned so that a future edit
    # adding a plausible-looking value has to justify itself against this.
    assert get_material("human_body").roughness_sigma == 0.0

    # and the reported loss must be the same number the tracer actually applies
    r = roughness_regime(77e9, c, 0.001)
    rho = float(roughness_factor(77e9, c, 0.001))
    assert abs(r["specular_loss_db"] - (-20 * np.log10(rho))) < 1e-12


def test_roughness_is_negligible_at_5ghz_and_material_at_60ghz():
    """Pin the band dependence that motivates the correction existing at all.

    The point of the term is that it is invisible at 5 GHz and cannot be
    ignored at 60 GHz.  If a future change made it matter at 5 GHz, every
    previously reported 5 GHz result would silently move, so the boundary is
    asserted here.
    """
    from rfdt.materials import roughness_factor
    c = torch.tensor(0.7, dtype=torch.float64)
    for name in ["metal", "foam_board", "concrete", "plasterboard"]:
        sig = get_material(name).roughness_sigma
        loss_5 = -20.0 * np.log10(float(roughness_factor(5e9, c, sig)))
        assert loss_5 < 0.1, (name, loss_5)
    rough = get_material("concrete").roughness_sigma
    loss_60 = -20.0 * np.log10(float(roughness_factor(60e9, c, rough)))
    assert loss_60 > 3.0, loss_60


# ---------------------------------------------------------------------------
# geometry and mesh structure
# ---------------------------------------------------------------------------
def test_welding_produces_correct_wedge_indices():
    """A closed room must have no free edges; a box must expose convex ones."""
    room = scenes.empty_room()
    idx = [w.n_index for w in room.wedges()]
    assert max(idx) <= 1.0 + 1e-6, "room shell should have no half-plane edges"
    assert sum(1 for i in idx if abs(i - 0.5) < 1e-6) == 12   # concave corners

    furnished = scenes.furnished_room()
    convex = sum(1 for w in furnished.wedges() if abs(w.n_index - 1.5) < 1e-6)
    assert convex == 36, convex          # 12 edges on each of 3 boxes


def test_surfaces_merge_coplanar_triangles():
    """Facet grouping must hide the triangulation from the physics."""
    assert len(scenes.plate_scene().surfaces()) == 1
    assert len(scenes.empty_room().surfaces()) == 6
    assert len(scenes.furnished_room().surfaces()) == 24


def test_trajectory_validation_catches_collisions():
    """A route through the furniture must raise rather than silently mislead."""
    bad = torch.tensor([[0.5, 4.5, 0.9]], dtype=torch.float64)   # inside cabinet
    try:
        scenes.validate_trajectory(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("expected a collision to be reported")
    scenes.validate_trajectory(scenes.survey_trajectory(n_samples=16).positions)


# ---------------------------------------------------------------------------
# forward model against closed-form references
# ---------------------------------------------------------------------------
def _free_space_tracer():
    """A scene whose only facet is far away, so only the LoS path matters."""
    mesh = scenes.plate_scene(center=(0.0, 0.0, -50.0))
    return RFDTracer(mesh, TracerConfig(max_order=0, enable_diffraction=False))


def test_line_of_sight_matches_friis():
    """The LoS path must reproduce the Friis transmission equation exactly."""
    tx = Transmitter((3.0, 2.5, 2.7), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((1.0, 1.0, 0.9), Antenna("isotropic"))
    got = float(_free_space_tracer().trace(tx, rx).power_dbm(20.0))
    d = float((rx.position - tx.position).norm())
    friis = 20.0 - 20.0 * np.log10(4.0 * np.pi * d * 5e9 / C0)
    assert abs(got - friis) < 1e-3, (got, friis)


def test_two_ray_matches_analytic_ground_reflection():
    """Tx + a large conducting plate must equal the analytic two-ray sum."""
    mesh = scenes.plate_scene(plate_size=(20.0, 20.0), material="metal")
    tx = Transmitter((0.0, 0.0, 2.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((3.0, 0.0, 1.0), Antenna("isotropic"))
    p = RFDTracer(mesh, TracerConfig(max_order=1, enable_diffraction=False)).trace(tx, rx)
    assert p.n_paths() == 2                       # one LoS, one facet reflection

    lam = C0 / 5e9
    k = 2 * np.pi / lam
    l_d = np.sqrt(3.0 ** 2 + 1.0 ** 2)
    l_r = np.sqrt(3.0 ** 2 + 3.0 ** 2)
    e = (lam / (4 * np.pi * l_d) * np.exp(-1j * k * l_d)
         - lam / (4 * np.pi * l_r) * np.exp(-1j * k * l_r))
    ref = 20.0 + 20.0 * np.log10(abs(e))
    assert abs(float(p.power_dbm(20.0)) - ref) < 0.05


def test_reciprocity():
    """Swapping transmitter and receiver must not change the path loss."""
    mesh = scenes.furnished_room()
    cfg = TracerConfig(max_order=1)
    a = (1.2, 1.2, 2.7)
    b = (4.5, 1.0, 0.9)
    fwd = RFDTracer(mesh, cfg).trace(
        Transmitter(a, 5e9, 20.0, Antenna("isotropic")),
        Receiver(b, Antenna("isotropic")))
    rev = RFDTracer(mesh, cfg).trace(
        Transmitter(b, 5e9, 20.0, Antenna("isotropic")),
        Receiver(a, Antenna("isotropic")))
    assert abs(float(fwd.power_dbm(20.0)) - float(rev.power_dbm(20.0))) < 0.05


# ---------------------------------------------------------------------------
# the RFDT claims: continuity and differentiability
# ---------------------------------------------------------------------------
def _edge_sweep(weighting, diffraction, n=401):
    """Scattered field as the specular point walks across a plate edge."""
    mesh = scenes.plate_scene(plate_size=(1.0, 1.0), material="metal")
    tx = Transmitter((0.0, 0.0, 1.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((0.0, 0.0, 1.0), Antenna("isotropic"))
    xs = torch.linspace(0.75, 1.25, n, dtype=torch.float64)   # edge at x = 1.0
    pos = torch.stack([xs, torch.zeros_like(xs), torch.ones_like(xs)], -1)
    cfg = TracerConfig(max_order=1, weighting=weighting,
                       enable_diffraction=diffraction, sigmoid_k=40.0)
    p = RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=pos)
    keep = [i for i, k in enumerate(p.kind) if k != "los"]
    return p.gain[:, keep].sum(-1).abs()


def test_heaviside_is_discontinuous_at_an_edge():
    """The conventional binary test steps by ~the whole reflected field."""
    e = _edge_sweep("heaviside", False)
    jump = float((e[1:] - e[:-1]).abs().max()) / float(e.max() - e.min())
    assert jump > 0.5, jump


def test_rfdt_is_continuous_at_an_edge():
    """The Eq. 11 weight plus diffraction keeps the field continuous."""
    e = _edge_sweep("rfdt", True)
    jump = float((e[1:] - e[:-1]).abs().max()) / float(e.max() - e.min())
    assert jump < 0.05, jump


def test_rfdt_specular_vanishes_outside_the_facet():
    """F(x) = 0 outside a facet, so no energy leaks past the edge."""
    mesh = scenes.plate_scene(plate_size=(1.0, 1.0), material="metal")
    tx = Transmitter((0.0, 0.0, 1.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((0.0, 0.0, 1.0), Antenna("isotropic"))
    pos = torch.tensor([[1.6, 0.0, 1.0]], dtype=torch.float64)   # well past the edge
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=False)
    p = RFDTracer(mesh, cfg).trace(tx, rx, rx_positions=pos)
    refl = [i for i, k in enumerate(p.kind) if k.startswith("refl")]
    assert float(p.gain[:, refl].abs().max()) == 0.0


def test_sigmoid_baseline_leaks_energy():
    """The soften-triangle baseline is smooth but biased (Eq. 4)."""
    e = _edge_sweep("sigmoid", False)
    assert float(e[-1]) > 1e-6      # still radiating far outside the plate


def test_gradients_match_finite_differences():
    """Autograd through the whole pipeline must match central differences."""
    mesh = scenes.plate_scene(plate_size=(1.0, 1.0), material="metal")
    rx = Receiver((1.1, 0.0, 1.0), Antenna("isotropic"))
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=True)

    def field(theta):
        """Field magnitude for a given vertical translation of the transmitter."""
        z = torch.zeros((), dtype=torch.float64)
        pos = torch.stack([z, z, 1.0 + theta])
        tx = Transmitter(pos, 5e9, 20.0, Antenna("isotropic"))
        return RFDTracer(mesh, cfg).trace(tx, rx).field().abs().squeeze()

    for th0 in [-0.20, -0.05, 0.0, 0.05, 0.20, 0.45]:
        t = torch.tensor(th0, dtype=torch.float64, requires_grad=True)
        y = field(t)
        y.backward()
        h = 1e-6
        fd = (float(field(torch.tensor(th0 + h, dtype=torch.float64)))
              - float(field(torch.tensor(th0 - h, dtype=torch.float64)))) / (2 * h)
        assert abs(float(t.grad) - fd) <= 1e-6 * max(abs(fd), 1e-9), (th0, float(t.grad), fd)


def test_gradient_flows_to_geometry():
    """Vertex positions must receive a non-zero gradient (Eq. 10)."""
    mesh = scenes.plate_scene(plate_size=(1.0, 1.0), material="metal")
    v = mesh.vertices.clone().requires_grad_(True)
    tx = Transmitter((0.0, 0.0, 1.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((0.8, 0.1, 1.0), Antenna("isotropic"))
    cfg = TracerConfig(max_order=1, weighting="rfdt", enable_diffraction=False)
    RFDTracer(mesh, cfg).trace(tx, rx, vertices=v).field().abs().backward()
    assert v.grad is not None and float(v.grad.abs().max()) > 0.0


def test_candidate_pruning_is_lossless():
    """Widening the search margin must not change the answer."""
    mesh = scenes.furnished_room()
    traj = scenes.survey_trajectory(n_samples=12)
    tx = antennas.wifi_ap((1.2, 1.2, 2.7), 5e9, 20.0)
    rx = antennas.robot_client(traj.positions[0])
    out = []
    for margin in (0.25, 4.0):
        tr = RFDTracer(mesh, TracerConfig(max_order=1, prune_margin=margin))
        p = tr.trace(tx, rx, rx_positions=traj.positions)
        out.append(p.power_dbm(20.0))
    assert float((out[0] - out[1]).abs().max()) < 1e-9


# ---------------------------------------------------------------------------
# signal domain (Eq. 18-22, 27)
# ---------------------------------------------------------------------------
def test_range_profile_peaks_at_the_true_range():
    """An FMCW range profile must place a target at its geometric range."""
    from rfdt.signal import FMCWConfig, range_profile_fft
    cfg = FMCWConfig()
    r_true = 3.0
    delays = torch.tensor([[2 * r_true / C0]], dtype=torch.float64)
    amps = torch.ones(1, 1, dtype=torch.complex128)
    prof = range_profile_fft(delays, amps, cfg).abs().squeeze()
    r_hat = float(cfg.range_axis()[int(prof.argmax())])
    assert abs(r_hat - r_true) < cfg.range_resolution


def test_surrogate_is_smoother_than_the_exact_transform():
    """The Dirichlet surrogate must have less high-frequency ripple (Fig. 4)."""
    from rfdt.signal import FMCWConfig, range_profile_fft, surrogate_range_profile
    cfg = FMCWConfig()
    delays = torch.tensor([[2 * 3.0 / C0, 2 * 3.4 / C0]], dtype=torch.float64)
    amps = torch.tensor([[1.0 + 0j, 0.6 + 0j]], dtype=torch.complex128)
    # compare against the rectangular window, which is the one the Dirichlet
    # kernel models, so the difference isolates the phase-agnostic smoothing
    exact = range_profile_fft(delays, amps, cfg, window="rect").abs().squeeze()
    surr = surrogate_range_profile(delays, amps, cfg).squeeze()

    def roughness(x):
        """Total variation of a peak-normalised profile, a smoothness proxy."""
        x = x / x.max()
        return float((x[1:] - x[:-1]).abs().sum())
    assert roughness(surr) < roughness(exact)


def test_anneal_schedule_is_monotone_and_reaches_one():
    """lambda(t) must rise from 0 to exactly 1 so the result is unbiased."""
    from rfdt.signal import anneal_schedule
    vals = [anneal_schedule(e, 100, 0.4) for e in range(100)]
    assert vals[0] == 0.0 and abs(vals[-1] - 1.0) < 1e-9
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_two_ray_notch_spacing_matches_theory():
    """Frequency-domain check: notch spacing of a two-ray channel is c / dL.

    A single reflector makes the transfer function a comb of notches whose
    spacing depends only on the path-length difference.  This exercises the
    frequency-domain path (``ofdm_channel``) against closed-form analysis, and
    is sensitive to any error in path length or phase.
    """
    from rfdt.signal import ofdm_channel
    mesh = scenes.plate_scene(plate_size=(30.0, 30.0), material="metal")
    tx = Transmitter((0.0, 0.0, 2.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((3.0, 0.0, 1.0), Antenna("isotropic"))
    p = RFDTracer(mesh, TracerConfig(max_order=1,
                                     enable_diffraction=False)).trace(tx, rx)
    assert p.n_paths() == 2

    freqs = torch.linspace(4e9, 6e9, 4001, dtype=torch.float64)
    h = ofdm_channel(p.delay, p.gain, freqs).squeeze().abs().numpy()
    interior = np.arange(1, len(h) - 1)
    minima = interior[(h[1:-1] < h[:-2]) & (h[1:-1] < h[2:])]
    spacing = float(np.mean(np.diff(freqs.numpy()[minima])))

    l_direct = np.sqrt(3.0 ** 2 + 1.0 ** 2)
    l_reflect = np.sqrt(3.0 ** 2 + 3.0 ** 2)
    expected = C0 / (l_reflect - l_direct)
    assert abs(spacing - expected) / expected < 0.01, (spacing, expected)


def test_group_delay_equals_propagation_delay():
    """The phase slope of a single-path channel must equal its delay."""
    from rfdt.signal import ofdm_channel
    mesh = scenes.plate_scene(center=(0.0, 0.0, -60.0))
    tx = Transmitter((0.0, 0.0, 2.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((3.0, 0.0, 1.0), Antenna("isotropic"))
    p = RFDTracer(mesh, TracerConfig(max_order=0,
                                     enable_diffraction=False)).trace(tx, rx)
    freqs = torch.linspace(4.9e9, 5.1e9, 201, dtype=torch.float64)
    h = ofdm_channel(p.delay, p.gain, freqs).squeeze().numpy()
    phase = np.unwrap(np.angle(h))
    gd = -np.gradient(phase, 2.0 * np.pi * freqs.numpy())
    assert abs(float(np.mean(gd)) - float(p.delay.squeeze())) < 1e-15


def test_doppler_sign_convention():
    """A receiver approaching the transmitter must give a positive shift."""
    mesh = scenes.plate_scene(center=(0.0, 0.0, -50.0))
    tx = Transmitter((0.0, 0.0, 0.0), 5e9, 20.0, Antenna("isotropic"))
    rx = Receiver((5.0, 0.0, 0.0), Antenna("isotropic"))
    p = RFDTracer(mesh, TracerConfig(max_order=0, enable_diffraction=False)).trace(tx, rx)
    v = torch.tensor([[-1.0, 0.0, 0.0]], dtype=torch.float64)     # moving towards tx
    df = RFDTracer.doppler(p, C0 / 5e9, v_rx=v)
    assert float(df.squeeze()) > 0


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def main() -> int:
    """Run every ``test_*`` function in this module and report the result."""
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}: {exc}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())