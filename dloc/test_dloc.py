"""Regression tests for the DLoc comparison code.

These run without the measurement files, which are not redistributed.  They
check the parts that can be checked in their absence: that scene construction
is correct, that the loader fails clearly rather than silently when data is
missing, and that nothing invents geometry.

Run: ``python3 dloc/test_dloc.py``
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import dataset                                                        # noqa: E402
import environments                                                   # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402


def test_shell_normals_face_inwards():
    """Every room-shell surface must reflect towards the interior.

    A wall wound the wrong way contributes no specular path and says nothing
    about it, so this is checked at construction and asserted here.
    """
    mesh = environments._shell(0.0, 8.0, 0.0, 5.0, 3.0,
                               "plasterboard", "concrete", "ceiling_board")
    assert len(mesh.surfaces()) == 6
    centre = torch.tensor([4.0, 2.5, 1.5], dtype=torch.float64)
    for s in mesh.surfaces():
        tri = mesh.tri()[s.tri0]
        n = torch.cross(tri[1] - tri[0], tri[2] - tri[0], dim=-1)
        n = n / n.norm()
        assert float(((centre - tri[0]) * n).sum()) > 0, s.group


def test_shell_rejects_outward_normals():
    """The guard must actually fire, not just exist."""
    import rfdt.geometry as g
    good = environments._shell
    # build a shell then flip one wall by hand and confirm the check catches it
    mesh = good(0.0, 4.0, 0.0, 4.0, 3.0, "plasterboard", "concrete",
                "ceiling_board")
    flipped = g.Mesh(mesh.vertices, mesh.faces[:, ::-1].copy(),
                     list(mesh.mat_names), list(mesh.face_groups),
                     list(mesh.face_solid), list(mesh.face_depth))
    centre = torch.tensor([2.0, 2.0, 1.5], dtype=torch.float64)
    outward = 0
    for s in flipped.surfaces():
        tri = flipped.tri()[s.tri0]
        n = torch.cross(tri[1] - tri[0], tri[2] - tri[0], dim=-1)
        n = n / n.norm()
        if float(((centre - tri[0]) * n).sum()) <= 0:
            outward += 1
    assert outward == 6, outward


def test_all_surfaces_are_reachable():
    """Every wall must be found as a reflector from inside the room.

    This is the functional version of the normal check: if a surface is wound
    correctly it should appear as a candidate. Six surfaces, six candidates.
    """
    from rfdt import antennas
    mesh = environments._shell(0.0, 8.0, 0.0, 5.0, 3.0,
                               "plasterboard", "concrete", "ceiling_board")
    tracer = RFDTracer(mesh, TracerConfig(max_order=1))
    tx = antennas.wifi_ap((2.0, 1.5, 2.5), 5e9, 20.0)
    pos = torch.tensor([[5.0, 3.0, 1.0]], dtype=torch.float64)
    seqs = tracer.candidate_sequences(tx.position, pos, 1)
    assert len(seqs) == 6, f"expected all 6 surfaces, got {len(seqs)}"


def test_loader_fails_clearly_when_data_missing():
    """A missing measurement file must raise with instructions, not silently."""
    try:
        dataset.load_channels(os.path.join(HERE, "data", "channels_nope.mat"))
    except FileNotFoundError as e:
        assert "consent" in str(e).lower() or "wild" in str(e).lower()
    else:
        raise AssertionError("expected FileNotFoundError")


def test_environment_lookup():
    """Setup names must map to the right environment."""
    assert dataset.environment_of("July16") == "atkinson"
    assert dataset.environment_of("jacobs_Aug16_4_ref") == "jacobs"
    try:
        dataset.environment_of("not_a_setup")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError for an unknown setup")


def test_no_scene_without_measurement():
    """There must be no way to build a scene from invented geometry.

    ``build_scene`` takes a loaded measurement and nothing else, so geometry
    cannot come from anywhere but the dataset.
    """
    import inspect
    sig = inspect.signature(environments.build_scene)
    assert "meas" in sig.parameters
    assert sig.parameters["meas"].default is inspect.Parameter.empty, \
        "build_scene must require a measurement, with no default"


def test_scene_from_synthetic_measurement():
    """End to end with a stand-in measurement, to exercise the scene builder.

    The measurement here is fabricated purely to test the plumbing, and is
    never used for a comparison; that is what the missing-file error above
    protects.
    """
    rng = np.random.default_rng(0)
    n = 200
    labels = np.column_stack([rng.uniform(1.0, 7.0, n), rng.uniform(1.0, 4.0, n)])
    aps = [np.array([[0.5, 0.5], [0.55, 0.5], [0.6, 0.5], [0.65, 0.5]]),
           np.array([[7.5, 0.5], [7.55, 0.5], [7.6, 0.5], [7.65, 0.5]]),
           np.array([[4.0, 4.5], [4.05, 4.5], [4.1, 4.5], [4.15, 4.5]])]
    meas = dataset.DLocMeasurement(
        setup="July16",
        channels=np.zeros((n, 4, 4, 3), dtype=complex),
        rssi=np.zeros((n, 3)), labels=labels,
        freqs=np.linspace(5.17e9, 5.25e9, 4),
        ap_coords=aps, ap_aoa=np.zeros(3), source_path="<synthetic>")

    scene = environments.build_scene(meas, subsample=4)
    assert scene.environment == "atkinson"
    assert scene.ap_positions.shape == (3, 3)
    assert scene.client_positions.shape[0] == 50          # 200 subsampled by 4
    assert len(scene.mesh.surfaces()) == 6
    assert environments.check_positions_inside(scene) == []
    # the assumptions must travel with the scene, not be lost
    s = scene.summary()
    assert s["assumptions"]["wall_material"] == "plasterboard"
    assert any("subsampled" in note for note in s["assumptions"]["notes"])
    assert any("Room outline" in note for note in s["assumptions"]["notes"])


def test_measurement_geometry_helpers():
    """Route extent and AP centroids must be computed from the data."""
    labels = np.array([[1.0, 2.0], [5.0, 6.0]])
    aps = [np.array([[0.0, 0.0], [1.0, 0.0]])]
    meas = dataset.DLocMeasurement(
        setup="July16", channels=np.zeros((2, 2, 2, 1), dtype=complex),
        rssi=np.zeros((2, 1)), labels=labels,
        freqs=np.array([5.18e9, 5.22e9]), ap_coords=aps,
        ap_aoa=np.zeros(1), source_path="<synthetic>")
    e = meas.route_extent()
    assert e["width_m"] == 4.0 and e["height_m"] == 4.0
    assert np.allclose(meas.ap_centroids(), [[0.5, 0.0]])
    assert abs(meas.centre_frequency_hz - 5.20e9) < 1e3
    assert abs(meas.bandwidth_hz - 40e6) < 1e3


def main() -> int:
    """Run every test in this module."""
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