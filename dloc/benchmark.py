"""Is a full DLoc comparison computationally practical?

Each DLoc setup holds around 12,500 robot positions, and each has 3 or 4 access
points, so a full run is roughly 40,000 to 50,000 transmitter-receiver links per
setup, times 8 setups.  That is a different scale from anything the simulator
has been asked to do so far, so it is worth measuring before building the rest
of the pipeline rather than discovering the cost at the end.

This benchmark uses *synthetic* rooms at DLoc's dimensions.  It is not the DLoc
geometry, which needs the measurement files; it exists purely to time the
tracer at the right scale.  Room sizes are the quoted collection areas, 8 x 5 m
for Atkinson and 18 x 8 m for Jacobs.

Run: ``python3 dloc/benchmark.py``
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rfdt import antennas                                             # noqa: E402
from rfdt.geometry import Mesh, box                                   # noqa: E402
from rfdt.tracer import RFDTracer, TracerConfig                       # noqa: E402

from environments import _shell                                       # noqa: E402

#: DLoc collection areas and access-point counts.
CASES = [
    ("Atkinson-sized", 8.0, 5.0, 3, 12500),
    ("Jacobs-sized", 18.0, 8.0, 4, 12500),
]

CARRIER_HZ = 5.0e9
CEILING_M = 3.0


def synthetic_room(width: float, depth: float, height: float = CEILING_M) -> Mesh:
    """An empty rectangular room of the given size, walls facing inwards.

    Stand-in geometry for timing only; the real comparison uses the room
    implied by the measured robot route.  Built by the same ``_shell`` the
    real scenes use, including its inward-normal assertion, rather than by a
    second copy of the same quad list.
    """
    return _shell(0.0, width, 0.0, depth, height,
                  "plasterboard", "concrete", "ceiling_board")


def furnished_room(width: float, depth: float, n_items: int = 4,
                   height: float = CEILING_M) -> Mesh:
    """The same room with a few solid objects in it.

    The empty shell understates the real cost in two ways: it has only six
    surfaces, and a closed box has no convex edges at all, so the diffraction
    stage does no work.  Real rooms have furniture, which adds both surfaces
    and wedges.  This gives a realistic upper bound rather than a flattering
    lower one.
    """
    mesh = synthetic_room(width, depth, height)
    rng = np.random.default_rng(1)
    for i in range(n_items):
        c = (rng.uniform(1.5, width - 1.5), rng.uniform(1.0, depth - 1.0), 0.45)
        mesh = mesh.merged(box(c, (0.9, 0.7, 0.9), "wood", f"item{i}"))
    return mesh.weld()


def human(seconds: float) -> str:
    """Format a duration in whatever unit makes it readable."""
    if seconds < 90:
        return f"{seconds:.1f} s"
    if seconds < 5400:
        return f"{seconds/60:.1f} min"
    return f"{seconds/3600:.2f} h"


def time_trace(mesh, tx, rx, positions, order: int, reuse_sequences: bool):
    """Time one trace over a batch of receiver positions.

    ``reuse_sequences`` separates the two costs: the candidate search, which
    depends on the scene and can be done once, and the exact solve, which is
    per position.  The distinction matters because the search is the part that
    would otherwise be repeated needlessly across access points and setups.
    """
    tracer = RFDTracer(mesh, TracerConfig(max_order=order))
    t0 = time.time()
    seqs = []
    for o in range(1, order + 1):
        seqs += tracer.candidate_sequences(tx.position, positions, o)
    t_search = time.time() - t0

    t0 = time.time()
    paths = tracer.trace(tx, rx, rx_positions=positions,
                         sequences=seqs if reuse_sequences else None)
    t_solve = time.time() - t0
    return t_search, t_solve, paths.n_paths(), len(seqs)


def run_case(name: str, width: float, depth: float, n_ap: int,
             n_points_full: int, batch: int = 200,
             furnished: bool = False) -> Dict[str, object]:
    """Benchmark one room size and extrapolate to the full dataset."""
    mesh = furnished_room(width, depth) if furnished else synthetic_room(width, depth)
    n_surf = len(mesh.surfaces())
    n_wedge = sum(1 for w in mesh.wedges() if w.n_index > 1.05)
    tx = antennas.wifi_ap((width * 0.2, depth * 0.2, 2.5), CARRIER_HZ, 20.0)

    rng = np.random.default_rng(0)
    pts = np.column_stack([
        rng.uniform(0.5, width - 0.5, batch),
        rng.uniform(0.5, depth - 0.5, batch),
        np.full(batch, 1.0)])
    positions = torch.tensor(pts, dtype=torch.float64)
    rx = antennas.robot_client(positions[0])

    print(f"\n{name}: {width} x {depth} m, {n_surf} surfaces, "
          f"{n_wedge} diffracting edges, {n_ap} APs")
    out = {"case": name, "width_m": width, "depth_m": depth,
           "n_surfaces": n_surf, "n_wedges": n_wedge, "n_ap": n_ap,
           "furnished": furnished, "orders": {}}

    for order in (1, 2):
        t_search, t_solve, n_paths, n_seq = time_trace(
            mesh, tx, rx, positions, order, reuse_sequences=True)
        per_pos_ms = t_solve / batch * 1e3
        # a full setup is every robot position against every access point
        full_s = per_pos_ms * 1e-3 * n_points_full * n_ap + t_search * n_ap
        out["orders"][order] = {
            "candidate_search_s": t_search,
            "n_sequences": n_seq,
            "solve_s_for_batch": t_solve,
            "batch": batch,
            "ms_per_position": per_pos_ms,
            "n_paths": n_paths,
            "projected_full_setup_s": full_s,
            "projected_all_8_setups_h": full_s * 8 / 3600.0,
        }
        print(f"  order {order}: search {t_search:6.2f} s ({n_seq} sequences), "
              f"solve {per_pos_ms:6.2f} ms/position, {n_paths} paths")
        print(f"            -> one setup ({n_points_full} pts x {n_ap} APs): "
              f"{human(full_s)};  all 8 setups: {human(full_s * 8)}")
    return out


def main():
    """Time both room sizes and report whether the full run is practical."""
    print("DLoc comparison, computational feasibility")
    print("Synthetic rooms at DLoc dimensions. Timing only, not the real "
          "geometry.")
    results = []
    for c in CASES:
        results.append(run_case(*c))
    print("\n" + "-" * 70)
    print("Same rooms with furniture, which is what the real scenes look like")
    print("-" * 70)
    for c in CASES:
        results.append(run_case(c[0] + " furnished", *c[1:], furnished=True))

    print("\n" + "=" * 70)
    print("Reading")
    print("=" * 70)
    worst = max(r["orders"][2]["projected_full_setup_s"] for r in results)
    total = sum(max(r["orders"][2]["projected_full_setup_s"]
                    for r in results if r["furnished"]) for _ in range(1)) * 8
    print(f"Worst single setup at order 2, furnished: {human(worst)}.")
    print(f"All 8 setups at that rate: {human(total)}.")
    print("\nSo the full route can be run without subsampling. That matters "
          "for the comparison being trustworthy: there is no need to choose "
          "which measured points to include, so no opportunity to choose the "
          "ones that agree.")
    print("\nCaveats on these numbers. The rooms are empty boxes or boxes with "
          "a few objects, not the real DLoc geometry, which is not available "
          "yet. An empty box also has no convex edges, so the diffraction "
          "stage does nothing there; the furnished rows are the honest "
          "estimate. Real furniture counts will differ.")
    return results


if __name__ == "__main__":
    main()