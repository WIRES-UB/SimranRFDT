"""Run the full RFDT study: tests first, then all six experiments.

Usage::

    python3 run_all.py                 # tests, then every experiment
    python3 run_all.py --skip-tests    # experiments only
    python3 run_all.py 1 3             # only experiments 1 and 3

The regression tests run first by default and the run stops if they fail:
every experiment's numbers depend on the physics the tests pin down, so
producing figures from a broken forward model would be worse than producing
nothing.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

#: Experiment number -> (module name, one-line description).
EXPERIMENTS = {
    1: ("exp1_differentiability",
        "verify continuity and gradients of the visibility term (Sec. 6.1)"),
    2: ("exp2_material_sweep",
        "indoor robot channel versus wall material, 5 and 60 GHz"),
    3: ("exp3_robot_radar",
        "robot-mounted 77 GHz FMCW radar versus material"),
    4: ("exp4_inverse_material",
        "recover material and geometry from RF measurements (Sec. 5.2)"),
    5: ("exp5_frequency_response",
        "channel frequency response: amplitude, phase and coherence bandwidth"),
    6: ("exp6_metal_vs_foam",
        "metal against foam head to head, the two extremes of experiment 2"),
}


def run_tests() -> bool:
    """Execute the regression suite; return True when everything passes."""
    sys.path.insert(0, os.path.join(HERE, "tests"))
    import test_rfdt
    print("=" * 78)
    print("Regression tests")
    print("=" * 78)
    return test_rfdt.main() == 0


def run_experiment(number: int) -> None:
    """Import and run one experiment module by number."""
    name, desc = EXPERIMENTS[number]
    print("\n" + "=" * 78)
    print(f"Experiment {number}: {desc}")
    print("=" * 78)
    sys.path.insert(0, os.path.join(HERE, "experiments"))
    module = __import__(name)
    t0 = time.time()
    module.main()
    print(f"[experiment {number} finished in {time.time() - t0:.0f} s]")


def main() -> int:
    """Parse arguments, run the requested work, and report total runtime."""
    ap = argparse.ArgumentParser(description=__doc__)
    # `choices` is deliberately not used here: argparse validates the default
    # of a `nargs="*"` positional against choices, so the no-argument case
    # (meaning "run everything") would be rejected.  Validate by hand instead.
    ap.add_argument("experiments", nargs="*", type=int,
                    help=f"experiment numbers to run, from "
                         f"{sorted(EXPERIMENTS)} (default: all)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="do not run the regression suite first")
    args = ap.parse_args()

    unknown = [n for n in args.experiments if n not in EXPERIMENTS]
    if unknown:
        ap.error(f"unknown experiment(s) {unknown}; choose from "
                 f"{sorted(EXPERIMENTS)}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    t0 = time.time()

    if not args.skip_tests:
        if not run_tests():
            print("\nRegression tests failed; refusing to generate results from "
                  "a forward model that does not pass its physics checks.")
            return 1

    for n in (args.experiments or sorted(EXPERIMENTS)):
        run_experiment(n)

    print(f"\nAll done in {time.time() - t0:.0f} s. "
          f"Results are in {os.path.join(HERE, 'results')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())