# DLoc comparison

Validating the simulator against real measurements, using the DLoc / WILD
dataset from UCSD WCSNG.

The question: at 5 GHz, with a robot moving through the DLoc environments, is
the simulated channel close enough to the measured one to support sim2real
work? If it is not, that is a useful result and motivates a different
simulator for sub-6 GHz.

**Both outcomes are real.** [`PROTOCOL.md`](PROTOCOL.md) fixes the comparison
metric, the success thresholds and the material assumptions *before* any
measurement file is opened, so that a negative result means something.

---

## Status

| Step | State |
|---|---|
| Comparison protocol written and fixed | done |
| Dataset loader against the documented schema | done, untested against real files |
| Scene builder from measured AP and route coordinates | done |
| Computational feasibility measured | done, see below |
| Measurement files downloaded | **blocked, needs consent form** |
| Atkinson comparison run | not started |
| Jacobs comparison run | not started |

### Blocked on the data

The WILD measurements are not redistributed here. They need a consent form and
a password-protected download, linked from
[wild.md](https://github.com/ucsdwcsng/DLoc_pt_code/blob/main/wild.md). Place
`channels_<setup>.mat` files in `dloc/data/`.

`python3 -c "import dataset; print(dataset.describe_availability())"` from this
directory reports what is present.

**Nothing in this directory fabricates geometry.** A scene can only be built
from a loaded measurement, because AP coordinates and robot positions come
from the dataset. There is no placeholder fallback, deliberately: a comparison
against real measurements is worthless if the geometry was guessed.

---

## Is it computationally practical?

Yes, comfortably. Measured with `python3 dloc/benchmark.py` on synthetic rooms
at DLoc's dimensions:

| Scene | Surfaces | Order | Per position | One setup | All 8 setups |
|---|---|---|---|---|---|
| Empty, 18 x 8 m | 6 | 2 | 0.30 ms | 15 s | 2.0 min |
| Furnished, 8 x 5 m | 30 | 2 | 27.5 ms | 17.2 min | 2.3 h |
| Furnished, 18 x 8 m | 30 | 2 | 22.0 ms | 18.3 min | 2.4 h |

So the **full 12,500-point route can be run without subsampling**, which
matters beyond convenience: there is no need to choose which measured points
to include, and therefore no opportunity to choose the ones that happen to
agree.

The empty-room rows are not the honest estimate. A closed box has no convex
edges, so the diffraction stage does no work at all there. The furnished rows,
with 30 surfaces and 48 diffracting edges, are the realistic figure. Real
furniture counts will differ from these guesses.

---

## What the dataset provides

Two environments, eight setups, roughly 12,500 points each.

| | Atkinson | Jacobs |
|---|---|---|
| Size | 8 x 5 m, 500 sq ft | 18 x 8 m, 1500 sq ft |
| APs | 3 | 4 |
| Character | mostly LOS | high multipath, NLOS |
| Setups | July16, July18, July22_2_ref | jacobs_Jul28, jacobs_Jul28_2, jacobs_Aug16_1, jacobs_Aug16_3, jacobs_Aug16_4_ref |

Per setup: complex CSI `[n_points x n_freq x n_ant x n_ap]`, RSSI, ground-truth
XY from SLAM, subcarrier frequencies, AP antenna coordinates and AoA rotation
offsets. Four antennas per AP. Ground truth from MapFind, a LIDAR and odometry
SLAM robot moving at 15 cm/s with a channel estimate every 50 ms.

---

## Why raw CSI cannot be compared directly

This is the crux of the whole exercise, and it is a property of WiFi hardware
rather than of the simulator:

1. **Time-of-flight offset.** Each packet carries an unknown timing offset.
   Removing it is the entire purpose of DLoc's consistency decoder. Absolute
   path delay is not comparable.
2. **Carrier and sampling frequency offset.** CSI phase is randomised per
   packet. Raw phase is not comparable.
3. **Automatic gain control.** CSI magnitude is in arbitrary units. RSSI is
   the calibrated amplitude quantity.

So the comparison uses RSSI, the *shape* of `|H(f)|` across subcarriers,
spatial correlation of RSS along the route, and AoA after applying the
dataset's rotation offsets. Full reasoning and the success thresholds are in
[`PROTOCOL.md`](PROTOCOL.md).

---

## Files

```
PROTOCOL.md      the comparison plan, fixed before the data was opened
dataset.py       loader for channels_<setup>.mat (MATLAB v7.3 / HDF5)
environments.py  builds a scene from a loaded measurement
benchmark.py     computational feasibility, runnable without the data
test_dloc.py     regression tests for this directory
data/            measurement files go here, not in version control
```

Requires `h5py` in addition to the root `requirements.txt`.

---

## Order of work

1. **Atkinson first.** Smaller, 3 APs, mostly LOS, far less sensitive to the
   guessed materials. If the simulator cannot match the simple environment,
   the complex one is not worth running.
2. **Jacobs second.** The real test, and where the screen wall and the metal
   wall near AP4 matter.
3. **July16 against July18** first of all. Same setup, different days, so the
   difference between them measures how much the *measurement* varies when
   nothing in the simulation changes. That sets a floor on how well any
   simulator could possibly do, and it is worth knowing before judging the
   simulator against anything.

---

## Reference

Ayyalasomayajula, Arun, Wu, Bharadia. *Deep Learning based Wireless
Localization for Indoor Navigation.* MobiCom 2020.
[paper](https://wcsng.ucsd.edu/files/dloc.pdf) ·
[project](https://wcsng.ucsd.edu/dloc/) ·
[code and dataset](https://github.com/ucsdwcsng/DLoc_pt_code)