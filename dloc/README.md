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
| Floor plan extracted from the published occupancy maps | done |
| Scene builder from the real outline, with screens and column | done |
| Computational feasibility measured on the real geometry | done, see below |
| Pixels-to-metres scale | **blocked, needs the dataset** |
| Measurement files downloaded | **blocked, needs consent form** |
| Atkinson comparison run | not started |
| Jacobs comparison run | not started |

### The floor plans are already available

The occupancy grids and the annotated photographs are public in the DLoc code
repository under `ref/`, no consent form needed. They are mirrored in `maps/`:

- `jacobs_default.png`, `jacobs_aug16_1/3/4_ref.png`, `atk_July22_2_ref.png`,
  the per-setup SLAM occupancy grids
- `jacobs.png`, `atkinson.png`, the annotated photographs

`floorplan.py` turns a grid into a polygon: threshold, keep the largest free
blob, fill interior holes, trace the boundary, simplify. The Jacobs outline
comes out at 69 corners, and is distinctly not a rectangle.

**What the photographs corrected.** Two assumptions in `PROTOCOL.md` were wrong
in kind, not just in placement:

- The plasma screens are on **two opposite sides** of Jacobs, not one wall. The
  photograph labels "Plasma Screens" twice. They are now assigned to a run of
  wall segments near given anchor points, which a single boundary wall cannot
  express.
- The structure beside AP4 is labelled **"Blockage"** and is a narrow vertical
  element, not a wall. It is now a free-standing column. Turning a whole
  boundary wall metallic would shadow far more of the room than it really does.

Both spaces are also building lobbies with glass doors and polished stone
floors rather than drywall rooms. "Assume drywall" remains the starting point
as instructed, but the glass and the specular floor are visible in both
photographs and are likely to matter.

### Scale is the open problem

The grids carry no physical units, and the scale cannot be recovered from the
quoted dimensions:

| Map | Scaled by quoted width | Resulting area | Quoted area | Error |
|---|---|---|---|---|
| Jacobs | 18 m | 1429 sq ft | 1500 sq ft | -4.7 % |
| Atkinson | 8 m | 254 sq ft | 500 sq ft | **-49 %** |

Jacobs agreeing to 5 % may well be luck, because Atkinson is out by a factor of
two on the same method. So `scale_from_extent` is a stopgap, `to_metres`
requires an explicit factor and has no default, and the real scale must come
from the measurement file's `d1`/`d2` spatial axes.

Getting this wrong would produce a correctly shaped room of the wrong size,
which is much harder to notice than an obviously wrong shape.

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

**These numbers were revised after the real floor plan was extracted.** The
first estimate used synthetic rectangular rooms and was far too optimistic:
the real Jacobs outline has 71 surfaces at full detail, not the 30 assumed.

Measured on the actual Jacobs map, at order 2, 12,500 points against 4 APs:

| Outline detail | Surfaces | Per position | One setup | All 8 setups |
|---|---|---|---|---|
| Full, 2 px | 71 | 380 ms | 5.3 h | 42 h |
| Simplified, 6 px | 25 | 38 ms | 32 min | 4.3 h |
| Simplified, 12 px | 18 | 20 ms | 17 min | 2.2 h |

So there is a **real accuracy-versus-cost decision** in how much the outline is
simplified, and it should be made deliberately and recorded, not left to a
default. The 6 px setting keeps the shape of the space while cutting the cost
by an order of magnitude, and is the current choice.

The earlier claim that the full route runs without subsampling holds only at
6 px or coarser. At full detail it does not.

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
floorplan.py     occupancy grid to floor-plan polygon
environments.py  builds a scene from the outline and a loaded measurement
benchmark.py     computational feasibility, runnable without the data
test_dloc.py     regression tests for this directory
maps/            occupancy grids and photographs, mirrored from the DLoc repo
data/            measurement files go here, not in version control
```

Requires `h5py` and `pillow` in addition to the root `requirements.txt`.

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