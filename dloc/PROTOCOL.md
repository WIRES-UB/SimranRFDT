# Comparison protocol, fixed before the measured data is opened

This document is written **before** any DLoc measurement file has been read.
It states what will be compared, how, and what counts as agreement or
disagreement. The point is that the answer should be trustworthy whichever way
it comes out.

The task, as set: simulate the DLoc environments at 5 GHz with the robot
moving, and check whether the simulated channel is close enough to the real
measurements to support sim2real work. If it is not, that is a useful result
and motivates a different simulator for sub-6 GHz.

That second outcome is a real possibility, not a formality. Everything below is
designed so that a negative result is as informative as a positive one.

---

## 1. What will be compared

Raw CSI cannot be compared directly against a simulator. Three separate
reasons, all of them properties of WiFi hardware rather than of the simulator:

1. **Time-of-flight offset.** Each packet carries an unknown timing offset.
   Removing it is the entire purpose of DLoc's consistency decoder. Absolute
   path delay is therefore not comparable.
2. **Carrier and sampling frequency offset.** CSI phase is randomised per
   packet. Raw phase is not comparable.
3. **Automatic gain control.** CSI magnitude is in arbitrary units. RSSI is the
   calibrated amplitude quantity.

So the comparison uses only quantities that survive those three problems.

### Primary metric

**Received signal strength across the map**, simulated against the dataset's
`RSSI` field, per access point.

- Reported as the CDF of absolute error in dB, with median and 90th
  percentile, matching how DLoc reports localisation error.
- Computed per AP and pooled across APs.
- A single global scalar offset per AP is allowed, fitted once, to absorb
  unknown transmit power, antenna gain and cable loss. This is a calibration
  constant, not a per-point fit. **No per-position, per-frequency or
  per-material tuning.**

### Secondary metrics

- **Frequency selectivity.** Standard deviation of `|H(f)|` in dB across
  subcarriers, and coherence bandwidth from the frequency correlation of
  `H(f)`. Both are invariant to the unknown per-packet phase and to the
  amplitude scale, so they test whether the multipath structure is right even
  when the absolute level is not.
- **Spatial correlation.** Correlation coefficient between simulated and
  measured RSS as a function of position along the robot's route. Tests
  whether the simulator puts the fades and peaks in the right *places*, which
  matters more for sim2real than matching absolute level.
- **Angle of arrival** per AP, after applying the dataset's `ap_aoa` rotation
  offsets, compared against the geometric direction to the ground-truth
  position. Tests whether the dominant path is coming from the right place.

### Explicitly not compared

Absolute delay, raw CSI phase, and absolute CSI magnitude. Any of these could
be made to look good or bad by choice of calibration, so none of them is
evidence either way.

---

## 2. Success criteria, stated in advance

These thresholds are set now, from what is reasonable for a ray tracer against
real WiFi measurements, and not adjusted afterwards.

| Outcome | Median RSS error | Spatial correlation |
|---|---|---|
| Good enough for sim2real | under 4 dB | above 0.7 |
| Marginal, needs work | 4 to 8 dB | 0.4 to 0.7 |
| Not good enough | over 8 dB | under 0.4 |

For context, RFDT's own paper reports a 5.2 dB median RSS error against
measurements in a complex indoor environment, and Sionna 7.7 dB, on a
different scene and setup. Landing in that region would be a reasonable
result; landing far outside it would not.

If the result is "not good enough", the deliverable is a diagnosis of *why*,
broken down by whether the error is a constant offset (calibration), a
position-dependent bias (geometry or material), or uncorrelated scatter
(missing propagation mechanisms). Those three point at different fixes.

---

## 3. Material assumptions

Provided by Roshan, who collected the data, and recorded here so they are
visible rather than buried in code:

- **All walls: drywall.** Implemented as ITU-R P.2040-1 `plasterboard`,
  eps' = 2.73, sigma = 0.0085 f^0.9395 S/m, valid 1 to 100 GHz.
- **Jacobs, top wall: LCD/plasma screens.** The DLoc paper describes "the wall
  of plasma television screens behind AP3" as creating a multipath-rich
  environment. Modelled as `metal` pending confirmation, since panel displays
  are effectively conducting at 5 GHz. **Open question for Roshan:** whether
  this should be a full conductor or partial.
- **Jacobs, wall beside AP4: metal.** The paper states AP4 "is hidden behind a
  wall, thus collecting significant NLOS data".
- **Floor and ceiling:** not specified in the dataset. Assumed concrete floor
  and mineral-fibre ceiling tile, which is typical for these buildings. Both
  flagged as assumptions.

Any material not listed above will be left at the simulator's default and
recorded in the run log.

---

## 4. Order of work

1. **Atkinson first** (8 x 5 m, 3 APs, mostly LOS). Smaller, fewer surfaces,
   far less sensitive to guessed materials. If the simulator cannot match the
   simple environment, the complex one is not worth running.
2. **Jacobs second** (18 x 8 m, 4 APs, high multipath, NLOS). The real test,
   and the one where the screen wall and metal wall matter.
3. Within Atkinson, `July16` and `July18` are the same setup on different
   days. Running both gives a measure of how much the *measurement* varies
   when nothing in the simulation changes, which sets a floor on how well any
   simulator could possibly do.

That last point is worth stating plainly: **the day-to-day repeatability of
the measurement bounds the best achievable agreement.** If July16 and July18
differ from each other by 3 dB median, then a simulator matching either to
3 dB has done as well as the data permits.

---

## 5. Geometry, and what is still missing

The dataset provides AP antenna coordinates (`ap`, one `[n_ant x 2]` cell per
AP) and ground-truth robot positions (`labels`, `[n_datapoints x 2]`) in the
SLAM map frame. Those will be used directly rather than reconstructed, so the
geometry is the dataset's and not an invention.

Note that the axes in Figure 8 of the DLoc paper span a larger area than the
quoted 8 x 5 m and 18 x 8 m, because the figure shows the full SLAM map while
the quoted size is the data collection area. Room outlines will therefore be
taken from the map extent in the data, not from the quoted floor area.

**Not yet available:** the measurement files themselves, which require a
consent form and a password-protected download. Until they arrive, no AP
coordinate, room outline or robot position in this directory is real, and
anything provisional is marked as such in code and refuses to be used for a
comparison run.

---

## 6. What would make this analysis untrustworthy

Recorded so it can be checked afterwards:

- Choosing the metric after seeing the measurements.
- Tuning materials, geometry or antenna parameters to improve agreement, then
  reporting the tuned result as a prediction.
- Fitting more than one scalar offset per AP.
- Reporting only the environment or setup that agreed best.
- Quietly dropping measurement points that disagreed.

All runs, including failed and superseded ones, will be logged.