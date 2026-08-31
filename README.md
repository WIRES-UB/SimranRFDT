# RF_Simulator

A differentiable RF propagation simulator for **indoor robotics**, implementing
the method of *"Physically Accurate Differentiable Inverse Rendering for Radio
Frequency Digital Twin"* (RFDT, MobiCom '26) and using it to measure how
**different materials** change what a moving robot's radios actually see.

The package builds a transmitter and a receiver, traces physically valid
propagation paths through a furnished room, and reports channel and radar
results for every material in the ITU-R P.2040-1 database. Because the whole
forward model is differentiable, the same code also runs *backwards* to recover
material and geometry parameters from RF measurements.

---

## 1. What is here

```
RF_Simulator/
  rfdt/                      the simulator
    materials.py             ITU-R P.2040 database, Fresnel, penetration   (Eq. 56-59)
    transition.py            UTD transition function F(x), wedge diffraction (Eq. 7, 8, 44)
    geometry.py              differentiable mesh ops, facets, wedges        (Eq. 50-55)
    antennas.py              transmitters, receivers, patterns, MIMO arrays
    tracer.py                the ray tracer                                 (Eq. 1, 9, 10, 11)
    signal.py                FMCW radar transforms, Dirichlet surrogate     (Eq. 18-22, 27)
    scenes.py                rooms, obstacles, robot trajectories
    metrics.py               channel metrics, SSIM / PSNR, gradient checks
    optimize.py              digital-twin optimisation loop                 (Eq. 23, 24)
  experiments/               the six studies described below
  tests/test_rfdt.py         29 physics and differentiability regression tests
  dloc/                      validation against the DLoc measured dataset
  results/                   figures, JSON and CSV written by the experiments
  run_all.py                 run the tests then every experiment
```

## 2. Installation and running

```bash
pip install -r requirements.txt      # torch >= 2.5, numpy, matplotlib
python3 run_all.py                   # tests, then all six experiments (~14 min)
python3 run_all.py 2                 # just the material sweep
python3 tests/test_rfdt.py           # regression tests alone
```

`torch >= 2.5` is required if you have `numpy >= 2`: earlier torch wheels are
built against the numpy 1.x ABI and fail with *"Failed to initialize NumPy"*.

Minimal use:

```python
from rfdt import RFDTracer, TracerConfig, scenes, antennas

mesh = scenes.furnished_room(wall="concrete")          # 6 x 5 x 2.8 m room
tx   = antennas.wifi_ap((1.2, 1.2, 2.7), frequency=5e9, power_dbm=20)
traj = scenes.survey_trajectory(n_samples=48)          # the robot's route
rx   = antennas.robot_client(traj.positions[0])

paths = RFDTracer(mesh, TracerConfig(max_order=2)).trace(
    tx, rx, rx_positions=traj.positions)
rss = paths.power_dbm(tx.power_dbm)                    # dBm at every route point
```

---

## 3. The transmitter and the receiver

Both are near-point sources with a directional pattern, which is the regime RF
ray tracing assumes (Sec. 3.1), and both are differentiable functions of their
position and orientation.

| | Communication link | Robot radar |
|---|---|---|
| Transmitter | ceiling access point, patch pattern, 20 dBm, 5 / 60 GHz | 77 GHz FMCW, 12-element Tx array, 12 dBm |
| Receiver | robot-mounted dipole, 20 MHz, NF 6 dB | 16-element Rx array, 3.5 GHz, NF 12 dB |
| Observable | RSS, delay spread, K-factor, Doppler | range profile, range-Doppler map |

Patterns available: isotropic, half-wave dipole, cosine-power patch and a
Gaussian main lobe with a sidelobe floor. Arrays follow App. C.4: paths are
traced once from the array centroid and the MIMO response is synthesised from
per-element phase offsets, which is what makes a 12 x 16 virtual array
affordable.

The receiver carries a thermal noise model (`kTB` plus noise figure), so every
result can be quoted as an SNR rather than a bare field strength.

---

## 4. What was implemented from the paper

**Reparameterised method of images** (Sec. 3.2, Eq. 50-55). Reflection points
are solved on the *infinite supporting planes* of candidate facets, so they
always exist and vary smoothly with the geometry. Paths are never created or
destroyed, which is what removes the discontinuity of Sec. 3.1.

**Physically consistent path validity** (Sec. 3.3, Eq. 11). The binary
in-facet test is replaced by the product of UTD transition functions
`W = prod_i F(k L_i a_i)`. `F` is evaluated through the Faddeeva function using
Weideman's spectral approximation, and its backward pass is the hand-written
analytic derivative `F'(x) = F(x)(1 + 2jx)/(2x) - j`, matching the paper's
"custom backward passes" (Sec. 5.3).

**Edge diffraction** (Eq. 6, 7). The diffraction point is the Fermat point of
Eq. 40; for a straight edge the stationarity condition of Eq. 42 has the closed
form `t* = (h_rx t_tx + h_tx t_rx)/(h_tx + h_rx)`, so no iterative root find is
needed and the gradient stays clean. Wedge indices come from mesh adjacency.

**Secondary visibility and penetration** (Eq. 9, App. E.2). Occlusion uses the
same transition function, plus a material transmission term, so a link fades
smoothly into a shadow and carries the correct material-dependent loss through
obstacles.

**Coarse-to-fine surrogate** (Sec. 4, Eq. 18-22). The Dirichlet-kernel point
spread function, phase-agnostic warm-up, and annealing to the exact FFT.

**Digital-twin optimisation** (Sec. 5.2, Eq. 23, 24). Multiscale MSE loss,
Adam, optional Laplacian mesh regularisation.

### Two places this implementation deviates, and why

1. **UTD boundary terms are damped by `|F|`.** Classical UTD pairs a
   *Heaviside*-switched geometric field with cotangent terms that are
   deliberately singular, so the two discontinuities cancel. RFDT replaces that
   Heaviside with the smooth weight `F`, so leaving the compensating step in
   place *introduces* a discontinuity rather than removing one. Damping the
   term by `|F(x)|` recovers classical UTD away from boundaries and lets the
   coefficient pass through zero continuously at them. Without this the field
   jumps by about half the reflected field at every reflection boundary.

2. **Reciprocity is enforced where the standard heuristics break it.** The UTD
   cone angle `beta0` and Luebbers' face reflection coefficients are evaluated
   as geometric means of the incident and outgoing values. Each reduces to the
   textbook value in the cases those formulas were derived for, and the result
   is a simulator that is reciprocal to machine precision (verified in the
   tests) instead of by a few hundredths to a couple of dB.

Both are documented at the point of implementation.

### Not implemented

Higher-order diffraction, diffuse scattering, near-field and full-wave effects
(the paper notes the same limits in App. A), dual-polarised path tracking, and
GPU/OptiX acceleration. Diffraction from concave junctions (`n < 1`) is skipped
as outside UTD validity.

---

## 5. Validation

`python3 tests/test_rfdt.py` runs 29 checks; all pass. The substantive ones:

| Check | Result |
|---|---|
| Line of sight vs the Friis equation | agrees to < 0.001 dB |
| Two-ray over a conducting plate vs closed form | agrees to < 0.05 dB |
| Reciprocity (swap Tx and Rx), all path types | ~1e-13 dB |
| Autograd vs central finite differences | ~1e-9 relative |
| Faddeeva function vs published reference values | < 1e-12 |
| `F(x)` asymptotics vs Eq. 46 and Eq. 49 | matches both limits |
| Candidate pruning is lossless | identical to < 1e-9 dB |
| Conductor and grazing-incidence reflection | `Gamma -> -1` in both |
| Radar range profile peak | within one range cell |
| Two-ray notch spacing vs `c / dL` | 0.06 % error |
| Group delay of a single path vs `L / c` | agrees to 1e-15 s |

The forward model is therefore checked against closed-form electromagnetics,
not only against itself.

---

## 6. Results

All numbers below are produced by the scripts in `experiments/` and are written
to `results/` as PNG, JSON and CSV. Nothing is hand-edited.

### Experiment 1: is the visibility term actually differentiable?

A conducting plate with the specular point walking across its edge.

| Scheme | Max field jump / range | Gradient vs FD | Gradient w.r.t. plate *size* |
|---|---|---|---|
| Conventional binary test (Eq. 3) | 0.96 | 3.7e-09 | **identically zero** |
| Soften triangle (Eq. 4) | 0.006 | 2.1e-09 | non-zero but biased field |
| RFDT weight (Eq. 11) | 0.011 | 2.6e-09 | non-zero on 49 % of the sweep |
| RFDT + diffraction | 0.007 | 2.1e-09 | non-zero everywhere, cos 0.996 vs FD |

This is slide 13 of the deck, and the decisive column is the last one. A correct gradient with respect to the
*transmitter position* proves little, because path length varies smoothly
however visibility is modelled. Plate size is different: scaling the plate does
not move its supporting plane, so it changes the field *only* through the
validity test. The conventional test therefore yields exactly zero gradient
there and can never learn a reflector's extent, which is precisely the
"d-Geometry: No" entry in the paper's Table 1.

### Experiment 2: the indoor robot channel versus wall material

Robot driving a 15 m route; only the wall material changes. Averaged over the
route and a 2 % frequency band, because a single point at a single frequency
mostly measures fading, not material.

At 5 GHz the channel changes monotonically with reflectivity:

Rows are sorted by reflectivity, most reflective first.

| Wall material | Reflectivity, normal incidence | Route-mean RSS | RMS delay spread | K-factor | Angular spread | Coherence BW (est.) |
|---|---|---|---|---|---|---|
| Metal | -0.00 dB | -37.43 dBm | 4.38 ns | -3.61 dB | 59.2 deg | 48 MHz |
| Human body (approx) | -2.30 dB | -38.77 dBm | 4.03 ns | -1.62 dB | 53.2 deg | 52 MHz |
| Marble | -6.87 dB | -40.47 dBm | 3.26 ns | 1.83 dB | 41.4 deg | 63 MHz |
| Glass | -7.32 dB | -40.58 dBm | 3.19 ns | 2.13 dB | 40.3 deg | 64 MHz |
| Concrete | -8.09 dB | -40.76 dBm | 3.07 ns | 2.63 dB | 38.3 deg | 67 MHz |
| Brick | -9.67 dB | -41.03 dBm | 2.83 ns | 3.60 dB | 34.4 deg | 72 MHz |
| Plasterboard | -12.16 dB | -41.34 dBm | 2.53 ns | 4.95 dB | 28.8 deg | 82 MHz |
| Chipboard | -12.56 dB | -41.38 dBm | 2.48 ns | 5.15 dB | 28.1 deg | 84 MHz |
| Wood | -15.34 dB | -41.54 dBm | 2.24 ns | 6.37 dB | 23.3 deg | 95 MHz |
| Ceiling board | -20.19 dB | -41.64 dBm | 1.98 ns | 7.89 dB | 17.7 deg | 113 MHz |
| Foam board (approx) | -36.73 dB | -41.55 dBm | 1.79 ns | 9.35 dB | 12.6 deg | 133 MHz |

Delay spread, K-factor, angular spread and coherence bandwidth are each
strictly monotone in the Fresnel coefficient across all eleven materials. The
coherence bandwidth column is the standard `1/(5 sigma_tau)` estimate from
delay spread, not a measurement; experiment 5 measures it directly and shows
where that estimate holds and where it does not. RSS
is monotone with **one exception**: foam board comes out 0.09 dB above ceiling
board, an inversion far smaller than the 5.7 dB standard deviation of RSS along
the route, so it should be read as the two being indistinguishable rather than
as a trend reversal.

A metal-walled room behaves like a reverberation chamber (delay spread 4.4 ns,
multipath 3.6 dB *above* the direct path, 59 degrees of angular spread) and a
foam-walled one like an anechoic chamber (1.8 ns, direct path 9.4 dB above the
multipath, 13 degrees). Coherence bandwidth nearly triples across the range,
which directly changes how much equalisation a robot's radio needs.

The RSS column spans only 4 dB, because the direct path dominates it and the
direct path does not touch the walls. The *shape* of the channel is what the
material changes, not its overall level.

At 60 GHz the picture changes in a way worth stating plainly. Received power
now spans 16 dB instead of 4 dB, and reflective walls *raise* it: metal
-62.3 dBm against foam -78.0 dBm. The reason is that the plasterboard partition
attenuates a 60 GHz direct path far more than a 5 GHz one, so over much of the
route the reflected paths are the only ones delivering energy, and a room with
absorbing walls has none of them. The K-factor confirms it: at 60 GHz it is
negative for every material, meaning the direct path is no longer dominant
anywhere along the route.

For mmWave robot coverage this inverts the usual intuition: in a partitioned
space, reflective surfaces are an asset rather than a source of fading. The
effect only appears because the simulator sums paths coherently rather than
adding powers.

At 60 GHz the monotonicity also breaks in two places, and both are informative
rather than noise. Delay spread rises again for the two least reflective walls
(wood 2.39 ns, ceiling board 2.49 ns, foam 3.07 ns) because once the walls stop
reflecting, the only paths still reaching the shadowed half of the room are
long diffracted ones, which lengthens the delay spread even as the total energy
falls. Coherence bandwidth inverts for metal against human body for the same
reason in reverse. Exact values for every material and both bands are in
`results/exp2_material_sweep.csv`.

Materials evaluated outside the frequency range their ITU-R P.2040-1 regression
was fitted over are flagged `(*)` in the console output and carry
`in_validity_range: false` in the JSON and CSV. Entries marked `source:
approx` (foam, plastic, paper board, human body) are order-of-magnitude
literature values, not ITU data, and are labelled as such everywhere.

### Experiment 3: the robot's own 77 GHz radar

| Study | Result |
|---|---|
| Wall echo vs material, wall at 3 m | metal -53.8 dBm (SNR 12.7 dB) down to foam -87.8 dBm (SNR -21.3 dB), monotone in reflectivity; range error 0.6 cm against a 4.3 cm resolution |
| Metal target through a board | plastic -2.0 dB, paper -3.3 dB, foam -0.4 dB of excess two-way loss |
| Doppler, robot closing at 1 m/s | predicted 513 Hz, simulated 513 Hz; range-Doppler peak at 1.06 m/s and 3.01 m |
| Second surface 25 cm behind plasterboard | metal -2.4 dB, concrete -10.5 dB, wood -17.7 dB, foam -38.3 dB below the front return |

The metal echo of -53.8 dBm matches a hand-computed two-way link budget of
-53.7 dBm, which is an independent check on the absolute scaling.

Practical reading: a robot radar can map metal, concrete and glass easily, will
struggle with wood and ceiling board, and will effectively not see foam at all.
Low-loss boards hide a target by only a few dB, so obstacles of that kind
degrade rather than prevent detection.

### Experiment 4: recovering the scene from measurements

**A. Wall material from an RSS survey.** 96 observations (32 route points at 3
frequencies), 0.5 dB noise, starting from a wrong material (wood instead of
concrete):

| Quantity | True | Initial guess | Recovered | Error |
|---|---|---|---|---|
| `eps'` | 5.240 | 1.990 | 5.134 | 2.0 % |
| `sigma` | 0.1627 | 0.0264 | 0.1750 | 7.6 % |
| `abs(Gamma)`, 0-75 deg | | 5.20 dB off | | **0.06 dB** |

**B. Wall distance and material from radar, surrogate ablation.** Reported
across a sweep of initial errors rather than at one chosen offset:

| Initial distance error | = range cells | With surrogate | Without surrogate |
|---|---|---|---|
| 2 cm | 0.5 | 0.00 cm | 0.00 cm |
| 5 cm | 1.2 | 0.00 cm | 0.00 cm |
| 10 cm | 2.3 | 0.01 cm | 0.00 cm |
| 20 cm | 4.7 | **0.00 cm** | 25.40 cm |
| 40 cm | 9.3 | **0.04 cm** | 43.22 cm |
| 80 cm | 18.6 | 89.71 cm | 77.77 cm |

The coarse-to-fine surrogate extends the capture range from about 2 range cells
to about 9, which is the benefit Fig. 16(b) reports. It is not unlimited: at
18 range cells **both methods fail**, and the surrogate run actually ends
further from the truth than it started. That row is kept deliberately. The
surrogate widens the basin of attraction; it does not remove the need for a
reasonable initialisation.

Recovering the wall distance at all depends on the simulator being
differentiable with respect to geometry, which experiment 1 shows the
conventional visibility test is not.

### Experiment 5: channel frequency response, amplitude and phase

The frequency domain, which is what a wideband receiver actually estimates:
`H(f) = sum_i alpha_i exp(-2 pi j f tau_i)`.

**Validation against closed form.** One transmitter, one receiver, one
conducting reflector. A single reflector turns a smooth channel into a comb of
notches, and their spacing is predicted analytically by `c / dL`:

| Quantity | Analytic | Simulated | Error |
|---|---|---|---|
| Notch spacing (path difference 1.0804 m) | 277.49 MHz | 277.67 MHz | 0.06 % |
| Direct-path slope across 4 to 6 GHz (1/f spreading) | 3.522 dB | 3.522 dB | 0.00 % |
| Direct-path group delay | 10.5482 ns | 10.5482 ns | agrees to 1e-15 s |
| Notches in the direct-only response | 0 | 0 | |

The deepest notch sits 19.9 dB below the peak. So a single reflector is enough
to put a 20 dB hole in a wideband channel, and the phase becomes non-linear
around each notch, which is what distorts a wideband signal.

**Frequency selectivity versus material**, in the furnished room at one
receiver position over 4.5 to 5.5 GHz:

| Wall material | Fading depth | RMS delay spread | Coherence BW, **measured** | `1/(5 sigma_tau)` estimate | Ratio |
|---|---|---|---|---|---|
| Metal | 31.7 dB | 3.79 ns | 55 MHz | 53 MHz | 1.04 |
| Concrete | 14.9 dB | 2.49 ns | 315 MHz | 80 MHz | 3.92 |
| Foam board | 7.7 dB | 1.44 ns | 485 MHz | 139 MHz | 3.49 |

**This is a correction to experiment 2, not a confirmation of it.** The
`1/(5 sigma_tau)` rule of thumb is accurate to 4 % with metal walls, where
multipath is dense, but understates the true coherence bandwidth by roughly a
factor of four for concrete and foam. The rule assumes a rich scattering
environment; a weakly reflecting room does not provide one. Experiment 2's
coherence bandwidth column is therefore labelled as an estimate, and the caveat
is documented at the source in `rfdt/metrics.py`. The ordering across materials
still holds under direct measurement, but the absolute values do not.

**One further honest note.** Path delays are pure geometry and so are exactly
frequency independent, but path amplitudes are not: free-space spreading falls
as 1/f and the Fresnel coefficients vary with frequency. The scene is therefore
re-traced at every frequency point rather than traced once and extrapolated.
The commonly used narrowband shortcut, tracing once at band centre and applying
only the delay phase, departs from the re-traced result by up to **3.31 dB**
over a 2 GHz span. That is the cost of the shortcut, measured rather than
assumed.

### Experiment 6: metal against foam, head to head

Experiment 2 shows the trend across eleven materials. This one drops everything
but the two extremes, because the contrast is what makes the mechanism visible.
Same room, same route, same transmitter; only the walls differ, by 36.7 dB of
reflectivity, a factor of about 4,300 in power.

| Quantity, 5 GHz | Metal | Foam board | Difference |
|---|---|---|---|
| Reflectivity at normal incidence | -0.0 dB | -36.7 dB | 36.7 dB |
| Route-mean received power | -37.4 dBm | -41.6 dBm | **4.1 dB** |
| RMS delay spread | 4.38 ns | 1.79 ns | 2.6 ns |
| Rice K-factor | -3.6 dB | +9.4 dB | 13.0 dB |
| Angular spread | 59.2 deg | 12.6 deg | 46.6 deg |
| Frequency selectivity at the probe | 13.3 dB | 5.1 dB | 8.2 dB |

Two rooms that could hardly differ more in reflectivity, and **received power
notices by 4 dB**. Everything describing the *shape* of the channel changes
several times over. The Rice K-factor is the clearest: metal is negative,
meaning the echoes together are louder than the direct signal, while foam is
+9.4 dB, meaning the direct signal wins by nearly a factor of ten.

At 60 GHz the power gap opens to **15.7 dB**, and metal is now the *better*
room. With the partition attenuating a 60 GHz direct path far more than a 5 GHz
one, reflections are the only energy reaching the shadowed half, and a room
with absorbing walls has none of them.

One detail worth keeping in view: the two reflectivity curves converge at
grazing incidence. Even foam reflects almost perfectly at shallow angles, which
is why a foam-walled room still has any multipath at all.

---

## 7. Honest limitations

- **Scalar polarisation.** One complex coefficient per path. The default is the
  TE (perpendicular) coefficient, which behaves correctly both for conductors
  and at grazing incidence. A strict unpolarised treatment needs dual-polarised
  path tracking; the `"unpolarised"` option is a documented approximation whose
  phase is taken from the TE component.
- **First-order diffraction only**, and none from concave junctions.
- **Convex facets.** The facet edge distance is a minimum over outline edges,
  exact for convex outlines (what the scene builders produce) and conservative
  for reflex corners.
- **No measured ground truth.** Everything is validated against closed-form
  electromagnetics and internal consistency, not against a real measurement
  campaign or a full-wave solver. The paper's FDTD and hardware comparisons
  (App. C.3, C.1) are not reproduced here.
- **Approximate material entries.** Foam, plastic board, paper board and human
  body are not in ITU-R P.2040; they are labelled `source: approx` in every
  output. The human body entry in particular is a coarse tissue-like fit and
  should not be read as a tissue dielectric reference.
- **Small scenes.** Candidate search is over facet sequences and is exhaustive
  to the chosen order. It is fast for the 24-facet rooms used here (order 2 over
  48 route points takes about 2 s) but is not a substitute for the paper's
  BVH/OptiX implementation on scenes with 10^5 to 10^6 triangles.

## 8. Reference

Xingyu Chen, Xinyu Zhang, Kai Zheng, Xinmin Fang, Tzu-Mao Li, Chris Xiaoxuan
Lu, Zhengxiong Li. *Physically Accurate Differentiable Inverse Rendering for
Radio Frequency Digital Twin.* MobiCom '26. Equation and section numbers
throughout this code refer to that paper.

Material data: Recommendation ITU-R P.2040-1, *Effects of building materials
and structures on radiowave propagation above about 100 MHz*, Table 3.