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

### Three places this implementation goes beyond the paper

**Surface roughness.** The Fresnel coefficients of Eq. 56 assume an ideally
smooth interface. Real building surfaces have a root-mean-square height of
roughly 0.1 mm to 2 mm. Against a 6 cm wavelength at 5 GHz that costs under
0.2 dB for ordinary interior surfaces and 0.75 dB for the roughest entry in the
library; against a 5 mm wavelength at 60 GHz it reaches 6 to 7 dB. The coherent
specular field is therefore multiplied by a roughness factor, derived in
`materials.roughness_factor` from the two-way phase spread across the surface
height distribution. The default is the Miller-Brown form rather than the more
commonly quoted Rayleigh one; that choice is made on validity range, since the
Rayleigh derivation assumes a small phase variance and the roughness parameter
reaches 2.5 for brick at 60 GHz. Experiment 7 measures what the correction
changes, and the surface heights are differentiable so they can be fitted
rather than trusted.

**Stratified walls.** A single homogeneous slab cannot represent a real
partition. A stud wall is plasterboard, an air cavity, then plasterboard; a
window is glass, a gap, then glass. Each internal boundary reflects and those
reflections interfere, giving a structure in frequency and angle that no
effective permittivity reproduces. Walls can now be defined as a stack of
layers, solved exactly by the Abeles characteristic matrix
(`materials.multilayer_coefficients`), which supplies both reflection and
transmission and is differentiable in every layer thickness and permittivity.

Replacing the single-slab formula exposed a defect in it. It took the phase
across the slab along the slanted ray path `d / cos(theta_t)`, where the
quantity that actually interferes at the exit face is the normal component
`d * cos(theta_t)`. The two agree exactly at normal incidence and diverge as
the square of the cosine, so no normal-incidence check could have caught it,
and the closed-form tests in section 5 all sit at or near normal incidence.
The symptom was that Fabry-Perot resonances moved the **wrong way with angle**,
by 22 % in free spectral range at 70 degrees. Correcting it changes
through-wall transmission by up to 7.7 dB and 135 degrees of phase at oblique
incidence. `slab_transmission` is now a single-layer call into the stratified
solver, so there is one implementation of the layer phase rather than two.

**Edge-to-edge diffraction, with the slope term.** First-order diffraction is
a correction almost everywhere, but deep in a shadow it is not a correction to
anything, because it is itself nearly zero there and what reaches the receiver
has bent around two edges rather than one. Second-order diffraction is now
available (`TracerConfig.max_diffraction_order = 2`), with the stationary
two-edge path found by alternating exact minimisation, which converges to the
global minimum because the total path length is jointly convex in the two edge
parameters, and stays differentiable because the iteration is simply unrolled.

Slope diffraction comes with it. Ordinary diffraction uses only the *value* of
the incident field at the edge; the field arriving at a second edge is a
diffracted field, and near the first edge's shadow boundary it varies rapidly
across the second, where the derivative term is comparable in size. That
derivative is taken by autograd on the coefficient itself, which is the one
place where the differentiability built for inverse rendering pays off in the
forward direction too.

Three things had to be got right, and each produced a plausible wrong answer
rather than an error:

- **Edges meeting at a corner are not a cascade.** The spreading factor carries
  `1/sqrt(s12)`, so letting both diffraction points collapse onto a shared
  vertex sends the amplitude to infinity. That configuration is the separate
  canonical problem of corner diffraction. Before excluding it, the spurious
  contribution put 27 dB of invented power into an ordinary room and made an
  indoor link beat free space.
- **The distance parameter is not reciprocal in a cascade.** `L = s's/(s'+s)`
  is built from the distance to the source on one side and to the observation
  point on the other, and in a cascade those are different things in the two
  directions of travel. The amplitude comes out symmetric on its own, so this
  was invisible except through reciprocity, where it was worth several dB. The
  geometric mean of the two directions restores it, the same device already
  used for the cone angle and Luebbers' face coefficients.
- **Masking a singularity with `torch.where` does not remove it.** The rejected
  branch is still evaluated, and its backward pass multiplies zero by the
  infinity from `1/sqrt(s12)`, giving NaN gradients for the whole batch while
  the forward values stay perfectly correct. Clamping before the mask fixes it.

This also exposed a latent defect in first-order diffraction: Luebbers' face
coefficients take `sqrt(|sin(phi_i)| |sin(phi_d)|)` with the clamp applied
after the square root rather than inside it, so a ray grazing exactly along a
wedge face produced a NaN gradient. Single diffraction rarely lands exactly on
that angle; cascades hit it routinely.

### Not implemented

Higher-order diffraction, diffuse scattering, near-field and full-wave effects
(the paper notes the same limits in App. A), dual-polarised path tracking, and
GPU/OptiX acceleration. Diffraction from concave junctions (`n < 1`) is skipped
as outside UTD validity.

---

## 5. Validation

`python3 tests/test_rfdt.py` runs 53 checks; all pass. The substantive ones:

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

### Experiment 7: surface roughness, and what it is worth

`results/exp7_roughness.{png,json}`

The Fresnel coefficient assumes a mirror-smooth wall. This experiment adds the
roughness correction described in section 4 and then measures, rather than
assumes, what it changes.

**Change in route-mean received power against the ideally smooth model:**

| Material | RMS surface height | 5 GHz | 60 GHz |
|---|---|---|---|
| Metal | 0.05 mm | -0.00 dB | -0.10 dB |
| Foam board | 0.30 mm | -0.00 dB | -0.23 dB |
| Plasterboard | 0.20 mm | -0.00 dB | -0.47 dB |
| Concrete | 1.00 mm | -0.03 dB | **-5.88 dB** |
| Brick | 2.00 mm | -0.08 dB | **-7.48 dB** |

Three things follow, and the first two are not what was expected before running
it.

1. **The metal-versus-foam comparison this study is built on is untouched.**
   Both materials happen to be smooth, so the correction cannot reach them. The
   60 GHz power gap moved from 15.72 dB to 15.85 dB, slightly wider rather than
   narrower. Every 5 GHz result in experiments 2, 5 and 6 is unchanged, and
   experiment 1, the differentiability claim, is unchanged to the last digit.

2. **What it does repair is concrete and brick**, which were overpredicted by 6
   to 7 dB at 60 GHz, and the 77 GHz radar echoes of experiment 3, which move by
   up to 21 dB.

3. **At 60 GHz the surface height matters more than the model.** Panel (d)
   sweeps the height over its factor-of-two uncertainty: concrete's route power
   spans **5.4 dB**, against 2.7 dB for the choice between the two roughness
   models. The weakest input now dominates the answer, which is the argument for
   fitting the surface height from data rather than tabulating it. The tracer
   supports that: `MaterialParams` takes an optional third learnable parameter
   whose analytic gradient is verified against finite differences in the tests.

Experiment 3 reports, per material, the share of specular power the roughness
term removes and that nothing re-radiates, and flags any surface past the
Rayleigh criterion. For concrete and brick at 77 GHz that share exceeds 96 %, so
those echoes are a specular residue rather than a predicted total return, and
the real echo is higher by an amount this simulator cannot supply. Saying so is
the point of the column.

### Experiment 8: stratified walls, and the phase defect they exposed

`results/exp8_layered_walls.{png,json}`

A single homogeneous slab cannot represent a real partition. Walls can now be
defined as a stack of layers, solved exactly by the transfer matrix.

**A stack is not a slab.** Against a solid wall of the *same total thickness*,
transmission differs by up to **7.9 dB** for a
stud partition and **8.8 dB** for double
glazing, with root-mean-square differences of
4.3 and 4.4 dB
across 1 to 12 GHz. The cavity resonances in panel (a) are structure that no
effective permittivity can produce, and averaging the layers does not blur them,
it deletes them. Panel (b) shows the same for reflection against angle, which
matters more indoors: the partition has deep nulls near 19, 50 and 71 degrees
that the solid equivalent has no trace of.

**The defect this exposed.** The previous single-slab formula took the phase
across the slab along the slanted ray path `d / cos(theta_t)`, where the
quantity that interferes at the exit face is the normal component
`d * cos(theta_t)`. The two agree exactly at normal incidence, and the only closed-form check that
exercised slab transmission sat at normal incidence, so nothing could have
caught it. The two-ray test is at 45 degrees but is a reflection off metal and
never touches the slab path.
Fabry-Perot resonances moved the wrong way with angle, off by 22 % in free
spectral range at 70 degrees. The correction is worth up to
**9.7 dB** for plasterboard,
10.0 dB for wood and
9.0 dB for concrete, with phase
changes reaching 179
degrees. Glass, at 6 mm, is thin enough that it barely notices
(0.04 dB), which is a useful reminder
that thickness in wavelengths is what decides.

**Does it reach the robot?** The same room and the same
115 mm of wall, layered against solid:

| | Solid | Layered | Change |
|---|---|---|---|
| Route-mean power | -41.34 dBm | -39.94 dBm | **+1.39 dB** |
| Delay spread | 2.53 ns | 3.29 ns | +0.76 ns |
| Rice K-factor | 4.96 dB | 0.80 dB | **-4.16 dB** |
| Angular spread | 28.8 deg | 42.6 deg | **+13.8 deg** |

Yes, and the interesting part is which quantities move. Power changes by
1.4 dB while the angular spread grows
by half again and the Rice K-factor drops by
4.2 dB. That is the same pattern
experiment 2 found for material: the level barely notices and the shape of the
channel changes completely. Note also that for a homogeneous wall this
simulator's reflection is the single-interface Fresnel value and does not depend
on thickness at all, so the entire difference above comes from the stack's
reflection, not its transmission.

### Experiment 9: edge-to-edge diffraction, and where it stops being optional

`results/exp9_double_diffraction.{png,json}`

Second-order diffraction is expensive, so the question is not whether it adds
something but where it is the difference between an answer and no answer. The
experiment measures both ends of that.

**Where it is the entire answer.** Two knife edges in series in free space, with
the transmitter below the first edge and the receiver below the second, so the
direct path is blocked and a ray bending over the first screen alone is blocked
by the second. At **30 of
31** receiver heights, single-edge diffraction predicts not a
small field but *exactly zero*; the cascade gives
-101 to
-80 dBm. Quoting a ratio there would be
meaningless, since the denominator is a numerical floor rather than a
prediction, so panel (a) plots levels and marks the region where one model has
nothing to say.

**Where it is nearly irrelevant, and why.** Inside the furnished room, along a
transect that starts behind the partition and ends past its open end, adding the
cascade changes the field by at most **0.95 dB**
(0.22 dB mean in the shadowed half,
0.52 dB where the direct path survives). The slope
term is worth up to 0.28 dB.

The reason is the more useful finding, and it was not what was expected before
running it: **a partition inside a reflective room does not create a deep
shadow.** Panel (c) shows both diffraction orders sitting
17 dB below the total on average,
and never closer than 7 dB. Wall
reflections fill the geometric shadow in completely, so no diffraction term of
any order can matter much there. That is why the shadowed and lit halves show
similar changes: neither is diffraction limited.

The practical consequence is that second-order diffraction is worth its
cost, roughly fifteen to twenty times a first-order trace on this machine, in
sparse or outdoor-like geometry, and is not worth it inside a furnished room.
The ratio is wall-clock and moves with machine load, so it is quoted as a range
rather than to two figures that would not reproduce. It is available and off by default for exactly
that reason.

---

## 7. Honest limitations

- **Scalar polarisation.** One complex coefficient per path. The default is the
  TE (perpendicular) coefficient, which behaves correctly both for conductors
  and at grazing incidence. A strict unpolarised treatment needs dual-polarised
  path tracking; the `"unpolarised"` option is a documented approximation whose
  phase is taken from the TE component.
- **Second-order diffraction is off by default, on cost rather than physics.**
  The furnished room has 36 diffracting edges and so 1260 ordered pairs, and
  evaluating them takes about six times as long as the whole of the rest of the
  trace. Experiment 9 turns it on and measures where it earns that.
- **The cascade is ray-optical and its distance parameters are symmetrised, not
  derived.** The rigorous treatment of two nearby edges is a joint coefficient
  with a two-variable transition function, which this does not implement. Pairs
  closer than a wavelength are rejected rather than approximated, and no
  diffraction is computed from concave junctions.
- **Roughness removes energy without re-radiating it.** The roughness factor
  takes power out of the specular direction, and nothing puts it back, because
  diffuse scattering is not modelled. Every roughness-enabled power is
  therefore a lower bound, by a known and signed amount. The two effects belong
  together and should be added together.
- **Roughness is not applied to the human body entry.** Its surface height is
  deliberately zero, which is not a claim that a body is smooth. The Rayleigh
  factor assumes a planar interface with small-scale height variation, and a
  body violates the planar assumption at a much larger scale, so the correction
  is outside its regime. Applying a plausible-looking clothing roughness there
  would have moved the 77 GHz radar echo by more than 20 dB on the strength of
  an invented number.
- **Layering applies to plate facets, not to closed solids.** A wall or a
  board is a slab and can be a stack. A closed solid still uses the
  two-interface volume model, so the partition box in the furnished room is
  homogeneous however its material is defined.
- **The stack coefficient carries the full propagation delay across the wall**,
  matching what `slab_transmission` has always returned, while the tracer also
  applies `exp(-jkL)` over a straight ray path that runs through the wall. The
  air-equivalent phase across the thickness is therefore counted twice. For
  thin partitions this is a fixed offset per crossing rather than an error that
  accumulates, but it is a real inconsistency and it is recorded rather than
  quietly changed, because correcting it would alter the meaning of a
  coefficient the rest of the simulator already agrees on.
- **The surface heights are estimates, not measurements.** Unlike the
  permittivities, which are ITU-R P.2040-1 regressions, the roughness values in
  `ROUGHNESS_SIGMA_M` are order-of-magnitude literature figures carrying about
  a factor-of-two uncertainty. Experiment 7 panel (d) shows what that
  uncertainty is worth at 60 GHz, and it is not small for rough materials.
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