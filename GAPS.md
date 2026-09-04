# Known gaps

Every physical approximation in this simulator that is known to be wrong, or
known to be unjustified, in one place. Updated whenever one is closed.

The convention: a gap is only closed when there is a measurement showing what
changed, not when code exists. Where a fix made something worse, or made no
difference, that is recorded too. Several entries below exist because a fix
created them.

Numbering is historical and stable, so gaps can be referred to by number across
time. Gaps 11 to 18 were found while fixing 1 to 10.

---

## Closed

### Gap 2. The transition width was inherited rather than derived

The whole method exists to produce a non-zero gradient, that gradient lives
entirely in the transition zone, and the width of that zone was set by the UTD
distance parameter `L`, designed to make the diffracted field asymptotically
correct and never intended as a validity weight. The gradient's direction was
physically grounded while its magnitude inherited a modelling choice nobody
made with differentiability in mind.

**Fixed by** reading the width off the exact Sommerfeld half-plane solution,
where it is the Fresnel argument `s = sqrt(2kL) sin(dbeta/2)`. Implemented in
the equivalent monotone form `s = d sin(beta0) sqrt(k / 2L)`, because writing
it as `sign(x) sqrt(|x|)` puts an infinite derivative exactly at the boundary,
the one place this must be differentiable.

**Difference:** the width stops being arbitrary, which was the whole objection.
It also mattered numerically: carrying the old angular clamp caps the argument
so the weight saturates near 0.98 instead of 1, losing a few percent at every
bounce of every path and compounding to 2.8 dB across a furnished room.

### Gap 3. Single-edge diffraction only

In a deep shadow, where the first-order term is itself nearly zero and energy
arrives having bent around two edges, the field was predicted as not a small
value but exactly zero.

**Fixed by** edge-to-edge cascades, with the two-edge stationary path found by
alternating exact minimisation, which reaches the global minimum because the
total path length is jointly convex in the two edge parameters and stays
differentiable because the iteration is unrolled. Slope diffraction comes with
it, its derivative of the coefficient taken by autograd.

**Difference:** behind two knife edges in series, single diffraction predicts
exactly zero at 30 of 31 receiver heights while the cascade gives -80 to
-100 dBm, so there it is the entire answer. Inside the furnished room it is
worth under 1 dB, because a partition in a reflective room never makes a deep
shadow at all: wall reflections sit 15 to 25 dB above every diffracted term.
That was the opposite of what was expected before measuring it.

Three defects on the way, each producing a plausible wrong answer rather than
an error. Edges sharing a vertex are corners rather than cascades and their
`1/sqrt(s12)` singularity was injecting 27 dB of invented power. The distance
parameter is not reciprocal in a cascade, worth several dB and invisible to
everything except reciprocity. Masking a singularity with `torch.where` does
not stop the rejected branch being evaluated, and its backward pass multiplies
zero by infinity, giving NaN gradients while forward values stay correct.

### Gap 4. No surface roughness

The Fresnel equations assume a mirror-smooth wall, so every reflection was
overpredicted at millimetre wave, where a 1 mm rough surface is a fifth of a
wavelength instead of a sixtieth.

**Fixed by** multiplying the specular field by a roughness factor derived from
the phase spread across the surface height, using the Miller-Brown form because
the plain Rayleigh one assumes small phase variance and returns an implausible
-110 dB for brick at 60 GHz.

**Difference:** nothing at 5 GHz and nothing for the metal-versus-foam
comparison, since both are smooth, but 6 to 7 dB for concrete and brick at
60 GHz. It also revealed that the assumed surface height now matters more than
the choice of model, which is why it is a fittable parameter rather than a
fixed table value. It created gap 5's imbalance.

### Gap 7. One homogeneous slab per wall

A real partition is plasterboard, an air cavity, then plasterboard, and those
internal boundaries reflect and interfere, giving structure in frequency and
angle that no single effective permittivity can reproduce.

**Fixed by** allowing walls to be a stack of layers solved exactly by the
Abeles characteristic matrix, supplying both reflection and transmission and
differentiable in every layer thickness and permittivity.

**Difference:** a stud partition differs from a solid wall of the same total
thickness by up to 7.9 dB, with reflection nulls the solid model shows no trace
of, and on the robot route it drops the Rice K-factor by 4.2 dB while barely
touching received power. It also exposed a real bug in the old formula, which
took the phase along the slanted ray path `d/cos(theta_t)` instead of the
normal `d cos(theta_t)`, worth up to 10 dB at oblique incidence and invisible
to every existing test because they all sat at normal incidence.

### Gap 5. No diffuse scattering

Every reflection was specular, so delay spreads were underestimates and Rice
K-factors overestimates. Fixing gap 4 made it worse: the roughness term removes
energy from the specular direction and nothing put it back, so at 77 GHz over 96
percent of concrete's and brick's specular power was simply deleted.

**Fixed by** re-radiating that energy with a lobe about the specular direction,
with the scattering coefficient tied to the roughness by conservation,
`S**2 = 1 - rho**2`, so no second unmeasured parameter is introduced beside the
surface height. The amplitude follows from conservation rather than a fitted
constant, and is normalised by the lobe's solid-angle integral computed for the
actual specular elevation, not its normal-incidence value, which is what keeps
it conservative when part of the lobe falls below the horizon. Patch phases are
true path lengths, so the patches interfere rather than being summed as powers.

**Difference:** at 60 GHz on the robot route, metal is untouched at +0.14 dB,
which is the control, since a smooth surface scatters nothing. Concrete gains
10.00 dB and foam board 10.57 dB, with the Rice K-factor falling by 14.4 and
15.7 dB and the angular spread rising by 13.6 and 15.9 degrees. That is the
energy gap 4 was deleting, recovered.

**Still open inside it.** Conservation fixes the normalisation from the incident
side alone and is not symmetric under swapping the ends, while the symmetrised
version is exactly reciprocal and radiates less, because it suppresses grazing
directions where the two disagree most. Reciprocity is a theorem, so that is the
property held exact, and the cost is a measured energy deficit that depends only
on the lobe width: 1.72 dB at alpha 1, 0.96 dB at the default alpha 4, 0.34 dB at
alpha 16, 0.17 dB at alpha 32. A rigorous fix needs a scattering model whose
normalisation is symmetric by construction rather than symmetrised afterwards.
See also gap 19.

### Gap 17. No reference with a width

The half plane is exact but has no width parameter, and width is the exact
quantity the headline gradient claim is about. Agreement was autograd against
this model's own finite differences, which shows the derivative is computed
correctly, not that it is right.

**Fixed by** a two-dimensional method-of-moments strip solver, which needed no
new dependency because `torch.special` supplies the Bessel functions. Validated
three ways: the boundary condition on the conductor holds to 1.4e-15,
convergence is clean first order with ratio 2.00 and 0.27 percent residual after
Richardson, and a wide strip reproduces image theory, which fixes sign and scale
as the boundary condition alone cannot.

**Difference:** it turned the gap 1 finding from an internal comparison into one
against physics. With the strip edge crossing the specular point at 0.35 m, the
exact field there is 0.4785 of its wide-plate value, the exact weight gives
0.4794, and RFDT with its diffraction term gives 0.0352. Mean absolute error
across the transition, 0.0458 against 0.2372. The solver also independently
corroborated the half-field prediction derived from Sommerfeld.

Getting there required fixing a real defect: evaluating the radiated field on
the strip puts the observation point inside a source segment where the kernel
is logarithmically singular, so it must use the same analytic self-integral the
impedance matrix uses. Before that, a correct solution looked 70 percent wrong.

## Closed for a free edge, reshaped for the rest

### Gap 1. The damping had no derivation

RFDT replaces the hard visibility step with a smooth weight but keeps the
standard diffraction coefficient, whose cotangent terms are deliberately
singular precisely so they cancel a step that no longer exists, so they
introduce a discontinuity instead of removing one. Damping them by `|F|`
restored continuity and passed every test with no asymptotic derivation behind
it and no characterised error away from the boundary.

**Addressed by** building the exact half-plane reference and noticing that,
written out, the exact solution is *already* RFDT's construction: a geometric
field times a smooth weight, with no additive diffraction term at all. So the
weights could be compared directly instead of arguing about the damping. They
are different functions, and replacing RFDT's with the modified Fresnel integral
makes the half plane bit-exact with no diffraction term, so for a free edge the
damping question does not arise.

**Difference:** at a reflection boundary, where physics requires exactly half
the infinite-plane field, RFDT with its diffraction term gives 0.032, which is
24 dB low, and the exact weight alone gives 0.5000. It also does the paper's own
job better with less machinery: continuity jump over range 0.0039 against
0.0067, and the gradient with respect to reflector size 6.32e-3 against
4.37e-3, matching finite differences to 1.6e-8, with no diffraction term at all.
On the occlusion path `F(s) + F(-s) = 1` identically, so around and through
become a true partition of the incident energy, where the UTD pair sums to zero
at the silhouette and energy simply disappears. Across a full furnished room the
exact weight unaided matches RFDT-plus-diffraction to 0.034 dB.

**Still open:** the exactness holds for a free edge. A general wedge needs a
residual diffraction term whose correct form is underived. See gap 11.

---

## Open

### Gap 6. Scalar polarisation

One complex coefficient per path, so cross-polar coupling is identically zero
and depolarisation from rough or edged scattering is absent. The perpendicular
default is self-consistent, but a real reflection rotates polarisation and a
dual-polarised radio cannot be represented at all.

### Gap 8. Lossy and rounded wedges

UTD coefficients are derived for a perfectly conducting sharp wedge, and lossy
dielectric wedges use Luebbers' heuristic rather than a solution to the
canonical impedance-wedge problem. It is weakest for low-contrast materials,
and foam is the lowest-contrast material in the study. Real edges are also
rounded, which is a different regime again. Measure against the strip solver
before building anything; it may prove acceptable.

### Gap 9. Far-field assumption at every hop

Borderline for an antenna five wavelengths from a wall at 5 GHz, and near-field
coupling to the robot's own chassis is absent entirely. Cheap partial fix
available: compute the Fraunhofer distance per interaction and report the
fraction of paths violating it, turning an unstated assumption into a printed
number.

### Gap 10. No measured validation

Everything is validated against closed-form electromagnetics, numerically exact
solvers and internal consistency, never against measurements in a real room. The
defensible claim is that the physics is implemented correctly, not that it has
been confirmed empirically. The DLoc work is scaffolded with its comparison
thresholds fixed in writing before the data is opened, and is blocked on the
consent form and the pixels-to-metres scale.

### Gap 11. The exact weight is not the default

Two things must be settled first. The bit-exact result holds for a free edge, so
a general wedge still needs a residual diffraction term whose correct form is
underived. And `fresnel` plus diffraction gives 0.524 against the required
0.500, so the two overlap and their division of labour has to be decided rather
than left to whichever fires.

### Gap 12. The weight uses only the nearest edge

A facet has four, each contributing its own transition, and the exact treatment
superposes them. Using one leaves the ripple of a single edge and misses the
others.

### Gap 13. The slab reference plane is double counted

The slab coefficient carries the full propagation delay across a wall while the
tracer also applies `exp(-jkL)` along a ray running through it, so the
air-equivalent phase is counted twice: a fixed offset per crossing. Documented
rather than changed, because it alters the meaning of a coefficient everything
else agrees on.

### Gap 14. Layering applies to plates, not closed solids

A closed solid still uses the two-interface volume model, so the partition box
in the furnished room is homogeneous however its material is defined.

### Gap 15. Residue of the diffraction cascade

Third-order and higher diffraction are absent. The cascade's distance parameters
are symmetrised rather than derived. Pairs closer than a wavelength are rejected
rather than approximated, the rigorous answer being a joint two-edge coefficient
with a two-variable transition function. Slope diffraction is applied only at
the second edge, and the face reflection weights are frozen inside its
derivative.

### Gap 16. Second-order diffraction is unaffordable

The amplitude prune measurably rejects nothing, because a cascade's attenuation
lives in the coefficients rather than the spreading the bound is made of. Order
2 costs fifteen to twenty times a first-order trace, which is why it is off by
default. A bound on the damped coefficient itself would fix this.

### Gap 18. The surface heights are estimates

The values driving gap 4 are order-of-magnitude literature figures, not ITU-R
values and not measurements, carrying about a factor-of-two uncertainty which at
60 GHz matters more than the choice of roughness model. Mitigated by making them
fittable, not closed.

---

### Gap 19. Diffuse scattering is first order only

A facet scatters the field arriving directly from the transmitter. It does not
scatter what arrives by reflection, and a diffuse contribution cannot then go on
to reflect or diffract. In a closed reflective room those higher-order terms are
not obviously negligible, and nothing here bounds them.

---

## Tally

Six closed (2, 3, 4, 5, 7, 17), one closed for a free edge (1), twelve open
(6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19).

Two of the open ones exist because a fix created them: gap 5 was made worse by
gap 4 before being closed, and gap 19 is the residue of closing it.
