"""Material electromagnetics for the RFDT simulator.

Implements Appendix E.2 of "Physically Accurate Differentiable Inverse Rendering
for Radio Frequency Digital Twin" (RFDT, MobiCom '26):

  * complex relative permittivity of building materials,
  * intrinsic impedance and Snell refraction         (Eq. 56, 57),
  * Fresnel reflection / transmission coefficients   (Eq. 56),
  * complex propagation constant, attenuation        (Eq. 58, 59),
  * a surface-roughness correction to the specular reflection, which is *not*
    in RFDT and is added here (see ``roughness_factor``).

Every quantity is a differentiable ``torch`` expression of the material
parameters ``(eps_r_real, sigma)``, so that
``d Gamma / d eps_r``, ``d T / d eps_r`` and ``d exp(-alpha d) / d eps_r``
of Eq. 60 are available by autograd.

Material parameters follow the ITU-R P.2040-1 regression

    eps' = a * f_GHz**b          sigma = c * f_GHz**d   [S/m]

Entries flagged ``source="approx"`` are *not* from ITU-R P.2040; they are
order-of-magnitude literature values used for the obstacle / human-body
scenarios and are labelled as such wherever they are reported.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

# ---------------------------------------------------------------------------
# physical constants (SI)
# ---------------------------------------------------------------------------
C0 = 299_792_458.0          # speed of light            [m/s]
EPS0 = 8.854_187_8128e-12   # vacuum permittivity       [F/m]
MU0 = 1.256_637_062_12e-6   # vacuum permeability       [H/m]
ETA0 = 376.730_313_668      # vacuum wave impedance     [Ohm]

CDTYPE = torch.complex128
FDTYPE = torch.float64


def _t(x, dtype=FDTYPE) -> torch.Tensor:
    """Coerce a scalar or tensor to a torch tensor of the requested dtype."""
    if isinstance(x, torch.Tensor):
        return x.to(dtype) if x.dtype != dtype else x
    return torch.tensor(x, dtype=dtype)


# ---------------------------------------------------------------------------
# material model
# ---------------------------------------------------------------------------
@dataclass
class Material:
    """An RF material.

    The dielectric behaviour is defined either by the ITU-R P.2040-1
    regression coefficients ``(a, b, c, d)`` or, when ``tan_delta`` is given,
    by a constant permittivity with a constant loss tangent.

    Attributes
    ----------
    name        : identifier used in scenes and result tables.
    a, b        : eps' = a * f_GHz**b.
    c, d        : sigma = c * f_GHz**d   [S/m].
    tan_delta   : if not None, sigma is derived as w*eps0*eps'*tan_delta.
    mu_r        : relative permeability (1.0 for all non-magnetic materials).
    thickness   : default slab thickness [m], used for penetration loss.
    f_range     : validity range of the regression [GHz] (reported, not enforced).
    source      : "ITU-R P.2040-1" or "approx" (see module docstring).
    roughness_sigma : root-mean-square surface height [m], 0.0 for an ideally
                  smooth surface.  Drives the specular reduction computed by
                  ``roughness_factor``.  These are order-of-magnitude values
                  from the building-materials literature, NOT measurements of
                  the particular surfaces simulated here, so they are reported
                  as estimates wherever they influence a result.  Being
                  differentiable, the value can also be recovered from data
                  rather than trusted.
    """

    name: str
    a: float = 1.0
    b: float = 0.0
    c: float = 0.0
    d: float = 0.0
    tan_delta: Optional[float] = None
    mu_r: float = 1.0
    thickness: float = 0.15
    f_range: Tuple[float, float] = (1.0, 100.0)
    source: str = "ITU-R P.2040-1"
    label: str = ""
    #: appended last so that existing positional constructions are unaffected
    roughness_sigma: float = 0.0

    def __post_init__(self):
        """Coerce the position to a float64 tensor."""
        if not self.label:
            self.label = self.name.replace("_", " ").title()

    # primitive frequency-dependent parameters -----------------------------
    def eps_real(self, f_hz) -> torch.Tensor:
        """Real part of the relative permittivity eps'."""
        f_ghz = _t(f_hz) / 1e9
        return _t(self.a) * f_ghz ** _t(self.b)

    def sigma(self, f_hz) -> torch.Tensor:
        """Equivalent conductivity [S/m]."""
        f_ghz = _t(f_hz) / 1e9
        if self.tan_delta is not None:
            omega = 2.0 * torch.pi * _t(f_hz)
            return omega * EPS0 * self.eps_real(f_hz) * _t(self.tan_delta)
        return _t(self.c) * f_ghz ** _t(self.d)

    # derived quantities ---------------------------------------------------
    def eps_complex(self, f_hz, params: Optional["MaterialParams"] = None) -> torch.Tensor:
        """Complex relative permittivity ``eps_r = eps' - j*sigma/(w*eps0)``.

        ``params`` optionally overrides ``(eps', sigma)`` with learnable
        tensors, which is how the digital twin optimises material properties
        (Sec. 5.1 / Eq. 60).
        """
        f = _t(f_hz)
        omega = 2.0 * torch.pi * f
        if params is not None:
            epsr, sig = params.eps_real, params.sigma
        else:
            epsr, sig = self.eps_real(f), self.sigma(f)
        epsr = _t(epsr).to(CDTYPE)
        sig = _t(sig).to(CDTYPE)
        return epsr - 1j * sig / (omega.to(CDTYPE) * EPS0)

    def impedance(self, f_hz, params=None) -> torch.Tensor:
        """Intrinsic impedance, Eq. 56: ``eta = sqrt(j*w*mu / (sigma + j*w*eps))``."""
        f = _t(f_hz)
        omega = (2.0 * torch.pi * f).to(CDTYPE)
        eps_c = self.eps_complex(f, params) * EPS0        # absolute [F/m]
        mu = _t(self.mu_r).to(CDTYPE) * MU0
        return torch.sqrt(1j * omega * mu / (1j * omega * eps_c))

    def refractive_index(self, f_hz, params=None) -> torch.Tensor:
        """Complex refractive index, Eq. 57: ``n = sqrt(eps_r * mu_r)``."""
        return torch.sqrt(self.eps_complex(f_hz, params) * _t(self.mu_r).to(CDTYPE))

    def propagation_constant(self, f_hz, params=None) -> torch.Tensor:
        """Complex propagation constant gamma = alpha + j*beta, Eq. 58."""
        f = _t(f_hz)
        omega = (2.0 * torch.pi * f).to(CDTYPE)
        return 1j * omega * self.refractive_index(f, params) / C0

    def attenuation(self, f_hz, params=None) -> torch.Tensor:
        """Attenuation constant alpha [Np/m] of Eq. 58/59."""
        return self.propagation_constant(f_hz, params).real

    def penetration_loss_db(self, f_hz, thickness=None, params=None) -> torch.Tensor:
        """One-way absorption loss through a slab, from Eq. 59 (|E| = |E0|e^{-a z})."""
        d = _t(self.thickness if thickness is None else thickness)
        alpha = self.attenuation(f_hz, params)
        return 20.0 * torch.log10(torch.exp(alpha * d))

    def in_validity_range(self, f_hz) -> bool:
        """Whether ``f_hz`` lies inside the regression's stated validity range.

        ITU-R P.2040-1 fits each material over a specific band; outside it the
        extrapolation is not supported by the recommendation.  Results are
        still produced, but callers should report the flag rather than quietly
        present an extrapolation as a tabulated value.
        """
        f_ghz = float(_t(f_hz)) / 1e9
        return self.f_range[0] <= f_ghz <= self.f_range[1]

    # reporting ------------------------------------------------------------
    def describe(self, f_hz) -> Dict[str, float]:
        """Human-readable summary of the material's properties at ``f_hz``."""
        eps_c = self.eps_complex(f_hz)
        return {
            "name": self.name,
            "eps_real": float(self.eps_real(f_hz)),
            "sigma_S_per_m": float(self.sigma(f_hz)),
            "eps_imag": float(-eps_c.imag),
            "loss_tangent": float(-eps_c.imag / eps_c.real),
            "alpha_Np_per_m": float(self.attenuation(f_hz)),
            "roughness_sigma_m": float(self.roughness_sigma),
            "rayleigh_param_normal": float(
                2.0 * torch.pi * _t(f_hz) / C0 * self.roughness_sigma),
            "is_rough_at_normal_incidence": bool(
                float(2.0 * torch.pi * _t(f_hz) / C0 * self.roughness_sigma)
                > RAYLEIGH_SMOOTH_LIMIT),
            "source": self.source,
            "f_range_ghz": list(self.f_range),
            "in_validity_range": self.in_validity_range(f_hz),
        }


@dataclass
class MaterialParams:
    """Learnable override of a material's ``(eps', sigma)`` pair.

    Optimising in log-space keeps ``sigma`` positive and conditions the
    gradient, which matters because sigma spans ~10 orders of magnitude
    between foam and metal.
    """

    log_eps_real: torch.Tensor
    log_sigma: torch.Tensor
    #: optional third parameter, the log root-mean-square surface height [m].
    #: ``None`` leaves roughness fixed at the material's tabulated value, which
    #: keeps every existing two-parameter inversion behaving exactly as before.
    log_roughness: Optional[torch.Tensor] = None

    @staticmethod
    def from_material(mat: Material, f_hz: float, requires_grad: bool = True,
                      learn_roughness: bool = False,
                      roughness_init: Optional[float] = None) -> "MaterialParams":
        """Initialise learnable parameters from an existing material at ``f_hz``.

        ``learn_roughness`` adds the surface height to the optimisation.  Note
        that the roughness factor is flat at ``sigma_h = 0`` (its derivative
        vanishes there, because the factor depends on ``sigma_h`` only through
        its square), so a material tabulated as perfectly smooth cannot be
        learned away from zero.  ``roughness_init`` supplies a non-zero start
        for that case; without it a zero-roughness material stays at zero and
        the gradient is legitimately, not erroneously, zero.
        """
        e = torch.log(_t(mat.eps_real(f_hz)).clone().detach())
        s = torch.log(_t(mat.sigma(f_hz)).clone().detach().clamp_min(1e-12))
        r = None
        if learn_roughness:
            init = mat.roughness_sigma if roughness_init is None else roughness_init
            r = torch.log(_t(init).clone().detach().clamp_min(1e-9))
            r = r.requires_grad_(requires_grad)
        return MaterialParams(
            log_eps_real=e.requires_grad_(requires_grad),
            log_sigma=s.requires_grad_(requires_grad),
            log_roughness=r,
        )

    @property
    def eps_real(self) -> torch.Tensor:
        """Current permittivity estimate, exponentiated out of log space."""
        return torch.exp(self.log_eps_real)

    @property
    def sigma(self) -> torch.Tensor:
        """Current conductivity estimate [S/m], exponentiated out of log space."""
        return torch.exp(self.log_sigma)

    @property
    def roughness(self) -> Optional[torch.Tensor]:
        """Current surface-height estimate [m], or None if it is not learned."""
        if self.log_roughness is None:
            return None
        return torch.exp(self.log_roughness)

    def tensors(self):
        """The leaf tensors to hand to an optimiser."""
        t = [self.log_eps_real, self.log_sigma]
        if self.log_roughness is not None:
            t.append(self.log_roughness)
        return t


# ---------------------------------------------------------------------------
# Fresnel coefficients (Eq. 56 / 57)
# ---------------------------------------------------------------------------
def snell_cos_theta_t(n1: torch.Tensor, n2: torch.Tensor, cos_ti: torch.Tensor) -> torch.Tensor:
    """cos(theta_t) from Snell's law with complex indices (Eq. 57)."""
    n1 = n1.to(CDTYPE)
    n2 = n2.to(CDTYPE)
    sin2_ti = (1.0 - cos_ti.to(CDTYPE) ** 2)
    return torch.sqrt(1.0 - (n1 / n2) ** 2 * sin2_ti)


def fresnel(
    f_hz,
    cos_ti: torch.Tensor,
    mat2: Material,
    mat1: Optional[Material] = None,
    params2: Optional[MaterialParams] = None,
) -> Dict[str, torch.Tensor]:
    """Fresnel reflection / transmission at a medium-1 -> medium-2 interface.

    Parameters
    ----------
    cos_ti : cosine of the incidence angle measured from the surface normal.
    mat2   : material behind the interface, mat1 defaults to vacuum.

    Returns ``{"gamma_perp", "gamma_par", "tau_perp", "tau_par",
    "cos_tt", "eta1", "eta2"}`` following Eq. 56.
    """
    mat1 = mat1 or VACUUM
    cos_ti = cos_ti.to(CDTYPE)
    eta1 = mat1.impedance(f_hz).to(CDTYPE)
    eta2 = mat2.impedance(f_hz, params2).to(CDTYPE)
    n1 = mat1.refractive_index(f_hz)
    n2 = mat2.refractive_index(f_hz, params2)
    cos_tt = snell_cos_theta_t(n1, n2, cos_ti)

    # Eq. 56 (perpendicular / s and parallel / p polarisation)
    g_perp = (eta2 * cos_ti - eta1 * cos_tt) / (eta2 * cos_ti + eta1 * cos_tt)
    g_par = (eta1 * cos_ti - eta2 * cos_tt) / (eta1 * cos_ti + eta2 * cos_tt)
    # field transmission coefficients (tau = 1 + gamma for the s case)
    t_perp = 2.0 * eta2 * cos_ti / (eta2 * cos_ti + eta1 * cos_tt)
    t_par = 2.0 * eta1 * cos_ti / (eta1 * cos_ti + eta2 * cos_tt)
    return {
        "gamma_perp": g_perp,
        "gamma_par": g_par,
        "tau_perp": t_perp,
        "tau_par": t_par,
        "cos_tt": cos_tt,
        "eta1": eta1,
        "eta2": eta2,
    }


# ---------------------------------------------------------------------------
# surface roughness (an addition to RFDT, not part of the paper)
# ---------------------------------------------------------------------------
def roughness_factor(
    f_hz,
    cos_ti: torch.Tensor,
    sigma_h,
    model: str = "miller_brown",
) -> torch.Tensor:
    """Reduction of the *coherent* specular field by surface roughness.

    Why this is needed
    ------------------
    The Fresnel coefficients of Eq. 56 are derived for an ideally smooth,
    infinite, homogeneous interface.  Real building surfaces are not smooth:
    paint texture, wallpaper, mortar lines and screw heads give a root-mean-
    square surface height of roughly 0.1 mm to 2 mm.  Whether that matters is
    decided by the wavelength, so it is negligible at 5 GHz (lambda = 6 cm) and
    substantial at 60 GHz (lambda = 5 mm).  A simulator that ignores it will
    overpredict every specular bounce in the millimetre-wave band, and it will
    do so silently.

    Where the expression comes from
    -------------------------------
    Consider two rays striking the surface a small distance apart, one on a
    bump of height ``h`` and one in a trough.  Both travel down to the surface
    and back up, so the extra path length of the bump ray is ``2 h cos(theta)``
    and its extra phase is ``2 k h cos(theta)``.  Take the surface heights to be
    zero-mean Gaussian with standard deviation ``sigma_h``.  The phase is then
    Gaussian with standard deviation

        s = 2 k sigma_h cos(theta),

    and the coherent (phasor-averaged) field is the expectation of
    ``exp(j*phi)`` over that distribution.  For a Gaussian phase this
    expectation is ``exp(-s**2 / 2)``, which gives the Ament, or Rayleigh,
    factor

        rho = exp(-2 (k sigma_h cos(theta))**2).

    Read it as a coherence loss: the rougher the surface in wavelengths, the
    more the contributions from different surface heights disagree in phase and
    cancel, and the less energy survives in the mirror direction.

    ``model="miller_brown"`` uses the refinement

        rho = exp(-x) * I0(x),        x = 2 (k sigma_h cos(theta))**2,

    where ``I0`` is the modified Bessel function of the first kind.  The plain
    Rayleigh form assumes the surface is flat on the scale of a Fresnel zone
    apart from its height variation; Miller and Brown account for the fact that
    a rough surface still returns some energy towards the specular direction
    from its favourably tilted facets, so it decays more slowly and does not
    drive the coherent field to zero.  It is the less severe of the two and is
    the better-validated model at large roughness.  It is evaluated through
    ``torch.special.i0e``, which computes ``exp(-x) * I0(x)`` as a single
    stable quantity, so no overflow occurs for large ``x``.

    Both limits are correct by construction: ``rho -> 1`` as ``sigma_h -> 0``
    (a smooth surface), and ``rho -> 1`` at grazing incidence where
    ``cos(theta) -> 0``, because a wave skimming along the surface barely
    resolves its height variation at all.

    What this deliberately does NOT do
    ----------------------------------
    The energy removed from the specular direction is scattered elsewhere, not
    absorbed.  This function removes it and nothing puts it back, because
    diffuse scattering is not modelled here.  The consequence is a *known and
    signed* bias: total received power is underpredicted once the roughness
    factor departs from unity, most visibly in the millimetre-wave band and on
    multi-bounce paths.  That is a deliberate, documented limitation rather
    than a hidden one; adding it back requires a diffuse scattering term, which
    is separate work.  Coherent transmission through a rough interface is
    likewise still computed as if the interface were smooth.

    Parameters
    ----------
    cos_ti  : cosine of the incidence angle measured from the surface normal.
    sigma_h : root-mean-square surface height [m], scalar or tensor.  May carry
              a gradient, which is how it is recovered by inversion.
    model   : "miller_brown" (default), "rayleigh", or "none" to disable.

    Why the default is Miller-Brown
    -------------------------------
    Chosen on validity range, before looking at what either model does to any
    result.  The Rayleigh form assumes the two-way phase variance is small, so
    that a Gaussian average is the whole story.  Its accuracy therefore decays
    as ``g = k sigma_h cos(theta)`` grows, and it decays in a known direction:
    it over-predicts the coherence loss, because it credits nothing to the
    energy a rough surface still returns near the specular direction from its
    favourably tilted facets.  Miller-Brown restores exactly that term.

    The two agree to within 1e-4 for ``g < 0.15``, which covers every material
    in the library at 5 GHz, so at that band the choice is immaterial.  At
    60 GHz ``g`` reaches 2.5 for brick, far outside the small-variance regime,
    and there the Rayleigh form returns a specular level near -110 dB, which is
    not a credible reflection from a brick wall.  Defaulting to the model whose
    derivation still holds over the range actually encountered is the
    conservative choice; ``"rayleigh"`` remains available and the difference
    between them is reported as an uncertainty, not hidden behind a default.
    """
    if model == "none":
        return torch.ones((), dtype=FDTYPE)
    c = torch.as_tensor(cos_ti)
    c = c.real if c.is_complex() else c
    c = c.to(FDTYPE).abs()
    k = 2.0 * torch.pi * _t(f_hz) / C0
    # g is the Rayleigh roughness parameter, the surface height in radians of
    # two-way phase; x is the variance of that phase.
    g = k * _t(sigma_h) * c
    x = 2.0 * g ** 2
    if model == "rayleigh":
        return torch.exp(-x)
    if model == "miller_brown":
        return torch.special.i0e(x)
    raise ValueError(
        "unknown roughness model %r; expected 'rayleigh', 'miller_brown' or "
        "'none'" % (model,))


#: Rayleigh smoothness criterion.  A surface counts as smooth while
#: ``k sigma_h cos(theta) < pi/4``, equivalently ``sigma_h < lambda/(8 cos)``,
#: the classical threshold at which the two-way phase spread across the surface
#: height reaches a quarter cycle.
RAYLEIGH_SMOOTH_LIMIT = float(torch.pi / 4.0)


def roughness_regime(f_hz, cos_ti, sigma_h, model: str = "miller_brown"):
    """Report whether a surface is in the regime where the missing diffuse
    term dominates, rather than silently returning a specular-only number.

    ``roughness_factor`` removes energy from the specular direction, and
    nothing in this simulator puts it back, because diffuse scattering is not
    modelled.  While the surface is smooth that removed fraction is negligible
    and the specular field is the whole answer.  Once the surface is rough the
    removed fraction is most of the energy, so the returned field is a small
    residue of a quantity the simulator no longer represents, and quoting it as
    a predicted echo would be misleading in exactly the way an out-of-range
    ITU-R extrapolation would be.

    The boundary used is the classical Rayleigh criterion,
    ``k sigma_h cos(theta) = pi/4``, not a threshold invented here.

    Returns ``{"g", "specular_loss_db", "is_rough", "unmodelled_fraction"}``,
    where ``unmodelled_fraction`` is the share of incident power taken out of
    the specular direction and not re-radiated anywhere.  Callers are expected
    to report the flag beside the number, as experiment 3 does.
    """
    rho = roughness_factor(f_hz, cos_ti, sigma_h, model)
    c = torch.as_tensor(cos_ti)
    c = (c.real if c.is_complex() else c).to(FDTYPE).abs()
    g = 2.0 * torch.pi * _t(f_hz) / C0 * _t(sigma_h) * c
    return {
        "g": float(g),
        "specular_loss_db": float(-20.0 * torch.log10(rho.clamp_min(1e-30))),
        "is_rough": bool(float(g) > RAYLEIGH_SMOOTH_LIMIT),
        "unmodelled_fraction": float(1.0 - rho ** 2),
    }


def _roughness_sigma(mat: "Material", params: Optional["MaterialParams"]):
    """Surface height to use: the learnable one when present, else tabulated."""
    if params is not None and params.roughness is not None:
        return params.roughness
    return mat.roughness_sigma


def reflection_coefficient(
    f_hz,
    cos_ti: torch.Tensor,
    mat: Material,
    polarisation: str = "perp",
    params: Optional[MaterialParams] = None,
    roughness: str = "miller_brown",
) -> torch.Tensor:
    """Scalar reflection coefficient used by the ray tracer.

    ``polarisation``:
      ``"perp"`` (default) is the TE / horizontal coefficient.  It is the
      standard choice for a scalar indoor model because it behaves correctly
      at both ends of the incidence range: ``Gamma -> -1`` for a conductor and
      ``Gamma -> -1`` at grazing incidence for any material.
      ``"par"`` is the TM / vertical coefficient.
      ``"unpolarised"`` keeps the power average of the two magnitudes,
      ``sqrt((|G_s|^2 + |G_p|^2)/2)``, and takes the phase from ``G_s``.

    Note on ``"unpolarised"``: a single complex scalar cannot represent both
    polarisations faithfully, because ``G_s`` and ``G_p`` are referenced to
    opposite field directions, so their phases differ by pi at normal
    incidence and agree at grazing.  Any scalar blend of the phases is
    therefore ill-defined somewhere in between; taking the ``G_s`` phase is a
    documented approximation.  A strict treatment needs dual-polarised path
    tracking, which is outside this scalar model.

    ``roughness`` selects the surface-roughness model applied on top of the
    smooth-interface Fresnel value; see ``roughness_factor``.  It multiplies
    both polarisations by the same real factor, which is correct to first
    order because the coherence loss is a path-length effect and does not
    distinguish the two field orientations.  Pass ``"none"`` to recover the
    ideally smooth coefficient, which is what every ablation against the
    previous behaviour uses.
    """
    fr = fresnel(f_hz, cos_ti, mat, params2=params)
    gs, gp = fr["gamma_perp"], fr["gamma_par"]
    rho = roughness_factor(f_hz, cos_ti, _roughness_sigma(mat, params),
                           roughness).to(CDTYPE)
    if polarisation == "perp":
        return gs * rho
    if polarisation == "par":
        return gp * rho
    mag = torch.sqrt(0.5 * (gs.abs() ** 2 + gp.abs() ** 2))
    phase = torch.angle(gs)
    return mag.to(CDTYPE) * torch.exp(1j * phase.to(CDTYPE)) * rho


def interface_transmission(
    f_hz,
    cos_ti: torch.Tensor,
    mat: Material,
    entering: Optional[torch.Tensor] = None,
    polarisation: str = "perp",
    params: Optional[MaterialParams] = None,
) -> torch.Tensor:
    """Field transmission across a *single* air / material interface.

    Used for closed solids, where a ray crosses two distinct faces: it enters
    through one and leaves through the other.  Modelling each face with a full
    slab formula would count four interfaces and two thicknesses instead of
    two and one.

    ``entering`` is a boolean tensor, true where the ray goes into the material
    (its direction opposes the outward face normal); the air -> material
    coefficient is then used, and material -> air otherwise.

    With ``entering=None`` (the default) the direction-independent geometric
    mean ``sqrt(tau_in * tau_out)`` is returned instead.  Over a full traversal
    the two crossed faces still multiply to exactly ``tau_in * tau_out``, so
    nothing is lost, but a ray that merely clips one face near a solid's
    silhouette now gets the same factor whichever way it travels.  That keeps
    the simulator reciprocal, which the direction-aware form is not: with it,
    swapping transmitter and receiver changes the predicted power by a few
    hundredths of a dB on grazing paths.
    """
    fr = fresnel(f_hz, cos_ti, mat, params2=params)
    g = fr["gamma_par"] if polarisation == "par" else fr["gamma_perp"]
    tau_in = fr["tau_par"] if polarisation == "par" else fr["tau_perp"]
    tau_out = 1.0 - g                       # material -> air, by symmetry
    if entering is None:
        return torch.sqrt(tau_in * tau_out)
    return torch.where(entering, tau_in, tau_out)


def absorption_factor(
    f_hz,
    path_length: torch.Tensor,
    mat: Material,
    cos_tt: Optional[torch.Tensor] = None,
    params: Optional[MaterialParams] = None,
) -> torch.Tensor:
    """Complex ``exp(-gamma * d)`` accumulated along a path inside a medium.

    This is Eq. 58-59 applied to an explicit traversal length, which is how
    App. E.2 handles propagation through a solid object: "the simulator
    accumulates the phase shift and attenuation along the ray path of length d
    through the medium".
    """
    gamma = mat.propagation_constant(f_hz, params)
    d = _t(path_length).to(CDTYPE)
    if cos_tt is not None:
        d = d / cos_tt.to(CDTYPE)
    return torch.exp(-gamma * d)


def slab_transmission(
    f_hz,
    cos_ti: torch.Tensor,
    mat: Material,
    thickness: Optional[float] = None,
    polarisation: str = "perp",
    params: Optional[MaterialParams] = None,
    coherent: bool = True,
) -> torch.Tensor:
    """Field transmission through a homogeneous slab of finite thickness.

    Combines the two interface coefficients of Eq. 56 with the internal
    absorption / phase term ``exp(-gamma*d)`` of Eq. 58-59.  With
    ``coherent=True`` the multiple internal reflections are summed in closed
    form (an Airy / Fabry-Perot series), which is the "piecewise propagation
    applied with multiple segments contributing multiplicatively" of App. E.2.
    """
    d = _t(mat.thickness if thickness is None else thickness)
    fr = fresnel(f_hz, cos_ti, mat, params2=params)
    if polarisation == "par":
        g, t_in = fr["gamma_par"], fr["tau_par"]
    else:
        g, t_in = fr["gamma_perp"], fr["tau_perp"]
    cos_tt = fr["cos_tt"]

    gamma = mat.propagation_constant(f_hz, params)          # alpha + j*beta
    # oblique path length inside the slab
    phi = gamma * d / cos_tt
    # medium-2 -> medium-1 interface: reflection is -g, so transmission is 1-g
    t_out = 1.0 - g
    single = t_in * t_out * torch.exp(-phi)
    if not coherent:
        return single
    return single / (1.0 - (g ** 2) * torch.exp(-2.0 * phi))


# ---------------------------------------------------------------------------
# material library
# ---------------------------------------------------------------------------
VACUUM = Material("vacuum", a=1.0, b=0.0, c=0.0, d=0.0, thickness=0.0,
                  f_range=(0.0, 1e6), label="Vacuum")

#: Root-mean-square surface height [m] per material.
#:
#: IMPORTANT: unlike the (a, b, c, d) permittivity regressions above, these are
#: NOT from ITU-R P.2040-1, which tabulates dielectric properties only and
#: treats roughness qualitatively.  They are order-of-magnitude values from the
#: building-materials literature, chosen to represent an ordinary interior
#: surface of each kind, and they carry roughly a factor-of-two uncertainty.
#: They are reported as estimates wherever they affect a result, and
#: ``roughness_factor`` is differentiable in them so they can be recovered from
#: measurements instead of trusted.
#:
#: Scale for reading them: the specular loss at normal incidence is
#: 20*log10(exp(-2*(k*sigma)**2)), which for sigma = 1 mm is 0.19 dB at 5 GHz
#: and 27 dB at 60 GHz.  Everything below 0.2 mm is effectively smooth even at
#: 60 GHz.
ROUGHNESS_SIGMA_M = {
    "metal":              0.00005,   # rolled or brushed sheet, near-optical finish
    "glass":              0.00002,   # float glass, smooth on any RF scale
    "marble":             0.00005,   # polished stone
    "plastic_board":      0.00010,   # moulded sheet
    "plasterboard":       0.00020,   # painted drywall, paint texture dominates
    "paper_board":        0.00020,
    "floorboard":         0.00030,   # sealed timber floor
    "wood":               0.00030,   # planed and finished timber
    "foam_board":         0.00030,   # extruded foam with a smooth skin
    "plywood":            0.00040,   # visible grain
    "ceiling_board":      0.00050,   # mineral-fibre tile, textured face
    "chipboard":          0.00050,   # pressed chip texture
    "concrete":           0.00100,   # smooth-formed; board-marked runs higher
    "carpet":             0.00200,   # pile, and only crudely a rough surface
    "brick":              0.00200,   # mortar lines dominate over the face
    # Deliberately zero, and NOT a claim that a body is smooth.  The Rayleigh
    # factor is derived for a planar interface carrying small-scale height
    # variation.  A human body violates the planar assumption at a scale far
    # larger than any surface roughness, and its return is governed by
    # curvature and volumetric scattering that this model does not represent.
    # Assigning a plausible-looking clothing roughness here would let an
    # invented number swing the 77 GHz radar echo by more than 20 dB while
    # applying a correction outside its regime.  The entry is therefore left
    # uncorrected and flagged in the limitations instead.
    "human_body":         0.00000,
    "very_dry_ground":    0.00500,
    "medium_dry_ground":  0.00500,
    "wet_ground":         0.00500,
}

#: ITU-R P.2040-1 Table 3 regression coefficients.
ITU_MATERIALS = {
    "concrete":      Material("concrete",      5.24,  0.0, 0.0462,  0.7822, thickness=0.30, f_range=(1, 100)),
    "brick":         Material("brick",         3.91,  0.0, 0.0238,  0.1600, thickness=0.20, f_range=(1, 40)),
    "plasterboard":  Material("plasterboard",  2.73,  0.0, 0.0085,  0.9395, thickness=0.013, f_range=(1, 100)),
    "wood":          Material("wood",          1.99,  0.0, 0.0047,  1.0718, thickness=0.03, f_range=(0.001, 100)),
    "glass":         Material("glass",         6.31,  0.0, 0.0036,  1.3394, thickness=0.006, f_range=(0.1, 100)),
    "ceiling_board": Material("ceiling_board", 1.48,  0.0, 0.0011,  1.0750, thickness=0.0095, f_range=(1, 100)),
    "chipboard":     Material("chipboard",     2.58,  0.0, 0.0217,  0.7800, thickness=0.018, f_range=(1, 100)),
    "plywood":       Material("plywood",       2.71,  0.0, 0.3300,  0.0000, thickness=0.019, f_range=(1, 40)),
    "marble":        Material("marble",        7.074, 0.0, 0.0055,  0.9262, thickness=0.02, f_range=(1, 60)),
    "floorboard":    Material("floorboard",    3.66,  0.0, 0.0044,  1.3515, thickness=0.02, f_range=(50, 100)),
    "metal":         Material("metal",         1.0,   0.0, 1.0e7,   0.0000, thickness=0.002, f_range=(1, 100)),
    "very_dry_ground":    Material("very_dry_ground",    3.0,  0.00, 0.00015, 2.52, thickness=1.0, f_range=(1, 10)),
    "medium_dry_ground":  Material("medium_dry_ground", 15.0, -0.10, 0.035,   1.63, thickness=1.0, f_range=(1, 10)),
    "wet_ground":         Material("wet_ground",       30.0, -0.40, 0.150,   1.30, thickness=1.0, f_range=(1, 10)),
}

#: Order-of-magnitude literature values, NOT from ITU-R P.2040.
#: Used for the NLOS obstacle boards of Fig. 15(a) and for human bodies.
APPROX_MATERIALS = {
    "plastic_board": Material("plastic_board", a=2.50, tan_delta=0.005, thickness=0.007,
                              source="approx", label="Plastic board (0.7 cm)"),
    "paper_board":   Material("paper_board",   a=2.20, tan_delta=0.040, thickness=0.003,
                              source="approx", label="Paper board (0.3 cm)"),
    "foam_board":    Material("foam_board",    a=1.06, tan_delta=0.001, thickness=0.020,
                              source="approx", label="Foam board (2 cm)"),
    "human_body":    Material("human_body",    a=180.0, b=-0.72, c=0.90, d=0.72, thickness=0.10,
                              source="approx", label="Human body (tissue-like)"),
    "carpet":        Material("carpet",        a=1.60, tan_delta=0.020, thickness=0.01,
                              source="approx", label="Carpet"),
}

MATERIALS: Dict[str, Material] = {"vacuum": VACUUM, **ITU_MATERIALS, **APPROX_MATERIALS}

# Attach the surface heights.  Done as a separate pass rather than inline so
# that the dielectric tables above stay a verbatim transcription of ITU-R
# P.2040-1 Table 3, with nothing of a different provenance mixed into them.
for _name, _sig in ROUGHNESS_SIGMA_M.items():
    MATERIALS[_name].roughness_sigma = _sig
_missing = sorted(set(MATERIALS) - set(ROUGHNESS_SIGMA_M) - {"vacuum"})
assert not _missing, "no roughness estimate for: %s" % (_missing,)
del _name, _sig, _missing


def get_material(name: str) -> Material:
    """Look up a material by name, with a helpful error listing the options."""
    try:
        return MATERIALS[name]
    except KeyError as exc:  # pragma: no cover
        raise KeyError(f"unknown material '{name}'; available: {sorted(MATERIALS)}") from exc


#: Coarse category grouping used for the per-category summaries of Fig. 16(a).
MATERIAL_CATEGORIES = {
    "metal": ["metal"],
    "dielectric_dense": ["concrete", "brick", "marble"],
    "indoor_light": ["plasterboard", "ceiling_board", "chipboard", "plywood", "wood", "floorboard"],
    "glass": ["glass"],
    "human": ["human_body"],
    "low_density": ["foam_board", "paper_board", "plastic_board", "carpet"],
}