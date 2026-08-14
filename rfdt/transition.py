"""UTD transition function and wedge diffraction: the core of RFDT.

RFDT replaces the discontinuous binary visibility test of classical ray
tracing (Eq. 3) with the *physically grounded* UTD transition function
(Eq. 8),

    F(x) = 2j sqrt(x) e^{jx} \\int_{sqrt(x)}^{\\infty} e^{-j tau^2} d tau ,

whose closed form (Eq. 44) is

    F(x) = j sqrt(pi) e^{-j pi/4} sqrt(x) e^{jx} erfc(e^{j pi/4} sqrt(x)) .

Two facts make this the right object (App. D.3):

    F(x) -> 1                       as x -> inf   (deep lit region)
    F(x) ~ sqrt(pi) e^{j pi/4} sqrt(x)  as x -> 0+ (edge / shadow boundary)

so the specular contribution decays *continuously* to zero as the reflection
point walks off a triangle, while the diffracted field of Eq. 6 rises to
replace it, giving continuity plus energy conservation (Fig. 3, right).

Implementation notes
--------------------
``F`` is evaluated through the Faddeeva function ``w(z) = e^{-z^2} erfc(-jz)``
using Weideman's spectral rational approximation, which is accurate to
~1e-14 in the upper half plane (where our arguments always live, since
``z = e^{j 3 pi/4} sqrt(x)`` has ``Im z = sqrt(x/2) >= 0``).

The backward pass is a *hand-written* analytic derivative rather than
autograd through the rational approximation (Sec. 5.3, "we manually implement
custom backward passes"):

    F'(x) = F(x) (1 + 2jx) / (2x) - j ,

derived by differentiating Eq. 8 under the integral sign.  Near x = 0 the
``1/x`` is regularised with the small-argument series of Eq. 49.
"""

from __future__ import annotations

import numpy as np
import torch

CDTYPE = torch.complex128
FDTYPE = torch.float64

_SQRT_PI = float(np.sqrt(np.pi))
_EXP_JPI4 = complex(np.exp(1j * np.pi / 4))
_EXP_MJPI4 = complex(np.exp(-1j * np.pi / 4))
_EXP_J3PI4 = complex(np.exp(3j * np.pi / 4))


# ---------------------------------------------------------------------------
# Faddeeva function, Weideman (1994) spectral approximation
# ---------------------------------------------------------------------------
def _weideman_coeffs(n: int = 32):
    """Polynomial coefficients of Weideman's rational approximation to w(z)."""
    m = 2 * n
    m2 = 2 * m
    k = np.arange(-m + 1, m)          # 2M-1 points
    ell = np.sqrt(n / np.sqrt(2.0))
    theta = k * np.pi / m
    t = ell * np.tan(theta / 2.0)
    f = np.exp(-(t ** 2)) * (ell ** 2 + t ** 2)
    f = np.concatenate(([0.0], f))
    a = np.real(np.fft.fft(np.fft.fftshift(f))) / m2
    a = np.ascontiguousarray(a[1: n + 1][::-1])   # descending powers, as for polyval
    return ell, a


_WEID_N = 32
_WEID_L, _WEID_A = _weideman_coeffs(_WEID_N)


def faddeeva(z: torch.Tensor) -> torch.Tensor:
    """Faddeeva function ``w(z) = e^{-z^2} erfc(-j z)`` for ``Im(z) >= 0``."""
    z = z.to(CDTYPE)
    ell = _WEID_L
    coeffs = torch.as_tensor(_WEID_A, dtype=FDTYPE, device=z.device).to(CDTYPE)
    denom = ell - 1j * z
    Z = (ell + 1j * z) / denom
    # Horner evaluation of the descending-power polynomial
    p = torch.zeros_like(Z)
    for c in coeffs:
        p = p * Z + c
    return 2.0 * p / denom ** 2 + (1.0 / _SQRT_PI) / denom


def _F_raw(x: torch.Tensor) -> torch.Tensor:
    """Eq. 44 evaluated via the Faddeeva function (no custom gradient)."""
    x = x.to(FDTYPE).clamp_min(0.0)
    s = torch.sqrt(x).to(CDTYPE)
    w = faddeeva(_EXP_J3PI4 * s)
    return 1j * _SQRT_PI * _EXP_MJPI4 * s * w


class _TransitionFunction(torch.autograd.Function):
    """``F(x)`` with the analytic backward pass ``F' = F(1+2jx)/(2x) - j``."""

    #: below this argument the series of Eq. 49 is used for value and slope
    SMALL = 1e-8

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Evaluate F(x), clamping the argument at zero (Eq. 44)."""
        x = x.to(FDTYPE)
        xc = x.clamp_min(0.0)
        F = _F_raw(xc)
        # keep the *unclamped* argument: the clamp is part of the function, so
        # the derivative must vanish on the clamped side
        ctx.save_for_backward(x, xc, F)
        return F

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        """Analytic derivative of Eq. 8, with the clamped side masked to zero."""
        x, xc, F = ctx.saved_tensors
        small = xc < _TransitionFunction.SMALL
        xs = torch.where(small, torch.full_like(xc, _TransitionFunction.SMALL), xc)
        # analytic derivative of Eq. 8
        dF = F * (1.0 + 2j * xs.to(CDTYPE)) / (2.0 * xs.to(CDTYPE)) - 1j
        # x -> 0+: F ~ sqrt(pi) e^{j pi/4} sqrt(x)  =>  F' ~ 0.5 sqrt(pi) e^{j pi/4} / sqrt(x)
        dF_small = 0.5 * _SQRT_PI * _EXP_JPI4 / torch.sqrt(xs).to(CDTYPE)
        dF = torch.where(small, dF_small, dF)
        # F(x) = F(max(x, 0)), so the gradient is identically zero for x <= 0.
        # Without this mask every candidate path whose reflection point lies
        # outside its facet (forward weight exactly 0) would still inject the
        # large small-argument slope into the backward pass, which corrupts
        # the gradient of the whole scene.
        dF = torch.where(x > 0, dF, torch.zeros_like(dF))
        # real-valued input: grad = Re(conj(dF) * grad_out) under torch's
        # convention for C->R composition of complex autograd
        g = (grad_out.conj() * dF).conj()
        return g.real.to(x.dtype)


def transition_F(x: torch.Tensor) -> torch.Tensor:
    """RFDT path-validity weight ``F(x)`` (Eq. 8), differentiable in ``x``."""
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=FDTYPE)
    return _TransitionFunction.apply(x)


# ---------------------------------------------------------------------------
# RFDT edge weight (Eq. 11) and the two competing baselines (Eq. 3, Eq. 4)
# ---------------------------------------------------------------------------
def edge_argument(
    d_edge: torch.Tensor,
    dist_param: torch.Tensor,
    k: float,
    sin_beta0: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Transition argument ``x = k * L * a`` of Eq. 11.

    ``a = 2 sin^2(dbeta/2)`` is the UTD angular-distance factor, evaluated
    with the angular offset of the reflection point from the edge as seen
    through the distance parameter ``L``:

        dbeta = d_edge * sin(beta0) / L ,      L = s' s / (s' + s)

    ``d_edge`` is the in-plane distance from the reflection point to the
    nearest triangle edge and ``beta0`` the angle between the incident ray and
    that edge.

    The returned argument carries the *sign* of ``d_edge``.  ``a`` itself is an
    even function of the angular offset, so without this the lit and shadow
    sides would be indistinguishable; ``transition_F`` clamps negatives to
    zero, so a reflection point outside the triangle contributes ``F(0) = 0``.
    The specular term therefore switches off continuously as the reflection
    point leaves the face, and the diffracted field of Eq. 6 takes over.
    """
    L = dist_param
    dbeta = d_edge * sin_beta0 / L.clamp_min(1e-12)
    # a(dbeta) is monotone only on [-pi, pi]; clamping keeps the weight
    # saturated at a = 2 deep inside a face instead of oscillating
    dbeta = dbeta.clamp(-torch.pi, torch.pi)
    a = 2.0 * torch.sin(0.5 * dbeta) ** 2
    return torch.sign(d_edge) * k * L * a


def weight_rfdt(x: torch.Tensor) -> torch.Tensor:
    """Eq. 11: per-interaction path-validity weight (product taken by caller)."""
    return transition_F(x)


def weight_heaviside(d_edge: torch.Tensor) -> torch.Tensor:
    """Eq. 3: conventional binary in-triangle test (discontinuous)."""
    return (d_edge > 0).to(FDTYPE).to(CDTYPE)


def weight_sigmoid(d_edge: torch.Tensor, k_soft: float) -> torch.Tensor:
    """Eq. 4: "soften triangles" baseline, ``sigma(k * d)`` (continuous, biased)."""
    return torch.sigmoid(k_soft * d_edge).to(CDTYPE)


# ---------------------------------------------------------------------------
# UTD wedge diffraction coefficient (Eq. 6, 7)
# ---------------------------------------------------------------------------
def _cot_times_F(beta: torch.Tensor, n: torch.Tensor, kL: torch.Tensor,
                 sign: int) -> torch.Tensor:
    """One regularised ``cot(.) * F(k L a^{+-})`` term of Eq. 7.

    Classical UTD and RFDT weighting interact here, so the term needs care.

    In classical UTD the geometric-optics field is switched by a *Heaviside*,
    and this cotangent term is deliberately singular: as the angular distance
    ``eps`` to a shadow or reflection boundary goes to zero the product tends
    to ``-(1/2) sgn(eps)`` times the corresponding GO field, which is exactly
    what cancels the Heaviside step and leaves a continuous total.

    RFDT replaces that Heaviside with the smooth weight ``F`` (Eq. 11, 14), so
    the GO term no longer steps.  Leaving the compensating step in place would
    then *introduce* a discontinuity of order the GO field rather than remove
    one, which is double counting.  We therefore damp the term by ``|F(x)|``,
    which is 1 outside the transition region (recovering classical UTD in both
    the deep lit and deep shadow zones) and goes to 0 at the boundary itself,
    so the coefficient passes through zero continuously instead of flipping
    sign.  The Kouyoumjian-Pathak singular limit then evaluates to zero, and
    the guarded branch below returns exactly that.
    """
    two_n_pi = 2.0 * torch.pi * n
    # N^{+} / N^{-}: nearest integer to (beta +- pi) / (2 n pi)
    N = torch.round((beta + sign * torch.pi) / two_n_pi)
    # a^{+-}: angular distance factor, zero exactly on the boundary
    a = 2.0 * torch.cos(0.5 * (two_n_pi * N - beta)) ** 2
    x = kL * a
    F = transition_F(x)

    cot_arg = (torch.pi + sign * beta) / (2.0 * n)
    # guard the tangent away from a zero before dividing
    tan_arg = torch.tan(cot_arg)
    safe = tan_arg.abs() > 1e-7
    cot = torch.where(safe, 1.0 / torch.where(safe, tan_arg, torch.ones_like(tan_arg)),
                      torch.zeros_like(tan_arg))
    return cot.to(CDTYPE) * F * F.abs().to(CDTYPE)


def diffraction_coefficient(
    phi_i: torch.Tensor,
    phi_d: torch.Tensor,
    wedge_n: torch.Tensor,
    k: float,
    L: torch.Tensor,
    sin_beta0: torch.Tensor | float = 1.0,
    refl_0: torch.Tensor | None = None,
    refl_n: torch.Tensor | None = None,
) -> torch.Tensor:
    """UTD wedge diffraction coefficient ``D`` of Eq. 7.

    Parameters
    ----------
    phi_i, phi_d : incidence and diffraction angles measured from face 0,
                   in the plane perpendicular to the edge.
    wedge_n      : ``n = (2 pi - alpha) / pi`` with ``alpha`` the interior
                   wedge angle; ``n = 2`` for a half plane / free edge.
    L            : distance parameter ``s' s / (s' + s) * sin^2(beta0)``.
    refl_0, refl_n : optional Fresnel coefficients of the two wedge faces.
                   When given, the reflection-boundary terms are weighted by
                   them (finite-conductivity UTD), which makes diffraction
                   material dependent.  Defaults to a perfect conductor.

    Returns the complex coefficient; the diffracted field then follows Eq. 6,
    ``E_d = E_i(p_d) * D * A(L) * exp(-jkL)``.
    """
    n = wedge_n.to(FDTYPE)
    kL = (k * L).to(FDTYPE)
    sb = torch.as_tensor(sin_beta0, dtype=FDTYPE)

    beta_minus = phi_d - phi_i        # incidence-shadow-boundary terms
    beta_plus = phi_d + phi_i         # reflection-boundary terms

    t1 = _cot_times_F(beta_minus, n, kL, +1)
    t2 = _cot_times_F(beta_minus, n, kL, -1)
    t3 = _cot_times_F(beta_plus, n, kL, +1)
    t4 = _cot_times_F(beta_plus, n, kL, -1)

    r0 = torch.ones_like(t3) if refl_0 is None else refl_0.to(CDTYPE)
    rn = torch.ones_like(t4) if refl_n is None else refl_n.to(CDTYPE)

    pref = -_EXP_MJPI4 / (2.0 * n.to(CDTYPE) * torch.sqrt(
        torch.as_tensor(2.0 * torch.pi * k, dtype=FDTYPE)).to(CDTYPE) * sb.to(CDTYPE))
    return pref * (t1 + t2 + r0 * t3 + rn * t4)


def spreading_factor(s_prime: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Amplitude spreading ``A(L) = 1/sqrt(L)`` of Eq. 6 for a spherical wave.

    For edge diffraction from a point source the astigmatic factor is
    ``sqrt(s' / (s (s' + s)))``, which reduces to the ``1/sqrt(L)`` quoted in
    Eq. 6 up to the incident-field normalisation.
    """
    return torch.sqrt(s_prime / (s * (s_prime + s)).clamp_min(1e-30))


def distance_parameter(s_prime: torch.Tensor, s: torch.Tensor,
                       sin_beta0: torch.Tensor | float = 1.0) -> torch.Tensor:
    """``L = s' s / (s' + s) * sin^2(beta0)``, the UTD distance parameter."""
    sb = torch.as_tensor(sin_beta0, dtype=FDTYPE)
    return s_prime * s / (s_prime + s).clamp_min(1e-12) * sb ** 2