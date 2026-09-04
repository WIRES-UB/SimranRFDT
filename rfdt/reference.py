"""Exact and numerically exact references, for checking the approximations.

Everything else in this package is an approximation to Maxwell's equations:
rays instead of waves, a first-order edge correction, a damped coefficient, a
smooth validity weight.  The closed-form checks in the test suite validate
those in the regions where they are easy, free space and specular reflection
far from any edge.  They say nothing about the transition region around a
shadow or reflection boundary, which is exactly the region the whole
differentiable formulation depends on, because that is where the weight runs
from one to zero and therefore where the gradient lives.

This module supplies references that are valid *inside* that region.

  * ``half_plane_field`` is the Sommerfeld solution for a perfectly conducting
    half plane.  It is exact, in closed form, and it is one of the very few
    diffraction problems for which that is true.

Nothing here is used by the simulator.  It exists to be disagreed with.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

from .transition import CDTYPE, FDTYPE, modified_fresnel

C0 = 299_792_458.0

def half_plane_field(k: float, rho: torch.Tensor, phi: torch.Tensor,
                     phi_0: float, polarisation: str = "soft") -> torch.Tensor:
    """Exact total field around a perfectly conducting half plane.

    Sommerfeld's 1896 solution, the canonical diffraction problem and one of
    the few with an exact closed form.  The half plane occupies ``phi = 0``,
    extending from the edge at the origin; a plane wave of unit amplitude
    arrives from the direction ``phi_0``.  ``rho`` and ``phi`` are the
    observation point in polar coordinates about the edge.

    The total field is the sum of two identical terms evaluated at the
    incident and the reflected angular distance,

        u = U(phi - phi_0)  -+  U(phi + phi_0),
        U(psi) = e^{j k rho cos psi} F(-sqrt(2 k rho) cos(psi / 2)),

    with the minus sign for the soft (Dirichlet) case, where the field
    vanishes on the plate, and plus for the hard (Neumann) case.  The first
    term carries the incident field and its shadow boundary; the second
    carries the reflected field and its own boundary, and the sign is what
    enforces the boundary condition on the surface.

    Read the structure rather than the algebra: each term is a plane wave
    multiplied by a transition function whose argument measures, in units of
    the square root of the distance in wavelengths, how far the observation
    point is from the corresponding boundary.  Far from it the transition
    function saturates and the term becomes geometrical optics, present or
    absent; close to it the term varies smoothly through one half.  That is
    the same shape RFDT's weight has, which is why this is the right thing to
    check it against.
    """
    rho = torch.as_tensor(rho, dtype=FDTYPE)
    phi = torch.as_tensor(phi, dtype=FDTYPE)
    sign = -1.0 if polarisation == "soft" else 1.0

    def term(psi):
        # The sign here is the whole convention.  With it reversed the
        # transition function saturates the wrong way and the solution places
        # the reflected wave in the region geometrical optics says cannot
        # contain one, which is how it was caught: the boundary condition on
        # the plate is satisfied either way, so only a comparison against ray
        # optics away from the boundaries distinguishes them.
        arg = torch.sqrt((2.0 * k * rho).clamp_min(0.0)) * torch.cos(psi / 2.0)
        return (torch.exp(1j * (k * rho * torch.cos(psi)).to(CDTYPE))
                * modified_fresnel(arg))

    return term(phi - phi_0) + sign * term(phi + phi_0)


def half_plane_geometrical_optics(k: float, rho: torch.Tensor,
                                  phi: torch.Tensor, phi_0: float,
                                  polarisation: str = "soft") -> torch.Tensor:
    """The ray-optics field the exact solution must approach away from edges.

    Incident everywhere outside the shadow, plus a reflected wave in the region
    that sees the illuminated face, and nothing at all in the shadow.  This is
    the discontinuous field that diffraction exists to repair, so it is the
    thing to compare against far from the boundaries and the thing that must
    *not* match near them.
    """
    rho = torch.as_tensor(rho, dtype=FDTYPE)
    phi = torch.as_tensor(phi, dtype=FDTYPE)
    sign = -1.0 if polarisation == "soft" else 1.0
    lit_incident = (phi - phi_0).abs() < np.pi
    lit_reflected = (phi + phi_0) < np.pi
    inc = torch.exp(1j * (k * rho * torch.cos(phi - phi_0)).to(CDTYPE))
    ref = torch.exp(1j * (k * rho * torch.cos(phi + phi_0)).to(CDTYPE))
    out = torch.where(lit_incident, inc, torch.zeros_like(inc))
    return out + torch.where(lit_reflected, sign * ref, torch.zeros_like(ref))


# ---------------------------------------------------------------------------
# numerically exact reference: a conducting strip, by method of moments
# ---------------------------------------------------------------------------
#: Euler-Mascheroni constant, needed for the self-term of the 2D kernel.
_EULER_GAMMA = 0.5772156649015329


def hankel1_0(x: torch.Tensor) -> torch.Tensor:
    """``H_0^{(1)}(x) = J_0(x) + j Y_0(x)``, the outgoing 2D Green's kernel.

    Outgoing for the ``e^{-i omega t}`` convention this module already uses for
    the half plane, where a radiating wave goes as ``e^{+ikr}``.  Both Bessel
    functions come from torch, so nothing new has to be trusted.
    """
    x = torch.as_tensor(x, dtype=FDTYPE)
    return (torch.special.bessel_j0(x).to(CDTYPE)
            + 1j * torch.special.bessel_y0(x).to(CDTYPE))


def strip_scattered_field(k: float, width: float, phi_inc: float,
                          obs_x: torch.Tensor, obs_y: torch.Tensor,
                          n_segments: int = 400,
                          source: Optional[Tuple[float, float]] = None):
    """Field scattered by a perfectly conducting strip, solved numerically.

    Why a strip and not another closed form
    ---------------------------------------
    The half plane above is exact but has no width.  The one quantity this
    whole project's headline claim rests on is the derivative of the field with
    respect to a reflector's *size*, and a half plane cannot supply a reference
    for it.  A finite strip is exactly the "reflector of finite extent" the
    differentiable formulation is about, and while it has no closed form it can
    be solved to whatever accuracy is wanted.

    The method
    ----------
    The strip lies along ``x`` in ``|x| <= width/2`` at ``y = 0``, infinite in
    the third direction, and is illuminated by a transverse-magnetic plane wave
    arriving from ``phi_inc``, so the electric field points along the invariant
    direction.  A perfect conductor forces the total tangential field to vanish
    on it, and the field radiated by the induced surface current is an integral
    of that current against the two-dimensional Green's function.  Setting the
    total to zero on the surface gives one integral equation for the current.

    Discretising the strip into ``n_segments`` pieces, taking the current
    constant on each and enforcing the boundary condition at each midpoint,
    turns it into a dense linear system.  The off-diagonal entries are just the
    kernel evaluated between two midpoints; the diagonal needs care, because
    the kernel has a logarithmic singularity where the observation point sits
    on the source segment.  That singularity is integrable and the integral is
    done analytically from the small-argument expansion
    ``H_0^{(1)}(x) -> 1 + j (2/pi) (ln(x/2) + gamma)``, giving

        Z_mm = (k eta Delta / 4) [1 + j (2/pi) (ln(k Delta / 4) + gamma - 1)],

    which is where the Euler constant enters.  Solving the system gives the
    current, and radiating it to the observation points gives the field.

    This is a reference, so it is deliberately written for clarity over speed:
    dense assembly, dense solve, no acceleration.
    """
    n = int(n_segments)
    dx = width / n
    xs = torch.linspace(-width / 2 + dx / 2, width / 2 - dx / 2, n,
                        dtype=FDTYPE)

    # Incident field on the strip.  A plane wave by default; ``source`` gives a
    # line source instead, which is what makes a *specular point* exist.  A
    # plane wave has no such point, and the gradient with respect to reflector
    # size is only interesting where the reflector's edge passes through one.
    if source is None:
        v = torch.exp(1j * (k * xs * float(np.cos(phi_inc))).to(CDTYPE))
    else:
        sx, sy = float(source[0]), float(source[1])
        rs = torch.sqrt((xs - sx) ** 2 + sy ** 2).clamp_min(1e-12)
        v = hankel1_0(k * rs)

    sep = (xs.reshape(-1, 1) - xs.reshape(1, -1)).abs()
    eye = torch.eye(n, dtype=torch.bool)
    off = hankel1_0((k * sep).clamp_min(1e-30))
    diag = 1.0 + 1j * (2.0 / np.pi) * (
        np.log(k * dx / 4.0) + _EULER_GAMMA - 1.0)
    Z = torch.where(eye, torch.full_like(off, diag), off) * dx

    # the constant (k eta / 4) multiplies both sides, so it cancels; the
    # current is returned in whatever units make the radiated field match
    current = torch.linalg.solve(Z, v)

    ox = torch.as_tensor(obs_x, dtype=FDTYPE).reshape(-1, 1)
    oy = torch.as_tensor(obs_y, dtype=FDTYPE).reshape(-1, 1)
    dxs = ox - xs.reshape(1, -1)
    r = torch.sqrt(dxs ** 2 + oy ** 2).clamp_min(1e-12)
    kern = hankel1_0(k * r)
    # An observation point lying on the strip falls inside one of the source
    # segments, where the kernel is logarithmically singular and its value at a
    # single point means nothing.  The matrix already integrates that case
    # analytically; the radiated field has to use the same integral, or
    # evaluating the field on the surface returns a large spurious number and
    # the boundary condition appears to be violated when it is not.
    on_segment = (dxs.abs() < dx / 2.0) & (oy.abs() < dx / 2.0)
    if bool(on_segment.any()):
        kern = torch.where(on_segment, torch.full_like(kern, diag), kern)
    return -(kern * current.reshape(1, -1) * dx).sum(-1)


def strip_incident_field(k: float, phi_inc: float, obs_x: torch.Tensor,
                         obs_y: torch.Tensor) -> torch.Tensor:
    """The illuminating plane wave, in the same convention as the strip solver."""
    ox = torch.as_tensor(obs_x, dtype=FDTYPE)
    oy = torch.as_tensor(obs_y, dtype=FDTYPE)
    phase = k * (ox * float(np.cos(phi_inc)) + oy * float(np.sin(phi_inc)))
    return torch.exp(1j * phase.to(CDTYPE))
