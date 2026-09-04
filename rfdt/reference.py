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
