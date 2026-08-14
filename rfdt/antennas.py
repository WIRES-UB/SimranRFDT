"""Transmitter and receiver models.

RF ray tracing treats antennas as near-point sources with a directional
amplitude pattern (Sec. 3.1); everything here is therefore a differentiable
function of position, orientation and the radiation pattern, so that
``dE/d p_tx`` (Fig. 9, "Tx position") is available by autograd.

Provided:
  * radiation patterns: isotropic, half-wave dipole, cosine-power patch,
    and a Gaussian main lobe specified by half-power beamwidth;
  * :class:`Transmitter` / :class:`Receiver` with power, gain, polarisation
    and (for the receiver) a thermal-noise model;
  * :class:`Array` for MIMO radars.  Following App. C.4 the ray tracing is run
    once from the array centroid and the full MIMO response is synthesised
    from the per-element phase offsets, which is what makes a 12x16 or 20x20
    virtual array affordable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch

from .geometry import as_t, normalize
from .materials import C0

FDTYPE = torch.float64
CDTYPE = torch.complex128
K_BOLTZMANN = 1.380649e-23
T0 = 290.0          # reference noise temperature [K]


# ---------------------------------------------------------------------------
# radiation patterns:  return *field* amplitude gain sqrt(G(theta, phi))
# ---------------------------------------------------------------------------
def _cos_boresight(direction: torch.Tensor, boresight: torch.Tensor) -> torch.Tensor:
    """Cosine of the angle between a direction and the antenna boresight."""
    return (normalize(direction) * normalize(boresight)).sum(-1).clamp(-1.0, 1.0)


def pattern_isotropic(direction, boresight) -> torch.Tensor:
    """Isotropic radiator: unit field gain in every direction."""
    return torch.ones(direction.shape[:-1], dtype=FDTYPE)


def pattern_dipole(direction, boresight) -> torch.Tensor:
    """Half-wave dipole, peak gain 1.643 (2.15 dBi), null along the axis."""
    c = _cos_boresight(direction, boresight)          # cos of angle to the dipole axis
    s = torch.sqrt((1.0 - c ** 2).clamp_min(1e-16))
    f = torch.cos(0.5 * torch.pi * c) / s.clamp_min(1e-8)
    return torch.sqrt(torch.as_tensor(1.643, dtype=FDTYPE)) * f


def pattern_patch(direction, boresight, q: float = 2.0, peak_gain: float = 6.0):
    """Cosine-power patch: ``G = G0 cos^q(theta)`` over the forward hemisphere."""
    c = _cos_boresight(direction, boresight)
    front = (c > 0).to(FDTYPE)
    return torch.sqrt(torch.as_tensor(peak_gain, dtype=FDTYPE)) * front * c.clamp_min(0.0) ** (q / 2.0)


def pattern_gaussian(direction, boresight, hpbw_deg: float = 60.0,
                     peak_gain_dbi: float = 10.0, sidelobe_db: float = -25.0):
    """Gaussian main lobe with a constant sidelobe floor.

    A smooth, everywhere-differentiable stand-in for a measured horn / array
    pattern.  ``hpbw_deg`` is the two-sided half-power beamwidth.
    """
    c = _cos_boresight(direction, boresight)
    theta = torch.acos(c)
    hp = torch.as_tensor(np.deg2rad(hpbw_deg) / 2.0, dtype=FDTYPE)
    g_db = -3.0 * (theta / hp) ** 2
    g_db = torch.clamp(g_db, min=sidelobe_db)
    g_lin = 10.0 ** ((g_db + peak_gain_dbi) / 10.0)
    return torch.sqrt(g_lin)


PATTERNS = {
    "isotropic": pattern_isotropic,
    "dipole": pattern_dipole,
    "patch": pattern_patch,
    "gaussian": pattern_gaussian,
}


# ---------------------------------------------------------------------------
# arrays (App. C.4)
# ---------------------------------------------------------------------------
@dataclass
class Array:
    """Antenna array described by element offsets in the local frame.

    ``offsets`` is ``(M, 3)`` in metres relative to the array centroid.  The
    per-element response for a path arriving from ``direction`` is
    ``exp(-j k d . offset)``: the ray tracing runs once from the centroid and
    the array manifold is applied afterwards.
    """

    offsets: torch.Tensor
    label: str = "array"

    @property
    def n_elem(self) -> int:
        """Number of elements in the array."""
        return int(self.offsets.shape[0])

    @staticmethod
    def single() -> "Array":
        """A single-element array, i.e. no array processing."""
        return Array(torch.zeros(1, 3, dtype=FDTYPE), "single")

    @staticmethod
    def ula(n: int, spacing: float, axis: str = "y", label: str = "ULA") -> "Array":
        """Uniform linear array of ``n`` elements spaced ``spacing`` along ``axis``."""
        i = torch.arange(n, dtype=FDTYPE) - (n - 1) / 2.0
        off = torch.zeros(n, 3, dtype=FDTYPE)
        off[:, {"x": 0, "y": 1, "z": 2}[axis]] = i * spacing
        return Array(off, f"{label}{n}")

    @staticmethod
    def upa(nx: int, nz: int, spacing: float, label: str = "UPA") -> "Array":
        """Uniform planar array of ``nx`` by ``nz`` elements in the x-z plane."""
        ix = torch.arange(nx, dtype=FDTYPE) - (nx - 1) / 2.0
        iz = torch.arange(nz, dtype=FDTYPE) - (nz - 1) / 2.0
        gx, gz = torch.meshgrid(ix, iz, indexing="ij")
        off = torch.stack([gx.reshape(-1) * spacing,
                           torch.zeros(nx * nz, dtype=FDTYPE),
                           gz.reshape(-1) * spacing], dim=-1)
        return Array(off, f"{label}{nx}x{nz}")

    def steering(self, direction: torch.Tensor, k: float) -> torch.Tensor:
        """``(..., M)`` element phases for unit propagation direction(s)."""
        d = normalize(direction)
        phase = -k * (d.unsqueeze(-2) * self.offsets).sum(-1)
        return torch.exp(1j * phase.to(CDTYPE))


# ---------------------------------------------------------------------------
# transmitter / receiver
# ---------------------------------------------------------------------------
@dataclass
class Antenna:
    """A directional antenna: pattern + boresight + polarisation."""

    pattern: str = "isotropic"
    boresight: Tuple[float, float, float] = (0.0, 0.0, -1.0)
    polarisation: str = "perp"            # "perp" | "par" | "unpolarised"
    kwargs: dict = field(default_factory=dict)

    def field_gain(self, direction: torch.Tensor) -> torch.Tensor:
        """Field-amplitude gain ``sqrt(G)`` towards ``direction`` (``(...,3)``)."""
        bs = as_t(self.boresight).expand_as(direction)
        return PATTERNS[self.pattern](direction, bs, **self.kwargs)

    def gain_dbi(self, direction: torch.Tensor) -> torch.Tensor:
        """Power gain towards ``direction`` in dBi."""
        return 20.0 * torch.log10(self.field_gain(direction).clamp_min(1e-12))


@dataclass
class Transmitter:
    """RF transmitter (AP, radar Tx, or robot-mounted source)."""

    position: torch.Tensor
    frequency: float                     # carrier [Hz]
    power_dbm: float = 20.0
    antenna: Antenna = field(default_factory=Antenna)
    array: Array = field(default_factory=Array.single)
    name: str = "tx"

    def __post_init__(self):
        """Coerce the position to a float64 tensor."""
        self.position = as_t(self.position)

    @property
    def wavelength(self) -> float:
        """Carrier wavelength [m]."""
        return C0 / self.frequency

    @property
    def k(self) -> float:
        """Wavenumber ``2 pi / lambda`` [rad/m]."""
        return 2.0 * float(np.pi) / self.wavelength

    @property
    def power_w(self) -> float:
        """Transmit power in watts."""
        return 10.0 ** ((self.power_dbm - 30.0) / 10.0)

    def eirp_dbm(self, direction: torch.Tensor) -> torch.Tensor:
        """Effective isotropic radiated power towards ``direction`` [dBm]."""
        return self.power_dbm + self.antenna.gain_dbi(direction)


@dataclass
class Receiver:
    """RF receiver (robot-mounted client, radar Rx, or a coverage sample point)."""

    position: torch.Tensor
    antenna: Antenna = field(default_factory=Antenna)
    array: Array = field(default_factory=Array.single)
    noise_figure_db: float = 6.0
    bandwidth_hz: float = 20e6
    name: str = "rx"

    def __post_init__(self):
        """Coerce the position to a float64 tensor."""
        self.position = as_t(self.position)

    @property
    def noise_floor_dbm(self) -> float:
        """``kTB`` plus the noise figure."""
        n = K_BOLTZMANN * T0 * self.bandwidth_hz
        return 10.0 * float(np.log10(n)) + 30.0 + self.noise_figure_db

    def snr_db(self, rss_dbm) -> torch.Tensor:
        """Signal-to-noise ratio [dB] for a given received power in dBm."""
        return as_t(rss_dbm) - self.noise_floor_dbm


# ---------------------------------------------------------------------------
# convenience factories used by the experiments
# ---------------------------------------------------------------------------
def wifi_ap(position, frequency=5.0e9, power_dbm=20.0) -> Transmitter:
    """Ceiling-mounted access point: downward-tilted patch, 20 dBm EIRP-ish."""
    return Transmitter(position, frequency, power_dbm,
                       Antenna("patch", (0.0, 0.0, -1.0), kwargs={"q": 2.0, "peak_gain": 4.0}),
                       Array.single(), "wifi_ap")


def robot_client(position, boresight=(0.0, 0.0, 1.0), bandwidth=20e6) -> Receiver:
    """Client radio on the robot: upward-looking dipole-like antenna."""
    return Receiver(position, Antenna("dipole", boresight), Array.single(),
                    noise_figure_db=6.0, bandwidth_hz=bandwidth, name="robot_rx")


def mmwave_radar(position, boresight=(1.0, 0.0, 0.0), frequency=77e9,
                 n_tx: int = 12, n_rx: int = 16, power_dbm: float = 12.0
                 ) -> Tuple[Transmitter, Receiver]:
    """Monostatic FMCW MIMO radar on the robot.

    Defaults mirror the TI AWR1843 cascade configuration of App. C.1
    (77 GHz, 12 Tx x 16 Rx).  Element spacing is lambda/2 on the Tx side and
    lambda/2 on the Rx side, giving a 192-element virtual array.
    """
    lam = C0 / frequency
    tx = Transmitter(position, frequency, power_dbm,
                     Antenna("gaussian", boresight,
                             kwargs={"hpbw_deg": 70.0, "peak_gain_dbi": 10.0}),
                     Array.ula(n_tx, 2.0 * lam, axis="y"), "radar_tx")
    rx = Receiver(position, Antenna("gaussian", boresight,
                                    kwargs={"hpbw_deg": 70.0, "peak_gain_dbi": 10.0}),
                  Array.ula(n_rx, 0.5 * lam, axis="y"),
                  noise_figure_db=12.0, bandwidth_hz=3.5e9, name="radar_rx")
    return tx, rx