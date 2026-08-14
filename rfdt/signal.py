"""Signal-domain transforms and RFDT's coarse-to-fine surrogate (Sec. 4).

A ray tracer produces a channel impulse response (CIR): a set of paths with
delays ``tau_i``, complex amplitudes ``alpha_i`` and Doppler shifts.  Radar and
communication systems never observe the CIR directly; they observe it after a
signal-domain transform (FMCW de-chirping + FFT, OFDM channel estimation,
angular beamforming).  Those transforms are what make the optimisation
landscape non-convex (Sec. 4, "Root causes of non-convexity"): finite bandwidth
gives every path a sidelobe-rich main lobe, and the 2*pi phase periodicity
creates wavelength-scale local minima.

RFDT's answer is the *coarse-to-fine surrogate* of Eq. 18-20: approximate the
FFT range profile with a sum of Dirichlet-kernel point spread functions

    R(f)^s = sum_i G_i(tau_i, f) * alpha_i ,
    G_i    = | sin(N pi r_i) / (N sin(pi r_i)) |^2 ,   r_i = (tau_i - D(f))/sigma

which is exactly the squared FFT magnitude of a finite-length sinusoid observed
through a rectangular window, then anneal from it to the exact FFT profile

    R(f) = lambda(t) R_FFT(f) + (1 - lambda(t)) R(f)^s .

During the phase-agnostic warm-up the path phases are dropped, producing the
smoother, more convex landscape of Fig. 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

from .materials import C0

FDTYPE = torch.float64
CDTYPE = torch.complex128


# ---------------------------------------------------------------------------
# radar waveform configuration
# ---------------------------------------------------------------------------
@dataclass
class FMCWConfig:
    """FMCW chirp parameters.

    Defaults follow the TI AWR1843 configuration of App. C.1: 77 GHz carrier,
    60 MHz/us slope, 256 ADC samples at 4.4 Msps, 128 chirps per frame.
    """

    f_c: float = 77e9              # carrier frequency [Hz]
    slope: float = 60e12           # chirp slope [Hz/s]  (60 MHz/us)
    n_samples: int = 256           # ADC samples per chirp
    fs: float = 4.4e6              # ADC sampling rate [Hz]
    n_chirps: int = 128            # chirps per frame (Doppler dimension)
    chirp_period: float = 72e-6    # chirp repetition interval [s]
    n_fft: int = 512               # zero-padded range FFT size

    @property
    def bandwidth(self) -> float:
        """Sweep bandwidth actually covered by the sampled part of the chirp [Hz]."""
        return self.slope * self.n_samples / self.fs

    @property
    def range_resolution(self) -> float:
        """Classical range resolution ``c / (2 B)`` [m]."""
        return C0 / (2.0 * self.bandwidth)

    @property
    def max_range(self) -> float:
        """Largest unambiguous range set by the IF sampling rate [m]."""
        return C0 * self.fs / (4.0 * self.slope) * 2.0

    def range_axis(self) -> torch.Tensor:
        """Range corresponding to each FFT bin, i.e. ``D(f)`` in Eq. 19 [m]."""
        f_beat = torch.arange(self.n_fft, dtype=FDTYPE) * self.fs / self.n_fft
        return f_beat * C0 / (2.0 * self.slope)

    def delay_axis(self) -> torch.Tensor:
        """Round-trip delay corresponding to each FFT bin [s]."""
        return 2.0 * self.range_axis() / C0

    def velocity_axis(self) -> torch.Tensor:
        """Radial velocity corresponding to each Doppler bin [m/s]."""
        lam = C0 / self.f_c
        f_d = torch.fft.fftshift(torch.fft.fftfreq(self.n_chirps,
                                                   d=self.chirp_period)).to(FDTYPE)
        return f_d * lam / 2.0


# ---------------------------------------------------------------------------
# exact (unbiased) signal transform
# ---------------------------------------------------------------------------
def fmcw_beat_signal(delays: torch.Tensor, amps: torch.Tensor, cfg: FMCWConfig,
                     doppler: Optional[torch.Tensor] = None,
                     chirp_index: int = 0) -> torch.Tensor:
    """Synthesise the de-chirped IF signal of Eq. 27.

    ``S_IF(t) = sum_i A_i exp(2 pi j (mu tau_i t + f_c tau_i))`` with an extra
    ``exp(2 pi j f_D t_slow)`` term for motion across chirps.

    Parameters
    ----------
    delays, amps : ``(..., P)`` path delays [s] and complex amplitudes.
    doppler      : optional ``(..., P)`` Doppler shifts [Hz].

    Returns the complex baseband samples, shape ``(..., n_samples)``.
    """
    t = torch.arange(cfg.n_samples, dtype=FDTYPE) / cfg.fs           # fast time
    phase = (2.0 * np.pi * (cfg.slope * delays.unsqueeze(-1) * t
                            + cfg.f_c * delays.unsqueeze(-1)))
    sig = amps.unsqueeze(-1).to(CDTYPE) * torch.exp(1j * phase.to(CDTYPE))
    if doppler is not None:
        t_slow = chirp_index * cfg.chirp_period
        sig = sig * torch.exp(2j * np.pi * (doppler.unsqueeze(-1) * t_slow).to(CDTYPE))
    return sig.sum(dim=-2)


def range_profile_fft(delays: torch.Tensor, amps: torch.Tensor, cfg: FMCWConfig,
                      window: str = "hann", doppler: Optional[torch.Tensor] = None
                      ) -> torch.Tensor:
    """Exact FFT range profile ``R_FFT(f)``, the unbiased target of Eq. 20.

    This is the transform a real radar applies, sidelobes and all; it is
    differentiable but its landscape is the "rugged" one of Fig. 4.

    ``window`` defaults to Hann because that is what a practical radar uses.
    Note that the Dirichlet-kernel surrogate of Eq. 19 is the exact squared
    magnitude of a *rectangular*-windowed sinusoid, so a like-for-like
    comparison of surrogate against exact should pass ``window="rect"``;
    otherwise the difference between them mixes the annealing effect with a
    change of window.
    """
    sig = fmcw_beat_signal(delays, amps, cfg, doppler)
    if window == "hann":
        w = torch.hann_window(cfg.n_samples, periodic=False, dtype=FDTYPE)
    elif window == "hamming":
        w = torch.hamming_window(cfg.n_samples, periodic=False, dtype=FDTYPE)
    else:
        w = torch.ones(cfg.n_samples, dtype=FDTYPE)
    return torch.fft.fft(sig * w.to(CDTYPE), n=cfg.n_fft, dim=-1)


def range_doppler_map(delays: torch.Tensor, amps: torch.Tensor,
                      doppler: torch.Tensor, cfg: FMCWConfig) -> torch.Tensor:
    """Range-Doppler map: range FFT per chirp, then an FFT across chirps.

    Used to visualise the moving-robot spectra of Fig. 19(b).
    """
    profiles = []
    for m in range(cfg.n_chirps):
        sig = fmcw_beat_signal(delays, amps, cfg, doppler, chirp_index=m)
        w = torch.hann_window(cfg.n_samples, periodic=False, dtype=FDTYPE)
        profiles.append(torch.fft.fft(sig * w.to(CDTYPE), n=cfg.n_fft, dim=-1))
    rd = torch.stack(profiles, dim=-2)                     # (..., n_chirps, n_fft)
    return torch.fft.fftshift(torch.fft.fft(rd, dim=-2), dim=-2)


# ---------------------------------------------------------------------------
# Dirichlet-kernel surrogate (Eq. 18, 19, 21, 22)
# ---------------------------------------------------------------------------
def dirichlet_kernel(tau: torch.Tensor, tau_axis: torch.Tensor, n: int,
                     sigma: float) -> torch.Tensor:
    """Dirichlet (periodic sinc) point spread function ``G_i`` of Eq. 19.

    ``G(tau, f) = |sin(N pi r) / (N sin(pi r))|^2`` with ``r = (tau - D(f))/sigma``.
    ``N`` is the window length (number of chirp samples) and ``sigma`` converts
    a delay difference into normalised digital frequency, so the first null
    sits exactly at ``1/B``, the classical range resolution.  The ``r -> 0``
    limit is 1, handled explicitly to keep the gradient finite.
    """
    r = (tau.unsqueeze(-1) - tau_axis) / sigma
    num = torch.sin(n * np.pi * r)
    den = n * torch.sin(np.pi * r)
    small = den.abs() < 1e-12
    val = torch.where(small, torch.ones_like(num),
                      num / torch.where(small, torch.ones_like(den), den))
    return val ** 2


def surrogate_range_profile(delays: torch.Tensor, amps: torch.Tensor,
                            cfg: FMCWConfig, phase_agnostic: bool = True,
                            sigma: Optional[float] = None,
                            broadening: float = 1.0) -> torch.Tensor:
    """Smooth surrogate profile ``R(f)^s`` of Eq. 18.

    Two mechanisms make this smoother than the exact transform, and both are
    needed:

    *Phase-agnostic*: with ``phase_agnostic=True`` the path phases ``phi_i``
    are dropped (Sec. 4, "Phase-agnostic initial optimization"), removing the
    wavelength-scale ripples that trap gradient descent early on.

    *Coarse-to-fine*: ``broadening`` divides the effective window length, which
    widens the Dirichlet main lobe by the same factor.  This is what makes the
    model "coarse-to-fine" rather than merely "smoothed".  It matters because
    dropping the phase does nothing for an estimate that starts several range
    cells away from the truth: two narrow peaks that do not overlap produce a
    flat loss and therefore no gradient at all.  A broadened kernel is the
    exact point spread function of a shorter observation window, so it stays
    physically interpretable while giving the loss a basin wide enough to
    reach the initial guess.  Annealing ``broadening`` back to 1 restores the
    full resolution.
    """
    tau_axis = cfg.delay_axis()
    if sigma is None:
        # a delay difference maps to normalised digital frequency via
        # r = slope * dtau / fs, so sigma = fs / slope
        sigma = cfg.fs / cfg.slope
    n_eff = max(4.0, cfg.n_samples / max(broadening, 1.0))
    G = dirichlet_kernel(delays, tau_axis, n_eff, sigma)
    a = amps.abs() if phase_agnostic else amps
    return (G * a.unsqueeze(-1).abs()).sum(dim=-2)


def annealed_range_profile(delays: torch.Tensor, amps: torch.Tensor,
                           cfg: FMCWConfig, lam: float,
                           phase_agnostic: bool = True,
                           max_broadening: float = 12.0) -> torch.Tensor:
    """Eq. 20: ``R = lambda R_FFT + (1 - lambda) R^s``.

    ``lam`` follows a monotone schedule from 0 (fully smoothed surrogate) to 1
    (exact, unbiased FFT), so every schedule converges to the same optimum.
    The surrogate's resolution is annealed on the same schedule, from
    ``max_broadening`` times coarser than the true range resolution back to
    full resolution, which is the "coarse-to-fine" half of the model.
    """
    exact = range_profile_fft(delays, amps, cfg).abs()
    if lam >= 1.0:
        return exact
    broadening = 1.0 + (max_broadening - 1.0) * (1.0 - lam)
    surr = surrogate_range_profile(delays, amps, cfg, phase_agnostic,
                                   broadening=broadening)
    # match the surrogate scale to the FFT scale so the blend is meaningful
    scale = exact.amax(dim=-1, keepdim=True) / surr.amax(dim=-1, keepdim=True).clamp_min(1e-30)
    return lam * exact + (1.0 - lam) * surr * scale


def anneal_schedule(epoch: int, total: int, warmup: float = 0.4) -> float:
    """Monotone ``lambda(t)`` in [0, 1] for Eq. 20.

    Stays at 0 for the first ``warmup`` fraction of training (pure surrogate),
    then rises smoothly to 1 so the final iterations optimise the exact
    transform and the result is unbiased.
    """
    if total <= 1:
        return 1.0
    t = epoch / (total - 1)
    if t <= warmup:
        return 0.0
    u = (t - warmup) / (1.0 - warmup)
    return float(0.5 * (1.0 - np.cos(np.pi * min(u, 1.0))))


# ---------------------------------------------------------------------------
# communication-side transforms
# ---------------------------------------------------------------------------
def ofdm_channel(delays: torch.Tensor, amps: torch.Tensor, freqs: torch.Tensor,
                 doppler: Optional[torch.Tensor] = None,
                 t: float = 0.0) -> torch.Tensor:
    """Frequency-domain channel ``H(f) = sum_i alpha_i exp(-2 pi j f tau_i)``.

    This is what an OFDM receiver estimates per subcarrier; combined with the
    Doppler term it gives the time-varying channel of App. E.3.
    """
    ph = -2.0 * np.pi * freqs * delays.unsqueeze(-1)
    if doppler is not None:
        ph = ph + 2.0 * np.pi * doppler.unsqueeze(-1) * t
    return (amps.unsqueeze(-1).to(CDTYPE) * torch.exp(1j * ph.to(CDTYPE))).sum(dim=-2)


def beamform(amps: torch.Tensor, steering: torch.Tensor) -> torch.Tensor:
    """Apply an array manifold to per-path amplitudes (App. C.4).

    ``steering`` is ``(..., P, M)``; returns the ``(..., M)`` element signals,
    i.e. the MIMO response synthesised from centroid-traced paths.
    """
    return (amps.unsqueeze(-1).to(CDTYPE) * steering).sum(dim=-2)