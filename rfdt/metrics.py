"""Channel metrics and evaluation measures.

Two groups:

*Propagation metrics* summarise a multipath channel the way a link budget or a
channel sounder would: received power, RMS delay spread, Rice K-factor,
coherence bandwidth, Doppler spread.  These are what change when a wall's
material changes, so they are the headline outputs of the material sweep.

*Evaluation metrics* (SSIM, PSNR, RMSE, median error) mirror Sec. 6: the paper
compares simulated field maps against finite-difference / FDTD ground truth
with SSIM and PSNR, and compares RSS predictions against measurements with
median error and RMSE.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .materials import C0

FDTYPE = torch.float64


# ---------------------------------------------------------------------------
# propagation metrics
# ---------------------------------------------------------------------------
def received_power_dbm(field: torch.Tensor, tx_power_dbm: float) -> torch.Tensor:
    """Coherently combined received power from the complex field ratio."""
    return tx_power_dbm + 10.0 * torch.log10((field.abs() ** 2).clamp_min(1e-30))


def path_loss_db(field: torch.Tensor) -> torch.Tensor:
    """Total path loss ``-20 log10 |E|`` for the field ratio ``E``."""
    return -20.0 * torch.log10(field.abs().clamp_min(1e-15))


def rms_delay_spread(delays: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Power-weighted RMS delay spread [s].

    ``sqrt(E[tau^2] - E[tau]^2)`` with weights ``|alpha_i|^2``.  It grows when
    a material reflects strongly (long-lived multipath) and collapses towards
    the LoS-only value for absorptive materials.
    """
    p = amps.abs() ** 2
    tot = p.sum(-1).clamp_min(1e-30)
    mean = (p * delays).sum(-1) / tot
    second = (p * delays ** 2).sum(-1) / tot
    return torch.sqrt((second - mean ** 2).clamp_min(0.0))


def mean_excess_delay(delays: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Power-weighted mean delay [s], relative to the earliest arriving path."""
    p = amps.abs() ** 2
    tot = p.sum(-1).clamp_min(1e-30)
    return (p * delays).sum(-1) / tot - delays.min(dim=-1).values


def rice_k_factor_db(amps: torch.Tensor, dominant: int = 0) -> torch.Tensor:
    """Rice K-factor in dB: dominant-path power over the scattered remainder.

    ``dominant`` selects which path index is the specular/LoS component
    (0 for the LoS path, which the tracer always emits first).
    """
    p = amps.abs() ** 2
    los = p[..., dominant]
    rest = p.sum(-1) - los
    return 10.0 * torch.log10((los / rest.clamp_min(1e-30)).clamp_min(1e-30))


def coherence_bandwidth(delays: torch.Tensor, amps: torch.Tensor,
                        correlation: float = 0.5) -> torch.Tensor:
    """Approximate coherence bandwidth [Hz] from the RMS delay spread.

    Uses the standard rules of thumb ``B_c ~ 1/(5 sigma_tau)`` at 0.5
    correlation and ``1/(50 sigma_tau)`` at 0.9.

    This is an *estimate*, not a measurement, and its accuracy depends on the
    channel.  Experiment 5 measures coherence bandwidth directly from the
    frequency correlation of H(f) and compares: with metal walls, where
    multipath is rich, the rule of thumb lands within a few per cent, but with
    concrete or foam walls it understates the true value by roughly a factor
    of four, because the rule assumes a dense scattering environment that a
    weakly reflecting room does not provide.  Prefer the direct measurement
    when the answer matters.
    """
    sigma = rms_delay_spread(delays, amps).clamp_min(1e-15)
    factor = 5.0 if correlation <= 0.5 else 50.0
    return 1.0 / (factor * sigma)


def doppler_spread(doppler: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Power-weighted RMS Doppler spread [Hz] (App. E.3)."""
    p = amps.abs() ** 2
    tot = p.sum(-1).clamp_min(1e-30)
    mean = (p * doppler).sum(-1) / tot
    second = (p * doppler ** 2).sum(-1) / tot
    return torch.sqrt((second - mean ** 2).clamp_min(0.0))


def angular_spread(directions: torch.Tensor, amps: torch.Tensor) -> torch.Tensor:
    """Circular RMS angular spread [rad] of the arrival azimuths.

    Computed from the power-weighted resultant length, which avoids the
    wrap-around bias of a naive standard deviation.
    """
    az = torch.atan2(directions[..., 1], directions[..., 0])
    p = amps.abs() ** 2
    tot = p.sum(-1).clamp_min(1e-30)
    r = ((p * torch.exp(1j * az.to(torch.complex128))).sum(-1) / tot).abs()
    return torch.sqrt((-2.0 * torch.log(r.clamp(1e-12, 1.0))).clamp_min(0.0))


def channel_summary(paths, tx_power_dbm: float, noise_floor_dbm: float,
                    doppler: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Bundle the propagation metrics for a traced :class:`Paths` object."""
    field = paths.field()
    out = {
        "rss_dbm": received_power_dbm(field, tx_power_dbm),
        "path_loss_db": path_loss_db(field),
        "snr_db": received_power_dbm(field, tx_power_dbm) - noise_floor_dbm,
        "delay_spread_ns": rms_delay_spread(paths.delay, paths.gain) * 1e9,
        "mean_excess_delay_ns": mean_excess_delay(paths.delay, paths.gain) * 1e9,
        "k_factor_db": rice_k_factor_db(paths.gain),
        "coherence_bw_rule_of_thumb_mhz":
            coherence_bandwidth(paths.delay, paths.gain) / 1e6,
        "angular_spread_deg": angular_spread(paths.arr_dir, paths.gain) * 180.0 / np.pi,
        "n_paths": torch.tensor(float(paths.n_paths())),
    }
    if doppler is not None:
        out["doppler_spread_hz"] = doppler_spread(doppler, paths.gain)
    return out


# ---------------------------------------------------------------------------
# evaluation metrics (Sec. 6)
# ---------------------------------------------------------------------------
def rmse(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Root mean square error between two tensors."""
    return torch.sqrt(((a - b) ** 2).mean())


def median_error(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Median absolute error, the robust metric used for RSS in Sec. 6.2."""
    return (a - b).abs().median()


def error_std(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Standard deviation of the signed error (the "SD" column of Sec. 6.2)."""
    return (a - b).std()


def psnr(a: torch.Tensor, b: torch.Tensor, data_range: Optional[float] = None
         ) -> torch.Tensor:
    """Peak signal-to-noise ratio [dB] between two field maps.

    ``data_range`` defaults to the dynamic range of the reference ``b``.
    Sensitive to amplitude accuracy, unlike SSIM (Sec. 6.1).
    """
    if data_range is None:
        data_range = float(b.max() - b.min())
    mse = ((a - b) ** 2).mean().clamp_min(1e-30)
    return 10.0 * torch.log10(torch.tensor(data_range ** 2, dtype=FDTYPE) / mse)


def _gaussian_window(size: int, sigma: float) -> torch.Tensor:
    """1-D normalised Gaussian kernel used by the SSIM filter."""
    x = torch.arange(size, dtype=FDTYPE) - (size - 1) / 2.0
    g = torch.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return g / g.sum()


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: Optional[float] = None,
         win_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural similarity index between two 2-D maps.

    Follows Wang et al. with a Gaussian window; SSIM prioritises structural
    fidelity, which is why a simulator can score high SSIM with poor PSNR when
    the layout is right but the amplitudes are biased (Sec. 6.1).
    """
    a = a.to(FDTYPE)
    b = b.to(FDTYPE)
    if data_range is None:
        data_range = float(b.max() - b.min()) or 1.0
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    g = _gaussian_window(win_size, sigma)
    kernel = torch.outer(g, g).reshape(1, 1, win_size, win_size)
    A = a.reshape(1, 1, *a.shape)
    B = b.reshape(1, 1, *b.shape)
    pad = win_size // 2

    def filt(x):
        """Gaussian-weighted local mean, used for the SSIM statistics."""
        return torch.nn.functional.conv2d(
            torch.nn.functional.pad(x, (pad,) * 4, mode="reflect"), kernel)

    mu_a, mu_b = filt(A), filt(B)
    saa = filt(A * A) - mu_a ** 2
    sbb = filt(B * B) - mu_b ** 2
    sab = filt(A * B) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2)
    return (num / den).mean()


def gradient_agreement(analytic: torch.Tensor, reference: torch.Tensor
                       ) -> Dict[str, float]:
    """Compare an analytic gradient against a finite-difference reference.

    Reports the relative L2 error and the cosine similarity, the two numbers
    that decide whether a differentiable simulator is usable for optimisation
    (Sec. 6.1).
    """
    a = analytic.reshape(-1).to(FDTYPE)
    r = reference.reshape(-1).to(FDTYPE)
    rel = float((a - r).norm() / r.norm().clamp_min(1e-30))
    cos = float((a * r).sum() / (a.norm().clamp_min(1e-30) * r.norm().clamp_min(1e-30)))
    return {"rel_l2_error": rel, "cosine_similarity": cos,
            "max_abs_error": float((a - r).abs().max())}


def continuity_jump(values: torch.Tensor) -> Dict[str, float]:
    """Largest single-step change in a swept curve, normalised by its range.

    A discontinuous visibility test (Eq. 3) produces a jump of order 1; a
    physically consistent transition (Eq. 11) produces one of order 1/N.
    """
    v = values.reshape(-1).to(FDTYPE)
    d = (v[1:] - v[:-1]).abs()
    rng = float(v.max() - v.min())
    return {"max_jump": float(d.max()),
            "max_jump_normalised": float(d.max() / (rng if rng > 0 else 1.0)),
            "mean_step": float(d.mean())}