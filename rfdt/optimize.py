"""Digital-twin optimisation loop (Sec. 5.2 of RFDT).

The inverse problem of Eq. 23,

    theta* = argmin_theta  L( RFDT(theta), y ) ,

where ``y`` is measured RF data and ``theta`` collects the explicit, physically
meaningful scene parameters: material properties, geometry, transmitter pose.
Because every one of them is a differentiable input to the simulator, the
gradient comes straight from backpropagation through the forward model, and
the update is the regularised Adam step of Eq. 24,

    theta_{t+1} <- Adam( theta_t, grad L(S(theta), y) + beta L theta ) ,

with ``L`` a graph Laplacian that keeps optimised meshes smooth.

The loss is the multiscale MSE of Sec. 5.2, which compares the signal at
several spectral resolutions so that coarse structure is matched before fine
structure, and it can be evaluated through the annealed surrogate transform of
Eq. 20 to avoid the local minima described in Sec. 4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from .materials import Material, MaterialParams
from .signal import anneal_schedule

FDTYPE = torch.float64


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------
def multiscale_mse(pred: torch.Tensor, target: torch.Tensor,
                   scales: Sequence[int] = (1, 2, 4, 8)) -> torch.Tensor:
    """Multiscale MSE loss of Sec. 5.2.

    The residual is compared at the native resolution and at successively
    coarser average-pooled versions.  Averaging suppresses the wavelength-scale
    ripple that dominates a raw RF residual, so the coarse scales pull the
    parameters into the right basin while the fine scale sharpens the fit.
    Averaging also "helps to mitigate the sparsity of the RF spectrum".
    """
    pred = pred.reshape(1, 1, -1)
    target = target.reshape(1, 1, -1)
    total = torch.zeros((), dtype=pred.dtype)
    for s in scales:
        if s == 1:
            p, t = pred, target
        else:
            if pred.shape[-1] < s:
                continue
            p = torch.nn.functional.avg_pool1d(pred, s)
            t = torch.nn.functional.avg_pool1d(target, s)
        total = total + ((p - t) ** 2).mean()
    return total / len(scales)


def log_spectral_loss(pred: torch.Tensor, target: torch.Tensor,
                      floor_db: float = -120.0) -> torch.Tensor:
    """MSE in the dB domain, with a floor.

    Power spectra span many decades, so a linear MSE is dominated by the
    strongest bin; comparing logs weights weak multipath components fairly.
    """
    def to_db(x):
        """Convert a complex amplitude to power in dB, clamped at the floor."""
        return 10.0 * torch.log10((x.abs() ** 2).clamp_min(10.0 ** (floor_db / 10.0)))
    return ((to_db(pred) - to_db(target)) ** 2).mean()


# ---------------------------------------------------------------------------
# regularisation
# ---------------------------------------------------------------------------
def uniform_laplacian(faces: np.ndarray, n_vertices: int) -> torch.Tensor:
    """Sparse uniform graph Laplacian ``L`` of a triangle mesh (Eq. 24).

    ``theta^T L theta`` is the Dirichlet energy of the vertex positions, so
    adding ``beta L theta`` to the gradient penalises rough geometry and
    prevents the optimiser from fitting noise with spiky meshes.
    """
    idx: Dict[int, set] = {i: set() for i in range(n_vertices)}
    for f in faces:
        for i in range(3):
            a, b = int(f[i]), int(f[(i + 1) % 3])
            idx[a].add(b)
            idx[b].add(a)
    rows, cols, vals = [], [], []
    for i, nbrs in idx.items():
        if not nbrs:
            continue
        rows.append(i)
        cols.append(i)
        vals.append(float(len(nbrs)))
        for j in nbrs:
            rows.append(i)
            cols.append(j)
            vals.append(-1.0)
    ind = torch.tensor([rows, cols], dtype=torch.long)
    return torch.sparse_coo_tensor(ind, torch.tensor(vals, dtype=FDTYPE),
                                   (n_vertices, n_vertices)).coalesce()


def laplacian_gradient(vertices: torch.Tensor, L: torch.Tensor,
                       beta: float) -> torch.Tensor:
    """The ``beta * L * theta`` term added to the gradient in Eq. 24."""
    return beta * torch.sparse.mm(L, vertices)


# ---------------------------------------------------------------------------
# optimisation driver
# ---------------------------------------------------------------------------
@dataclass
class OptimResult:
    """Outcome of a digital-twin optimisation run."""

    loss_history: List[float] = field(default_factory=list)
    param_history: List[Dict[str, float]] = field(default_factory=list)
    lambda_history: List[float] = field(default_factory=list)
    best_loss: float = float("inf")
    best_params: Dict[str, float] = field(default_factory=dict)
    epochs: int = 0
    seconds: float = 0.0

    def summary(self) -> Dict[str, float]:
        """Compact dictionary of the run for tables and JSON output."""
        return {"epochs": self.epochs, "best_loss": self.best_loss,
                "final_loss": self.loss_history[-1] if self.loss_history else float("nan"),
                "seconds": self.seconds, **self.best_params}


def optimize_digital_twin(
    forward: Callable[[int, float], torch.Tensor],
    target: torch.Tensor,
    parameters: Sequence[torch.Tensor],
    epochs: int = 300,
    lr: float = 0.05,
    loss_fn: Callable = multiscale_mse,
    use_surrogate: bool = True,
    warmup: float = 0.4,
    readout: Optional[Callable[[], Dict[str, float]]] = None,
    vertices: Optional[torch.Tensor] = None,
    laplacian: Optional[torch.Tensor] = None,
    beta: float = 0.0,
    verbose: bool = False,
) -> OptimResult:
    """Run the Eq. 23 / Eq. 24 optimisation loop.

    Parameters
    ----------
    forward     : ``(epoch, lambda) -> prediction``.  Called once per epoch;
                  ``lambda`` is the Eq. 20 annealing weight, which the callback
                  should pass to the signal transform (0 = smooth surrogate,
                  1 = exact FFT).
    target      : the measurement ``y`` of Eq. 23.
    parameters  : leaf tensors with ``requires_grad=True`` to optimise.
    use_surrogate : follow the coarse-to-fine schedule; if False, ``lambda`` is
                  pinned at 1 so the exact transform is used throughout, which
                  is the ablation of Fig. 16(b).
    readout     : optional callback returning current parameter values to log.
    vertices, laplacian, beta : enable the Laplacian regularisation of Eq. 24.

    Returns an :class:`OptimResult` with the loss curve and the best parameters.
    """
    import time

    params = [p for p in parameters if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    res = OptimResult(epochs=epochs)
    t0 = time.time()

    for epoch in range(epochs):
        lam = anneal_schedule(epoch, epochs, warmup) if use_surrogate else 1.0
        opt.zero_grad()
        pred = forward(epoch, lam)
        loss = loss_fn(pred, target)
        loss.backward()

        if vertices is not None and laplacian is not None and beta > 0.0:
            if vertices.grad is None:
                vertices.grad = torch.zeros_like(vertices)
            vertices.grad += laplacian_gradient(vertices.detach(), laplacian, beta)

        opt.step()

        lv = float(loss.detach())
        res.loss_history.append(lv)
        res.lambda_history.append(lam)
        cur = readout() if readout is not None else {}
        res.param_history.append(cur)
        # only accept a new best once annealing has reached the exact
        # transform, since surrogate losses are not comparable to exact ones
        if lam >= 1.0 and lv < res.best_loss:
            res.best_loss = lv
            res.best_params = dict(cur)
        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            extra = " ".join(f"{k}={v:.4g}" for k, v in cur.items())
            print(f"  epoch {epoch:4d}  lambda={lam:.2f}  loss={lv:.6e}  {extra}")

    if not res.best_params and res.param_history:
        res.best_loss = res.loss_history[-1]
        res.best_params = dict(res.param_history[-1])
    res.seconds = time.time() - t0
    return res


def material_parameter_set(names: Sequence[str], f_hz: float,
                           initial: Optional[Dict[str, str]] = None,
                           materials: Optional[Dict[str, Material]] = None
                           ) -> Dict[str, MaterialParams]:
    """Build learnable ``(eps', sigma)`` overrides for a set of materials.

    ``initial`` optionally maps a material name to the name of a *different*
    material to start from, which is how an experiment can begin from a
    deliberately wrong guess and check that the optimiser recovers the truth.
    """
    from .materials import get_material

    out: Dict[str, MaterialParams] = {}
    for name in names:
        src = name
        if initial and name in initial:
            src = initial[name]
        mat = (materials or {}).get(src) or get_material(src)
        out[name] = MaterialParams.from_material(mat, f_hz, requires_grad=True)
    return out