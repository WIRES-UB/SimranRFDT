"""RFDT: a physically accurate differentiable RF simulator for indoor robotics.

A from-scratch implementation of the forward model and differentiation scheme
of "Physically Accurate Differentiable Inverse Rendering for Radio Frequency
Digital Twin" (RFDT, MobiCom '26), applied to a mobile robot moving through a
furnished indoor room.

Module map
----------
``materials``   ITU-R P.2040 material database, Fresnel coefficients,
                penetration loss (Eq. 56-59).
``transition``  UTD transition function ``F(x)`` with a hand-written analytic
                backward pass, and the wedge diffraction coefficient
                (Eq. 7, 8, 11, 44).
``geometry``    Differentiable mesh operations for the reparameterised method
                of images (Eq. 50-55), plus wedge extraction.
``antennas``    Transmitter and receiver models, radiation patterns, arrays.
``tracer``      The ray tracer: candidate search, specular paths, diffraction,
                secondary visibility, Doppler (Eq. 1, 9, 10, 11).
``signal``      FMCW radar transforms and the Dirichlet-kernel surrogate with
                annealing (Eq. 18-22).
``scenes``      Indoor rooms, obstacles and robot trajectories.
``metrics``     Channel metrics and the SSIM / PSNR / gradient-agreement
                measures used for evaluation.
``optimize``    The digital-twin optimisation loop of Sec. 5.2 (Eq. 23, 24).

Quick start
-----------
>>> from rfdt import RFDTracer, TracerConfig, scenes, antennas
>>> mesh = scenes.furnished_room()
>>> tx = antennas.wifi_ap((3.0, 2.5, 2.7), frequency=5e9)
>>> rx = antennas.robot_client((1.0, 1.0, 0.9))
>>> paths = RFDTracer(mesh, TracerConfig(max_order=2)).trace(tx, rx)
>>> float(paths.power_dbm(tx.power_dbm))
"""

from . import (antennas, geometry, materials, metrics, scenes, signal,
               transition)
from .antennas import (Antenna, Array, Receiver, Transmitter, mmwave_radar,
                       robot_client, wifi_ap)
from .geometry import Mesh
from .materials import MATERIALS, Material, MaterialParams, get_material
from .signal import FMCWConfig
from .tracer import Paths, RFDTracer, TracerConfig

__all__ = [
    "RFDTracer", "TracerConfig", "Paths",
    "Mesh", "Material", "MaterialParams", "MATERIALS", "get_material",
    "Transmitter", "Receiver", "Antenna", "Array",
    "wifi_ap", "robot_client", "mmwave_radar", "FMCWConfig",
    "materials", "transition", "geometry", "antennas", "signal", "scenes",
    "metrics",
]

__version__ = "1.0.0"