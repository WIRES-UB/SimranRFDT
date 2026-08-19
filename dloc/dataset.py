"""Loader for the DLoc / WILD measurement files.

The dataset accompanies "Deep Learning based Wireless Localization for Indoor
Navigation" (Ayyalasomayajula et al., MobiCom 2020).  Measurements were
collected by MapFind, a SLAM robot carrying a WiFi client, moving at a constant
15 cm/s with a channel estimate every 50 ms.

Files are MATLAB v7.3, which is HDF5 underneath, so ``h5py`` reads them without
MATLAB.  Two kinds:

``channels_<setup>.mat``
    channels  [n_datapoints x n_frequency x n_ant x n_ap]  complex CSI
    RSSI      [n_datapoints x n_ap]
    labels    [n_datapoints x 2]                            ground-truth XY [m]
    opt.freq  [n_frequency x 1]                             subcarrier freqs
    opt.lambda[n_frequency x 1]
    ap        n_ap cells, each [n_ant x 2]                  antenna coords [m]
    ap_aoa    [n_ap x 1]                                    AoA rotation offset
    d1, d2                                                  sampled space axes

``features_<setup>.mat``
    the precomputed heatmaps DLoc trains on; not needed for a channel
    comparison, so only the channel files are read here.

Two conventions this module handles so callers do not have to:

*HDF5 transposes MATLAB.* A MATLAB ``[a x b x c]`` array reads back as
``(c, b, a)``.  Every accessor here returns arrays in the *documented MATLAB
order*, so the shapes in the table above are what you get.

*MATLAB has no native complex HDF5 type.* Complex arrays come back as a
compound dtype with ``real`` and ``imag`` fields, which is reassembled here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

#: Setup names, grouped by environment, from the dataset documentation.
ENVIRONMENTS = {
    "atkinson": {
        "label": "Atkinson Hall, LOS-based, 500 sq ft",
        "quoted_extent_m": (8.0, 5.0),
        "n_ap": 3,
        "setups": ["July16", "July18", "July22_2_ref"],
    },
    "jacobs": {
        "label": "Jacobs Hall, complex multipath and NLOS, 1500 sq ft",
        "quoted_extent_m": (18.0, 8.0),
        "n_ap": 4,
        "setups": ["jacobs_Jul28", "jacobs_Jul28_2", "jacobs_Aug16_1",
                   "jacobs_Aug16_3", "jacobs_Aug16_4_ref"],
    },
}

#: Robot speed and channel rate, from the paper; used to convert a run of
#: consecutive samples into elapsed time and hence into Doppler.
ROBOT_SPEED_M_S = 0.15
CHANNEL_PERIOD_S = 0.050


def environment_of(setup: str) -> str:
    """Which environment a setup name belongs to."""
    for name, meta in ENVIRONMENTS.items():
        if setup in meta["setups"]:
            return name
    raise KeyError(f"unknown setup {setup!r}; known: "
                   + ", ".join(s for m in ENVIRONMENTS.values() for s in m["setups"]))


@dataclass
class DLocMeasurement:
    """One DLoc setup: channels, RSSI, ground truth and AP geometry.

    Arrays follow the documented MATLAB dimension order, not the transposed
    order HDF5 returns.
    """

    setup: str
    channels: np.ndarray          # (n_points, n_freq, n_ant, n_ap) complex
    rssi: np.ndarray              # (n_points, n_ap)
    labels: np.ndarray            # (n_points, 2) ground-truth XY [m]
    freqs: np.ndarray             # (n_freq,) [Hz]
    ap_coords: List[np.ndarray]   # n_ap arrays of (n_ant, 2) [m]
    ap_aoa: np.ndarray            # (n_ap,) rotation offsets
    source_path: str

    @property
    def n_points(self) -> int:
        """Number of measured robot positions."""
        return int(self.labels.shape[0])

    @property
    def n_ap(self) -> int:
        """Number of access points."""
        return len(self.ap_coords)

    @property
    def n_ant(self) -> int:
        """Antennas per access point."""
        return int(self.ap_coords[0].shape[0])

    @property
    def bandwidth_hz(self) -> float:
        """Span of the subcarrier frequencies."""
        return float(self.freqs.max() - self.freqs.min())

    @property
    def centre_frequency_hz(self) -> float:
        """Mid-band frequency, used as the simulator's carrier."""
        return float(0.5 * (self.freqs.max() + self.freqs.min()))

    def ap_centroids(self) -> np.ndarray:
        """(n_ap, 2) centroid of each AP's antenna array.

        The simulator traces from an array centroid and applies the element
        phases afterwards (the App. C.4 approach), so this is the position a
        transmitter is placed at.
        """
        return np.stack([c.mean(axis=0) for c in self.ap_coords])

    def route_extent(self) -> Dict[str, float]:
        """Bounding box of the ground-truth positions, in metres.

        This is the area the robot actually covered, which is smaller than the
        full SLAM map and is what the quoted floor area refers to.
        """
        lo, hi = self.labels.min(axis=0), self.labels.max(axis=0)
        return {"x_min": float(lo[0]), "x_max": float(hi[0]),
                "y_min": float(lo[1]), "y_max": float(hi[1]),
                "width_m": float(hi[0] - lo[0]), "height_m": float(hi[1] - lo[1])}

    def summary(self) -> Dict[str, object]:
        """Compact description, for logging what a run was actually given."""
        return {
            "setup": self.setup,
            "environment": environment_of(self.setup),
            "n_points": self.n_points,
            "n_ap": self.n_ap,
            "n_ant_per_ap": self.n_ant,
            "n_freq": int(self.freqs.size),
            "centre_frequency_ghz": self.centre_frequency_hz / 1e9,
            "bandwidth_mhz": self.bandwidth_hz / 1e6,
            "route_extent": self.route_extent(),
            "ap_centroids": self.ap_centroids().tolist(),
            "source": os.path.basename(self.source_path),
        }


def _as_complex(arr: np.ndarray) -> np.ndarray:
    """Rebuild a complex array from MATLAB's compound real/imag HDF5 dtype."""
    if arr.dtype.names and "real" in arr.dtype.names:
        return arr["real"] + 1j * arr["imag"]
    return arr


def _deref(f, obj):
    """Follow an HDF5 object reference, MATLAB's representation of a cell."""
    import h5py
    return f[obj] if isinstance(obj, h5py.Reference) else obj


def load_channels(path: str) -> DLocMeasurement:
    """Read a ``channels_<setup>.mat`` file.

    Raises a clear error if the file is missing, since the dataset needs a
    consent form and a separate download and is not in this repository.
    """
    import h5py

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found.\n"
            "The DLoc/WILD measurements are not redistributed with this code. "
            "They are obtained by consent form from the WILD dataset page "
            "linked in https://github.com/ucsdwcsng/DLoc_pt_code/blob/main/wild.md, "
            "then placed in dloc/data/.")

    setup = os.path.basename(path).replace("channels_", "").replace(".mat", "")
    with h5py.File(path, "r") as f:
        # HDF5 reverses MATLAB's dimension order, so transpose back
        channels = _as_complex(np.array(f["channels"])).transpose()
        rssi = np.array(f["RSSI"]).transpose()
        labels = np.array(f["labels"]).transpose()

        opt = f["opt"]
        freqs = np.array(opt["freq"]).reshape(-1)

        ap_group = f["ap"]
        ap_coords = []
        for ref in np.array(ap_group).reshape(-1):
            ap_coords.append(np.array(_deref(f, ref)).transpose().astype(float))
        ap_aoa = np.array(f["ap_aoa"]).reshape(-1)

    return DLocMeasurement(setup=setup, channels=channels, rssi=rssi,
                           labels=labels.astype(float), freqs=freqs.astype(float),
                           ap_coords=ap_coords, ap_aoa=ap_aoa, source_path=path)


def data_dir() -> str:
    """Where measurement files are expected to live."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def available_setups() -> List[str]:
    """Setup names whose channel file is present locally."""
    d = data_dir()
    if not os.path.isdir(d):
        return []
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith("channels_") and fn.endswith(".mat"):
            out.append(fn[len("channels_"):-len(".mat")])
    return out


def describe_availability() -> str:
    """Human-readable statement of what is and is not downloaded yet."""
    have = set(available_setups())
    lines = [f"data directory: {data_dir()}"]
    for env, meta in ENVIRONMENTS.items():
        lines.append(f"  {env} ({meta['label']}):")
        for s in meta["setups"]:
            lines.append(f"    [{'x' if s in have else ' '}] {s}")
    if not have:
        lines.append("\n  nothing downloaded yet. The measurements require a "
                     "consent form; see dloc/README.md.")
    return "\n".join(lines)