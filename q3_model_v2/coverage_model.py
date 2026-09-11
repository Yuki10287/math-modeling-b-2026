"""Conservative continuous-domain absence certificates for Q3.

A no-signal result at q rules out every possible source within 1000 m of q:
all Q3 sources are omnidirectional and have reception radius >= 1000 m.
We cover the full 1800 m target disk with closed square cells. A cell is
excluded only when ONE observed no-signal disk contains its entire square.
Therefore this is a conservative continuous certificate, not a sampled-point
coverage claim. The square overhang outside the target disk is harmless.

At the default 20 m resolution, the original seven stations still complete
the certificate: nearest-station distance on the target disk is <= 968.902 m,
and the additional square-centre plus half-diagonal margin is <= 28.285 m.
The online model uses only public negative observations, never source truth.
"""
from __future__ import annotations

from collections import OrderedDict
import math
from typing import Iterable

import numpy as np


class CoverageTracker:
    """Track arbitrary negative measurement locations independently per channel.

    ``observe`` must be called only after an accepted ``no_signal`` feedback.
    ``complete`` means that the source cannot be anywhere in the continuous
    target disk under the Q3 assumptions. Failure to complete is inconclusive.
    ``gain`` is an area-proxy count, not a probability of source discovery.
    """

    def __init__(self, channels: Iterable[int] = range(1, 21), *,
                 target_r: float = 1800.0, signal_r: float = 1000.0,
                 cell_size: float = 20.0, cache_size: int = 128):
        channels = tuple(int(c) for c in channels)
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channels must be distinct and nonempty")
        if not all(math.isfinite(v) and v > 0
                   for v in (target_r, signal_r, cell_size)):
            raise ValueError("radii and cell_size must be positive and finite")
        if cache_size < 1:
            raise ValueError("cache_size must be positive")
        self.channels = channels
        self._indices = {c: i for i, c in enumerate(channels)}
        self.target_r = float(target_r)
        self.signal_r = float(signal_r)
        self.cell_size = float(cell_size)
        self.cell_half_diagonal = self.cell_size / math.sqrt(2.0)
        # Coordinates cover [-ceil(R/h)*h, ceil(R/h)*h]^2, including boundary.
        n = math.ceil(self.target_r / self.cell_size)
        axis = (np.arange(-n, n, dtype=float) + 0.5) * self.cell_size
        x, y = np.meshgrid(axis, axis)
        centers = np.column_stack((x.ravel(), y.ravel()))
        # A square intersects the target disk iff its minimum distance to O
        # is <= target_r. Checking only cell centres would omit boundary arcs.
        nearest = np.maximum(np.abs(centers) - self.cell_size / 2.0, 0.0)
        keep = np.sum(nearest * nearest, axis=1) <= (self.target_r + 1e-9) ** 2
        self.centers = centers[keep]
        self.centers.flags.writeable = False
        self.total_cells = len(self.centers)
        self._excluded = np.zeros((len(channels), self.total_cells), dtype=bool)
        self._counts = np.zeros(len(channels), dtype=np.int64)
        self._points = {c: [] for c in channels}
        self._seen = {c: set() for c in channels}
        self._cache = OrderedDict()
        self._cache_size = int(cache_size)

    def _index(self, channel: int) -> int:
        try:
            return self._indices[channel]
        except KeyError as exc:
            raise ValueError(f"unknown channel {channel!r}") from exc

    @staticmethod
    def _point(q) -> tuple[float, float]:
        point = np.asarray(q, dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError("q must be a finite two-dimensional point")
        return float(point[0]), float(point[1])

    def _mask(self, q) -> np.ndarray:
        key = self._point(q)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # No rounded cache keys: nearby positions may certify different cells.
        inset_r = self.signal_r - self.cell_half_diagonal - 1e-9
        if inset_r < 0:
            mask = np.zeros(self.total_cells, dtype=bool)
        else:
            difference = self.centers - np.asarray(key)
            mask = np.sum(difference * difference, axis=1) <= inset_r ** 2
        mask.flags.writeable = False
        self._cache[key] = mask
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return mask

    def observe(self, channel: int, q) -> int:
        """Accept a no-signal point and return number of newly excluded cells."""
        index = self._index(channel)
        key = self._point(q)
        if key in self._seen[channel]:
            return 0
        mask = self._mask(key)
        newly = int(np.count_nonzero(mask & ~self._excluded[index]))
        self._excluded[index] |= mask
        self._counts[index] += newly
        self._seen[channel].add(key)
        self._points[channel].append(key)
        return newly

    def complete(self, channel: int) -> bool:
        return bool(self._counts[self._index(channel)] == self.total_cells)

    def uncovered_count(self, channel: int) -> int:
        return self.total_cells - int(self._counts[self._index(channel)])

    def channel_gain(self, q, channel: int) -> int:
        return int(np.count_nonzero(self._mask(q) & ~self._excluded[self._index(channel)]))

    def gain(self, q, channels: Iterable[int] | None = None) -> int:
        """Total new channel/cell exclusions if every given channel is negative."""
        mask = self._mask(q)
        selected = self.channels if channels is None else tuple(channels)
        return sum(int(np.count_nonzero(mask & ~self._excluded[self._index(c)]))
                   for c in selected)

    def uncovered_centers(self, channel: int, stride: int = 1) -> np.ndarray:
        """Return representative candidates, not a substitute for certification."""
        if stride < 1:
            raise ValueError("stride must be positive")
        return self.centers[~self._excluded[self._index(channel)]][::stride].copy()

    def certificate(self, channel: int) -> dict:
        index = self._index(channel)
        return dict(channel=int(channel), complete=bool(self.complete(channel)),
                    basis="whole_cell_negative_disk_cover",
                    target_radius_m=self.target_r, minimum_reception_radius_m=self.signal_r,
                    cell_size_m=self.cell_size, total_cells=self.total_cells,
                    excluded_cells=int(self._counts[index]),
                    uncovered_cells=self.uncovered_count(channel),
                    negative_points=[list(q) for q in self._points[channel]],
                    assumptions="Q3 fixed omnidirectional source; radius >= minimum_reception_radius_m; accepted no_signal feedback")
