"""Isolated Q3 design worlds conditioned only on the solver's evidence.

These worlds are planning quadrature, not environment truth or a calibrated
posterior.  Cloning is restricted to moments when every discovered source
has been cleared, so surviving historical measurements are all no_signal.
No real API member other than position and channel is read.
"""
from __future__ import annotations

import numpy as np

import bootstrap  # the experiment bootstrap installs the frozen import path
from belief_model import FeedbackModel
from coverage_model import CoverageTracker
from environment import LocalArena
from scan_preview import _worlds
from validate_model import PublicAPI


def _eligible(model):
    return not model.beliefs and model.discovered == model.cleared


def scenario_worlds(model, count=6):
    """Return up to 6 or 12 independent copies of the old design worlds.

Six worlds are the evenly spaced indices 0,2,4,6,8,10 of the old twelve.
The old sampler and its global constants are never changed. Empty output
means the gate failed or the finite prior contains no compatible world.
"""
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or count not in (6, 12)):
        raise ValueError('scenario count must be 6 or 12')
    if not _eligible(model):
        return []
    worlds = _worlds(model, model.unknown())
    if not worlds:
        return []
    selected = worlds[::2] if count == 6 else worlds
    return [{int(c): (np.asarray(g, float).copy(), float(radius))
             for c, (g, radius) in world.items()} for world in selected[:count]]


def _validated_sources(model, world):
    if not _eligible(model):
        raise ValueError('world cloning requires all discovered sources cleared')
    if not 10 <= len(model.discovered) + len(world) <= 16:
        raise ValueError('scenario source count violates the 10..16 bound')
    unknown = set(model.unknown())
    sources = []
    for channel, pair in world.items():
        if (isinstance(channel, bool) or not isinstance(channel, (int, np.integer))
                or channel not in unknown):
            raise ValueError('world contains a non-unknown channel')
        point, radius = np.asarray(pair[0], float), float(pair[1])
        if (point.shape != (2,) or not np.isfinite(point).all()
                or np.linalg.norm(point) > model.coverage.target_r + 1e-9
                or not np.isfinite(radius) or not 1000. <= radius <= 1500.):
            raise ValueError('invalid scenario source position or radius')
        # At this gate an unknown channel can only have negative history.
        # Checking both ledgers also rejects manually supplied worlds that
        # would contradict a repeated historical measurement.
        negatives = list(model.coverage._points[channel]) + list(model.positions[channel])
        if any(np.linalg.norm(point - np.asarray(q, float)) <= radius for q in negatives):
            raise ValueError('scenario contradicts an accepted historical no_signal')
        sources.append(dict(channel=int(channel), position=point.tolist(), radius=radius))
    return sources


def _copy_coverage(original):
    clone = CoverageTracker(channels=original.channels, target_r=original.target_r,
                            signal_r=original.signal_r, cell_size=original.cell_size,
                            cache_size=original._cache_size)
    if not np.array_equal(clone.centers, original.centers):
        raise ValueError('unsupported coverage grid geometry')
    clone._excluded = original._excluded.copy()
    clone._counts = original._counts.copy()
    clone._points = {c: [tuple(q) for q in points] for c, points in original._points.items()}
    clone._seen = {c: set(points) for c, points in original._seen.items()}
    # The observation mask cache is a performance detail, not evidence.
    # A fresh cache prevents scenario decisions from mutating the real one.
    return clone


def clone_for_world(model, world, field='smooth', scenario_index=0):
    """Return (FeedbackModel copy, synthetic LocalArena) with incremental time.

The arena contains only the still-undiscovered scenario sources. Its time,
counts, events and evaluation therefore concern the simulated continuation;
the model's discovered/cleared sets retain the historical source count.

Unknown historical no_signal records are reproduced by geometric
compatibility, with the same fixed radius. Cleared channels remain silent.
There are no surviving historical bearings at this gate; all future
bearings use one deterministic spatial field, including repeated points.
"""
    if (isinstance(scenario_index, bool)
            or not isinstance(scenario_index, (int, np.integer)) or scenario_index < 0):
        raise ValueError('scenario_index must be a nonnegative integer')
    if field not in ('smooth', 'hash', 'extreme', 'constant'):
        raise ValueError('unsupported scenario error field')
    sources = _validated_sources(model, world)
    # This is the complete permitted interaction with the original API.
    position = np.asarray(model.position, float).copy()
    channel = model.channel
    if (position.shape != (2,) or not np.isfinite(position).all()
            or isinstance(channel, bool) or not isinstance(channel, (int, np.integer))
            or not 1 <= channel <= 20):
        raise ValueError('invalid public receiver state')
    arena = LocalArena(sources, seed=int(scenario_index), field=field, start=position)
    arena.channel = int(channel)
    clone = FeedbackModel(PublicAPI(arena), [], use_negatives=model.use_negatives,
                          range_cuts=model.range_cuts, search_fraction=model.search_fraction)
    clone.coverage = _copy_coverage(model.coverage)
    clone.positions = {c: [np.asarray(q, float).copy() for q in points]
                       for c, points in model.positions.items()}
    clone.discovered = set(model.discovered)
    clone.cleared = set(model.cleared)
    return clone, arena
