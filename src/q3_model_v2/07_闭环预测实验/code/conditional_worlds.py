"""Evidence-conditioned design worlds with pending Q3 source beliefs.

Samples rank actions; they never replace continuous geometric certificates.
Historical angles are replayed at exact coordinates, while every new point
uses one fixed synthetic spatial error field. No real hidden API is read.
"""
from __future__ import annotations

import copy
import math
import numpy as np

import bootstrap
from belief_model import FeedbackModel
from environment import LocalArena
from local_policy import area_scenarios
from scan_preview import _worlds
from scenario_model import _copy_coverage
from validate_model import PublicAPI

ANGLE_BOUND = 1.005


class LedgerModel(FeedbackModel):
    """Same actions and returned objects as FeedbackModel, plus accepted history."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history = []

    def measure(self, q, channel):
        point = np.asarray(q, float).copy()
        reply = super().measure(q, channel)
        self.history.append(dict(action='measure', position=point.tolist(),
                                 channel=int(channel), reply=copy.deepcopy(reply)))
        return reply

    def clear(self, q, channel):
        point = np.asarray(q, float).copy()
        reply = super().clear(q, channel)
        self.history.append(dict(action='clear', position=point.tolist(),
                                 channel=int(channel), reply=copy.deepcopy(reply)))
        return reply


def _point_key(q, channel):
    point = np.asarray(q, float)
    return int(channel), float(point[0]), float(point[1])


def _history_map(model):
    if not isinstance(model, LedgerModel):
        raise ValueError('a complete LedgerModel history is required')
    result = {}
    for row in model.history:
        if row['action'] != 'measure' or row['channel'] in model.cleared:
            continue
        key = _point_key(row['position'], row['channel'])
        reply = row['reply']
        if key in result:
            previous = result[key]
            if any(previous.get(k) != reply.get(k) for k in ('measure_result', 'svd_deg')):
                raise ValueError('historical fixed-point feedback is inconsistent')
        result[key] = copy.deepcopy(reply)
    # All still-relevant previously measured points must have a real reply.
    for c, points in model.positions.items():
        if c in model.cleared:
            continue
        if any(_point_key(q, c) not in result for q in points):
            raise ValueError('historical measurement ledger is incomplete')
    return result


def _inside_polygon(point, polygon):
    polygon = np.asarray(polygon, float)
    if not len(polygon) or not np.isfinite(polygon).all():
        return False
    if len(polygon) == 1:
        return bool(np.linalg.norm(point - polygon[0]) <= 1e-7)
    edges = np.roll(polygon, -1, axis=0) - polygon
    area2 = float(np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1)
                         - polygon[:, 1] * np.roll(polygon[:, 0], -1)))
    if abs(area2) < 1e-7:
        distances = np.sum((polygon[:, None] - polygon[None, :]) ** 2, axis=2)
        i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
        delta = polygon[j] - polygon[i]
        fraction = float(np.dot(point - polygon[i], delta) / max(float(delta @ delta), 1e-30))
        nearest = polygon[i] + np.clip(fraction, 0., 1.) * delta
        return bool(np.linalg.norm(point - nearest) <= 1e-7)
    offsets = point - polygon
    cross = edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]
    return bool(np.all(cross * (1 if area2 > 0 else -1)
                       >= -1e-7 * np.maximum(1., np.linalg.norm(edges, axis=1))))


def _radius_interval(model, channel, point):
    """Closed numeric [L,U]; nextafter converts every strict negative bound."""
    point = np.asarray(point, float)
    if (point.shape != (2,) or not np.isfinite(point).all()
            or np.linalg.norm(point) > 1800.):
        return None
    if channel in model.beliefs and not _inside_polygon(point, model.beliefs[channel]['P']):
        return None
    lower, upper = 1000., 1500.
    for q in model.coverage._points[channel]:
        distance = float(np.linalg.norm(point - np.asarray(q)))
        upper = min(upper, float(np.nextafter(distance, -np.inf)))
    for row in model.history:
        if row['channel'] != channel:
            continue
        delta = point - np.asarray(row['position'], float)
        distance = float(np.linalg.norm(delta))
        reply = row['reply']
        if row['action'] == 'measure':
            kind = reply['measure_result']
            if kind == 'no_signal':
                upper = min(upper, float(np.nextafter(distance, -np.inf)))
            elif kind == 'near':
                if distance > 5.:
                    return None
                lower = max(lower, distance)
            elif kind == 'direction':
                if distance <= 5.:
                    return None
                bearing = math.degrees(math.atan2(delta[1], delta[0])) % 360.
                difference = (bearing - float(reply['svd_deg']) + 180.) % 360. - 180.
                if abs(difference) > ANGLE_BOUND + 1e-10:
                    return None
                lower = max(lower, distance)
            else:
                return None
        elif row['action'] == 'clear':
            # Every world contains only currently uncleared sources.
            if reply['clear_result'] == 'success' or distance <= 20.:
                return None
    return (lower, upper) if lower <= upper else None


def _validate_world(model, world):
    _history_map(model)
    known = set(model.beliefs)
    if model.discovered != model.cleared | known or model.cleared & known:
        raise ValueError('inconsistent discovered, cleared and belief channels')
    if not 10 <= len(model.cleared) + len(world) <= 16:
        raise ValueError('world source count violates 10..16')
    if not known <= set(world) or not set(world) <= known | set(model.unknown()):
        raise ValueError('world omits a known source or includes an unavailable channel')
    sources = []
    for channel, pair in world.items():
        if isinstance(channel, bool) or not isinstance(channel, (int, np.integer)):
            raise ValueError('invalid scenario channel')
        point, radius = np.asarray(pair[0], float), float(pair[1])
        interval = _radius_interval(model, channel, point)
        if (interval is None or not math.isfinite(radius)
                or not interval[0] <= radius <= interval[1]):
            raise ValueError('world contradicts historical feedback or geometric evidence')
        sources.append(dict(channel=int(channel), position=point.tolist(), radius=radius))
    return sources


def conditional_worlds(model, count=6):
    """Pair conditioned known-source quadrature with old unknown-source worlds.

Every known channel needs count compatible entries among its twelve initial
area samples. A thin/holed feasible region can yield too few; return [] and
let the caller keep lean, without inventing samples or using hidden truth.
"""
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or count not in (6, 12)):
        raise ValueError('scenario count must be 6 or 12')
    try:
        _history_map(model)
        if model.discovered != model.cleared | set(model.beliefs):
            return []
        banks = {}
        for channel, belief in sorted(model.beliefs.items()):
            if belief.get('conflict'):
                return []
            bank = []
            for point in area_scenarios(belief['P'], 12):
                interval = _radius_interval(model, channel, point)
                if interval is not None:
                    bank.append((np.asarray(point).copy(), interval))
            if len(bank) < count:
                return []
            banks[channel] = bank
        unknown_worlds = _worlds(model, model.unknown())
        if not unknown_worlds or len(unknown_worlds) < count:
            return []
        selected = unknown_worlds[::2] if count == 6 else unknown_worlds
        worlds = []
        for i, unknown in enumerate(selected[:count]):
            world = {c: (np.asarray(g).copy(), float(radius)) for c, (g, radius) in unknown.items()}
            for channel, bank in banks.items():
                point, (lower, upper) = bank[(i * len(bank) // count + channel) % len(bank)]
                fraction = (((i + channel) % 3) + .5) / 3.
                world[channel] = point.copy(), float(lower + fraction * (upper - lower))
            _validate_world(model, world)
            worlds.append(world)
        return worlds
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return []


class ScenarioArena(LocalArena):
    """Synthetic feedback with exact historical readings and ordinary fees."""
    def __init__(self, sources, *, historical_readings, historically_cleared,
                 field, scenario_index, start, channel):
        super().__init__(sources, seed=scenario_index, field=field, start=start)
        self.channel = int(channel)
        self._historical_readings = copy.deepcopy(historical_readings)
        self._historically_cleared = set(historically_cleared)

    def measure(self, q, channel):
        reply = super().measure(q, channel)  # travel, switching and sensing fees
        if channel in self._removed or channel in self._historically_cleared:
            return reply
        historical = self._historical_readings.get(_point_key(q, channel))
        if historical is None:
            return reply
        reply = copy.deepcopy(historical)
        event = self.events[-1]
        event.pop('svd_deg', None)
        event['measure_result'] = reply['measure_result']
        if 'svd_deg' in reply:
            event['svd_deg'] = reply['svd_deg']
        return reply


def clone_conditional(model, world, field='smooth', scenario_index=0):
    """Clone public evidence and supplied hypotheses, never the original API."""
    if (isinstance(scenario_index, bool)
            or not isinstance(scenario_index, (int, np.integer)) or scenario_index < 0):
        raise ValueError('scenario_index must be a nonnegative integer')
    if field not in ('smooth', 'extreme', 'hash', 'constant'):
        raise ValueError('unsupported scenario error field')
    sources = _validate_world(model, world)
    position = np.asarray(model.position, float).copy()
    channel = model.channel
    if (position.shape != (2,) or not np.isfinite(position).all()
            or isinstance(channel, bool) or not isinstance(channel, (int, np.integer))
            or not 1 <= channel <= 20):
        raise ValueError('invalid public receiver state')
    arena = ScenarioArena(sources, historical_readings=_history_map(model),
                          historically_cleared=model.cleared, field=field,
                          scenario_index=int(scenario_index), start=position, channel=channel)
    clone = LedgerModel(PublicAPI(arena), [], use_negatives=model.use_negatives,
                        range_cuts=model.range_cuts, search_fraction=model.search_fraction)
    clone.coverage = _copy_coverage(model.coverage)
    clone.positions = {c: [np.asarray(q).copy() for q in points] for c, points in model.positions.items()}
    clone.beliefs = copy.deepcopy(model.beliefs)
    clone.discovered, clone.cleared = set(model.discovered), set(model.cleared)
    clone.history = copy.deepcopy(model.history)
    return clone, arena
