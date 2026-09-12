"""Finite, history-conditioned predictive worlds for a local planning window.

These are private working hypotheses, never environment truth or absence
certificates. Only public observations enter their deterministic seed. Unknown
sources use the stated uniform count/subset/position/radius/type/heading prior.
Known active sources use uniform proposals on the current convex position
region and rejection against every actual observation. Rounded bearing support
is checked, but its likelihood is not calibrated; no exact posterior or policy
improvement theorem is claimed. Already-cleared source geometry is integrated
out: its existence still counts toward the sampled total.

The two prediction functions are pure: callers own position, clock, radio and
the cleared set. In particular a clear never retunes the radio. Previously
observed measurements are reused only for an uncleared source; other points
use a fixed, world-specific bounded error map. All insufficient-support cases
return no worlds, instructing the caller to use its existing policy.
"""
import copy
import hashlib
import json
import math

import numpy as np

MODEL_ID = 'window_worlds_v1'
MAX_WORLDS = 6
UNKNOWN_DRAWS = 4096
KNOWN_DRAWS = 8192
KNOWN_BATCH = 256
KNOWN_TARGET = 48


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _point(point):
    _require(len(point) == 2, 'point_shape')
    q = tuple(float(x) for x in point)
    _require(all(math.isfinite(x) for x in q), 'nonfinite_point')
    return tuple(0. if x == 0 else x for x in q)


def _channel(value):
    c = int(value)
    _require(not isinstance(value, bool) and c == float(value) and 1 <= c <= 20, 'channel')
    return c


def _key(channel, point):
    x, y = _point(point)
    return f'{_channel(channel)}:{x.hex()}:{y.hex()}'


def _digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _rng(history_digest, label):
    seed = int(hashlib.sha256((history_digest+'|'+label).encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def _normalise_ledger(ledger):
    """Read only action/coordinates/channel and public reply fields, no clock."""
    output, cache, positive_channels, successes = [], {}, set(), set()
    for item in ledger:
        action, channel, q = item['action'], _channel(item['channel']), _point(item['position'])
        _require(action in ('measure', 'clear'), 'ledger_action')
        _require(item.get('accepted', True) is not False, 'unaccepted_ledger_action')
        name = 'measure_result' if action == 'measure' else 'clear_result'
        reply = item.get('reply', {})
        result = item.get(name, item.get('result', reply.get(name)))
        row = dict(action=action, channel=channel, position=list(q))
        if action == 'measure':
            _require(result in ('direction', 'near', 'no_signal'), 'measurement_kind')
            row[name] = result
            if result != 'no_signal':
                _require(channel not in successes, 'positive_after_clear')
                positive_channels.add(channel)
            if result == 'direction':
                bearing = float(item.get('svd_deg', reply.get('svd_deg')))
                _require(math.isfinite(bearing) and 0 <= bearing < 360 and round(bearing, 2) == bearing,
                         'bearing_not_two_decimal_degrees')
                row['svd_deg'] = bearing
            response = {k:row[k] for k in ('measure_result', 'svd_deg') if k in row}
            key = _key(channel, q)
            _require(key not in cache or cache[key] == response, 'inconsistent_repeated_measurement')
            cache[key] = response
        else:
            _require(result in ('success', 'no_target_in_range'), 'clear_kind')
            row[name] = result
            if result == 'success':
                _require(channel not in successes, 'duplicate_successful_clear')
                successes.add(channel)
                cache = {k:v for k,v in cache.items() if not k.startswith(str(channel)+':')}
        output.append(row)
    return output, cache, positive_channels, successes


def _bearing_supported(theta, reading):
    """Exhibit an error in [-1,1] that rounds to this actual two-decimal reply."""
    difference = (reading-theta+180.) % 360.-180.
    error = min(1., max(-1., difference))
    return round((theta+error) % 360., 2) % 360. == reading


def _visible(source, point):
    dx, dy = point[0]-source['position'][0], point[1]-source['position'][1]
    return dx*dx+dy*dy <= source['radius']**2 and (
        source['type'] == 'omni' or dx*source['heading'][0]+dy*source['heading'][1] >= 0.)


def _consistent(source, records):
    g = source['position']
    if sum(x*x for x in g) > 1800.**2 or not 1000 <= source['radius'] <= 1500:
        return False
    for row in records:
        q = row['position']
        distance2 = (q[0]-g[0])**2+(q[1]-g[1])**2
        if row['action'] == 'clear':
            if (distance2 <= 20.**2) != (row['clear_result'] == 'success'):
                return False
            continue
        kind, visible = row['measure_result'], _visible(source, q)
        if kind == 'no_signal':
            if visible:
                return False
        elif not visible or (distance2 <= 5.**2) != (kind == 'near'):
            return False
        elif kind == 'direction':
            theta = math.degrees(math.atan2(g[1]-q[1], g[0]-q[0])) % 360.
            if not _bearing_supported(theta, row['svd_deg']):
                return False
    return True


def _cloud(rng, positions):
    size = len(positions)
    angles = rng.uniform(0., 2*math.pi, size)
    return (np.asarray(positions, float), rng.uniform(1000., 1500., size),
            rng.random(size) < .5, np.c_[np.cos(angles), np.sin(angles)])


def _source(cloud, index):
    positions, radii, omni, headings = cloud
    return dict(position=positions[index].tolist(), radius=float(radii[index]),
                type='omni' if omni[index] else 'directional',
                heading=None if omni[index] else headings[index].tolist())


def _filter(cloud, records):
    positions, radii, omni, headings = cloud
    mask = np.sum(positions*positions, axis=1) <= 1800.**2
    for row in records:
        delta = np.asarray(row['position'])-positions
        distance2 = np.sum(delta*delta, axis=1)
        if row['action'] == 'clear':
            mask &= (distance2 <= 400.) if row['clear_result'] == 'success' else (distance2 > 400.)
            continue
        visible = (distance2 <= radii*radii) & (omni | (np.sum(delta*headings, axis=1) >= 0.))
        kind = row['measure_result']
        if kind == 'no_signal':
            mask &= ~visible
        elif kind == 'near':
            mask &= visible & (distance2 <= 25.)
        else:
            angle = np.degrees(np.arctan2(-delta[:, 1], -delta[:, 0])) % 360.
            error = (row['svd_deg']-angle+180.) % 360.-180.
            # This is a cheap outer screen only. Scalar Python rounding below
            # must exhibit an admissible error before any particle is accepted.
            mask &= visible & (distance2 > 25.) & (np.abs(error) <= 1.006)
    return np.flatnonzero(mask)


def _hull(points):
    points = sorted(set(_point(p) for p in points))
    _require(bool(points), 'empty_position_region')
    if len(points) < 3:
        return np.asarray(points)
    def cross(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves = []
    for sequence in (points, points[::-1]):
        stack = []
        for p in sequence:
            while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
                stack.pop()
            stack.append(p)
        halves.append(stack)
    return np.asarray(halves[0][:-1]+halves[1][:-1])


def _positions(rng, polygon, size):
    if len(polygon) == 1:
        return np.repeat(polygon, size, axis=0)
    if len(polygon) == 2:
        return polygon[0]+rng.random((size,1))*(polygon[1]-polygon[0])
    a, b = polygon[1:-1]-polygon[0], polygon[2:]-polygon[0]
    areas = np.abs(a[:,0]*b[:,1]-a[:,1]*b[:,0])
    _require(float(areas.sum()) > 0, 'zero_area_polygon')
    triangles = rng.choice(len(areas), size=size, p=areas/areas.sum())
    u, v = np.sqrt(rng.random((size,1))), rng.random((size,1))
    return (1-u)*polygon[0]+u*((1-v)*polygon[triangles+1]+v*polygon[triangles+2])


def _known_pool(belief, records, rng, count):
    polygon, accepted, drawn = _hull(belief.P), [], 0
    while drawn < KNOWN_DRAWS and len(accepted) < KNOWN_TARGET:
        size = min(KNOWN_BATCH, KNOWN_DRAWS-drawn)
        cloud = _cloud(rng, _positions(rng, polygon, size))
        drawn += size
        for index in _filter(cloud, records):
            source = _source(cloud, index)
            if _consistent(source, records):
                accepted.append(source)
                if len(accepted) == KNOWN_TARGET:
                    break
    return accepted, dict(proposals=drawn, compatible_particles=len(accepted), sufficient=len(accepted) >= count)


def _suffix_polynomials(weights):
    table = np.zeros((len(weights)+1, 17))
    table[-1,0] = 1.
    for i in range(len(weights)-1, -1, -1):
        table[i] = table[i+1]
        table[i,1:] += weights[i]*table[i+1,:-1]
    return table


def _select_channels(channels, weights, suffix, size, rng):
    selected = []
    for i, c in enumerate(channels):
        if size == 0:
            break
        denominator = suffix[i,size]
        _require(denominator > 0, 'empty_channel_subset_support')
        probability = weights[i]*suffix[i+1,size-1]/denominator
        if rng.random() < min(1., max(0., probability)):
            selected.append(c)
            size -= 1
    _require(size == 0, 'channel_subset_sampling_failed')
    return selected


def sample_worlds(beliefs, cover, discovered, cleared, ledger, count=6):
    diagnostics = dict(model=MODEL_ID, worlds_are_predictions_only=True,
        unknown_draw_budget=UNKNOWN_DRAWS, known_draw_budget_per_channel=KNOWN_DRAWS,
        known_target=KNOWN_TARGET, known_pools={}, unknown_support={},
        prior='N uniform 10..16; uniform channel subsets; independent uniform disk1800 position, radius1000..1500, half type, uniform heading.',
        approximation='Finite rejection support; known bearing likelihood is support-conditioned, not calibrated. Cleared-source geometry is integrated out.')
    try:
        _require(isinstance(count,int) and not isinstance(count,bool) and 1 <= count <= MAX_WORLDS, 'world_count')
        history, cache, positives, successes = _normalise_ledger(ledger)
        discovered, cleared = set(map(_channel,discovered)), set(map(_channel,cleared))
        beliefs = {_channel(c):b for c,b in beliefs.items()}
        _require(cleared <= discovered and len(discovered) <= 16, 'discovered_count_or_cleared_set')
        _require(successes == cleared and positives | successes == discovered, 'history_status_mismatch')
        _require(set(beliefs) == discovered-cleared, 'active_belief_status_mismatch')
        seed_hash = _digest(history)
        diagnostics.update(history_sha256=seed_hash, actual_discovered=len(discovered),
                           integrated_out_cleared=sorted(cleared))
        records = {c:[row for row in history if row['channel']==c] for c in range(1,21)}
        # Validate the public belief/cover ledgers against accepted observations.
        for c in range(1,21):
            negative = {_point(row['position']) for row in records[c]
                        if row.get('measure_result') == 'no_signal'}
            _require(all(_point(p) in negative for p in cover.negative[c]), 'unbound_cover_negative')
            if c in beliefs:
                positive = {_point(row['position']) for row in records[c]
                            if row.get('measure_result') in ('near','direction')}
                _require(bool(positive), 'active_source_without_positive')
                _require(all(_point(p) in positive for p in beliefs[c].positives), 'unbound_belief_positive')
                _require(all(_point(p) in negative for p in beliefs[c].negatives), 'unbound_belief_negative')
        pools = {}
        for c, belief in sorted(beliefs.items()):
            pools[c], info = _known_pool(belief, records[c], _rng(seed_hash,'known:'+str(c)), count)
            diagnostics['known_pools'][str(c)] = info
            _require(info['sufficient'], 'insufficient_known_support:'+str(c))
        D = len(discovered)
        excluded = [c for c in range(1,21) if c not in discovered and cover.complete(c)]
        eligible = [c for c in range(1,21) if c not in discovered and c not in excluded] if D < 16 else []
        diagnostics['existing_cover_absent_channels'] = excluded
        unknown_cloud, surviving, weights = None, {}, []
        if eligible:
            rng = _rng(seed_hash,'unknown_prior')
            radial, angle = 1800*np.sqrt(rng.random(UNKNOWN_DRAWS)), rng.uniform(0,2*math.pi,UNKNOWN_DRAWS)
            unknown_cloud = _cloud(rng, np.c_[radial*np.cos(angle),radial*np.sin(angle)])
            for c in eligible:
                ids = _filter(unknown_cloud, records[c])
                # Unknown records cannot contain positives/successes by the
                # status check, so the vector distance/visibility test suffices.
                surviving[c] = ids
                diagnostics['unknown_support'][str(c)] = len(ids)
                _require(len(ids) >= count, 'insufficient_unproven_unknown_support:'+str(c))
                weights.append(len(ids)/UNKNOWN_DRAWS)
        suffix = _suffix_polynomials(weights)
        totals = [n for n in range(max(10,D),17) if n-D <= len(eligible)]
        masses = np.array([suffix[0,n-D]/math.comb(20,n) for n in totals])
        _require(bool(len(masses)) and float(masses.sum()) > 0, 'total_count_support_empty')
        masses /= masses.sum()
        diagnostics['total_count_posterior'] = {str(n):float(w) for n,w in zip(totals,masses)}
        diagnostics['count_formula'] = 'e_(N-D)(w_c)/C(20,N); equals proportional C(N,D)*w^(N-D) for equal unknown ledgers.'
        worlds, count_rng = [], _rng(seed_hash,'count_stratification')
        for i in range(count):
            rng = _rng(seed_hash,'world:'+str(i))
            quantile = (i+float(count_rng.random()))/count
            n = totals[min(int(np.searchsorted(np.cumsum(masses),quantile)),len(totals)-1)]
            sources = {str(c):copy.deepcopy(pool[int(rng.integers(len(pool)))]) for c,pool in sorted(pools.items())}
            for c in _select_channels(eligible,weights,suffix,n-D,rng):
                sources[str(c)] = _source(unknown_cloud,int(rng.choice(surviving[c])))
            _require(len(sources)+len(cleared) == n, 'sampled_count_mismatch')
            for c, source in sources.items():
                _require(_consistent(source,records[int(c)]), 'selected_particle_history_conflict:'+c)
            worlds.append(dict(model=MODEL_ID, prediction_only=True, weight=1/count,
                sources=sources, base_cleared=sorted(cleared), sampled_total_count=n,
                historical_measure_cache=copy.deepcopy(cache),
                error_salt=hashlib.sha256((seed_hash+'|errors|'+str(i)).encode()).hexdigest()))
        diagnostics.update(passed=True, reason='sampled', count=len(worlds),
                           world_sha256=[_digest(world) for world in worlds])
        return dict(worlds=worlds,diagnostics=diagnostics)
    except (ValueError,TypeError,KeyError,IndexError,OverflowError,AttributeError) as exc:
        diagnostics.update(passed=False,reason=str(exc),count=0,fallback_required=True)
        return dict(worlds=[],diagnostics=diagnostics)


def _inactive(world, channel, cleared):
    inactive = set(world['base_cleared']) | (set(map(_channel,cleared)) if cleared is not None else set())
    return channel in inactive or str(channel) not in world['sources']


def predict_measure(world, q, channel, cleared=None):
    """Pure fixed feedback. A prior successful clear overrides old positives."""
    q, channel = _point(q), _channel(channel)
    if _inactive(world,channel,cleared):
        return dict(measure_result='no_signal')
    cached = world['historical_measure_cache'].get(_key(channel,q))
    if cached is not None:
        return dict(cached)
    source = world['sources'][str(channel)]
    if not _visible(source,q):
        return dict(measure_result='no_signal')
    g = source['position']
    if (q[0]-g[0])**2+(q[1]-g[1])**2 <= 25.:
        return dict(measure_result='near')
    theta = math.degrees(math.atan2(g[1]-q[1],g[0]-q[0])) % 360.
    digest = hashlib.sha256((world['error_salt']+'|'+_key(channel,q)).encode()).digest()
    error = 2*(int.from_bytes(digest[:8],'big')+.5)/(2**64)-1
    return dict(measure_result='direction',svd_deg=round((theta+error)%360.,2)%360.)


def predict_clear(world, q, channel, cleared=None):
    """Pure optical feedback; caller applies success, position and action fee."""
    q, channel = _point(q), _channel(channel)
    if _inactive(world,channel,cleared):
        return dict(clear_result='no_target_in_range')
    g = world['sources'][str(channel)]['position']
    hit = (q[0]-g[0])**2+(q[1]-g[1])**2 <= 400.
    return dict(clear_result='success' if hit else 'no_target_in_range')
