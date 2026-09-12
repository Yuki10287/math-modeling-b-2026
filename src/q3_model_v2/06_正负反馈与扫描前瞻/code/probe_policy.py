"""Rank uncertain-reception probes using explicit positive/negative branches.

Finite position quadrature and conditional uniform radius are design choices,
not an official prior or a completion certificate. Real feedback uses the
unchanged conservative FeedbackModel.
"""
import math
import numpy as np
from bootstrap import core, exclude_disk_hull
from local_policy import area_scenarios, short_candidates


def radius_scenarios(model, channel, count=8):
    item = model.beliefs[channel]
    positives = np.asarray(item['positives'])
    negatives = np.asarray(model.coverage.certificate(channel)['negative_points']).reshape(-1, 2)
    failed = [np.asarray(row['position']) for row in model.trace
              if row.get('phase')=='belief_update' and row.get('channel')==channel
              and row.get('reason')=='failed_optical_clear']
    rows = []
    for g in area_scenarios(item['P'], count):
        if np.linalg.norm(g)>1800 or any(np.linalg.norm(g-q)<=20 for q in failed):
            continue
        lo = max(1000., float(np.linalg.norm(positives-g, axis=1).max()))
        hi = min(1500., float(np.linalg.norm(negatives-g, axis=1).min())-1e-6) if len(negatives) else 1500.
        if lo <= hi:
            rows.append((g, lo, hi))
    return rows


def negative_region(P, q, positives):
    region = exclude_disk_hull(P, q, 1000.)
    for w in positives:
        region = core.clip(region, 2*(q-w), float(q@q-w@w))
        if not len(region):
            break
    return region


def positive_region(P, q, g, error, negatives):
    if np.linalg.norm(g-q) <= 5:
        region = core.disk_clip(P, q, 5.)
    else:
        bearing = math.degrees(math.atan2(*(g-q)[::-1])) % 360
        reading = round((bearing+error) % 360, 2) % 360
        region = core.disk_clip(core.wedge(P, q, reading), q)
    for p in negatives:
        region = core.clip(region, 2*(p-q), float(p@p-q@q))
    return region


def legal_tail(P, s, witness, g, error, observed, negatives, steps=2):
    """A fixed, reception-certified continuation plus an explicit terminal proxy."""
    P = P.copy()
    position = s.copy()
    witness = witness.copy()
    seen = list(observed)
    total = 0.
    for _ in range(steps):
        center, radius = core.mec(P)
        safe = core.nearest_certified_clear(P, position) if radius <= 20 else None
        if safe is not None:
            return total+float(np.linalg.norm(safe-position))/5+5
        _, (i, j) = core.diameter(P)
        axis = P[j]-P[i]
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        desired = center+.3*radius*np.array([-axis[1], axis[0]])
        used = lambda p:any(np.linalg.norm(p-old) <= .1 for old in seen)
        if not used(desired) and core.reception_certified(P, desired, witness):
            q = desired
        else:
            options = [p for p in short_candidates(P, position, witness) if not used(p)]
            if not options:
                return math.inf
            q = min(options, key=lambda p:float(np.linalg.norm(p-desired)))
        total += float(np.linalg.norm(q-position))/5+5
        P = positive_region(P, q, g, error, negatives)
        if not len(P):
            return math.inf
        if np.linalg.norm(g-q) <= 5:
            return total+5
        position = q.copy()
        witness = q.copy()
        seen.append(q.copy())
    center, radius = core.mec(P)
    return total+float(np.linalg.norm(center-position))/5+5+2*radius/5+10


def score_probe(model, channel, q, scenarios):
    item = model.beliefs[channel]
    P, s = item['P'], np.asarray(model.position)
    positives = [np.asarray(p) for p in item['positives']]
    negatives = [np.asarray(p) for p in model.coverage.certificate(channel)['negative_points']]
    values, probabilities = [], []
    neg = None
    for g, lo, hi in scenarios:
        distance = float(np.linalg.norm(g-q))
        probability = float(distance <= lo) if hi-lo < 1e-8 else float(np.clip((hi-distance)/(hi-lo), 0, 1))
        probabilities.append(probability)
        per_error = []
        for error in (-1., 0., 1.):
            value = 0.
            if probability > 0:
                region = positive_region(P, q, g, error, negatives)
                if not len(region):
                    return None
                cost = 5. if distance <= 5 else legal_tail(region, q, q, g, error,
                    [*model.positions[channel], q], negatives)
                value += probability*cost
            if probability < 1:
                if neg is None:
                    neg = negative_region(P, q, positives)
                if not len(neg):
                    return None
                cost = legal_tail(neg, q, item['witness'], g, error,
                    [*model.positions[channel], q], [*negatives, q])
                value += (1-probability)*cost
            per_error.append(value)
        values.append(float(np.mean(per_error)))
    if not values or not np.isfinite(values).all():
        return None
    score = float(np.linalg.norm(q-s))/5+5+int(model.channel != channel)
    score += .8*float(np.mean(values))+.2*float(np.max(values))
    return dict(score=score, signal_mass=float(np.mean(probabilities)), scenarios=len(values))


def candidates(model, channel, hints):
    item = model.beliefs[channel]
    P, s, w = item['P'], np.asarray(model.position), item['witness']
    center, radius = core.mec(P)
    _, (i, j) = core.diameter(P)
    axis = P[j]-P[i]
    axis /= max(float(np.linalg.norm(axis)), 1e-9)
    perp = np.array([-axis[1], axis[0]])
    raw = [s.copy()]
    for offset in (20., 80.):
        raw += [s+offset*perp, s-offset*perp]
    for p in hints[:2]:
        p = np.asarray(p)
        v = p-s
        t = np.clip(float((center-s)@v)/max(float(v@v), 1e-9), 0, 1)
        raw += [s+t*v, p, (s+p)/2]
    # These candidates emphasize small movement and following the planned corridor.
    out = []
    for q in raw:
        if not np.isfinite(q).all() or np.any(np.abs(q)>2_000_000):
            continue
        if model.recorded(channel, q) or core.reception_certified(P, q, w):
            continue
        if all(np.linalg.norm(q-old)>.1 for old in out):
            out.append(q.copy())
    return sorted(out, key=lambda q:float(np.linalg.norm(q-s)))[:6]


def choose_probe(model, channel, fallback, hints=()):
    if fallback['kind'] != 'measure' or model.beliefs[channel]['state'] is not None:
        return fallback
    options = candidates(model, channel, hints)
    if not options:
        return fallback
    scenarios = radius_scenarios(model, channel)
    if len(scenarios) < 4:
        return fallback
    base = score_probe(model, channel, fallback['q'], scenarios)
    if base is None:
        return fallback
    ranked = []
    for index, q in enumerate(options):
        score = score_probe(model, channel, q, scenarios)
        if score is not None and score['signal_mass'] >= .05:
            ranked.append((score['score'], index, q, score))
    best = min(ranked, default=None, key=lambda row:(row[0], row[1]))
    chosen = best is not None and best[0] < base['score']-5.
    model.trace.append(dict(phase='probe_evaluation', channel=channel, candidates=len(options),
        valid_candidates=len(ranked), baseline_score=base['score'],
        candidate_score=best[0] if best else None, selected=chosen))
    if not chosen:
        return fallback
    _, _, q, score = best
    return dict(kind='measure', q=q, state=None, certified=False, uncertain_reception=True,
        rationale='positive_negative_probe', estimated_s=score['score'],
        design_signal_mass=score['signal_mass'], design_scenarios=score['scenarios'])
