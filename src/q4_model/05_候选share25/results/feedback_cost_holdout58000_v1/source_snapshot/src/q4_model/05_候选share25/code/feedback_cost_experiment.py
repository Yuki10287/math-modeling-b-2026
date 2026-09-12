"""Local research: feedback-conditioned cost of the existing ordered fallback.

Preserve the original ten view points, outer routing, sharing, 25-station proof,
and executed optical order. Only the service action comparison changes.
Each hypothetical observation is applied to a private belief; its fallback uses
the same certified-clear rule and the same original-path ordered pruning.
The .8 work expectation/.2 branch-endpoint risk is a ranking heuristic. Positive
readings remain finite quadrature; this is not a global robust-time guarantee.
"""
import ast
import copy
import hashlib
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
sys.path.insert(0, str(Q4))

import ordered_optical_prune as baseline
from localization import optical_plan, samples
from shared import core
from receive_guarantee_v1 import receipt_certificate

MODEL_ID = 'feedback_cost_v1'
BASELINE_SHA256 = '2854b5d7e38ec5a56f9eb02bdbb126bb71b9c494030d0b3ae7e803e344d0f92a'
MEAN_WEIGHT = .8
RISK_WEIGHT = .2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def forecast_clear(belief, start, plan=None):
    """Forecast immediate fallback using only this (possibly hypothetical) belief.

    Pruned-path mean integrates new points in the retained position polygon.
    This is a working position measure, not an official posterior. Sampling
    outside the original polygon is discarded; uncovered/empty quadrature
    falls back conservatively to the full kept-path cost, never zero.
    """
    start = np.asarray(start, float)
    q = belief.near
    kind = 'near' if q is not None else 'certified_clear'
    if q is None:
        q = core.nearest_certified_clear(belief.P, start)
    if q is not None:
        value = float(np.linalg.norm(np.asarray(q)-start))/5+5
        return dict(kind=kind, mean_s=value, worst_s=value, score=value,
                    clearpoint=np.asarray(q).tolist(), original_points=1, kept_points=1,
                    removed_points=0, contraction_applied=False,
                    sample_count=0, sampled_uncovered=0, mean_fallback_to_worst=False)
    plan = optical_plan(belief.P, start) if plan is None else plan
    proof = baseline.prune_path(belief.P, belief.positives, belief.negatives, plan['path'])
    path = np.asarray([plan['path'][i] for i in proof['kept_indices']])
    require(bool(len(path)), 'empty kept optical path')
    cumulative = np.cumsum(np.linalg.norm(np.diff(np.vstack((start, path)), axis=0), axis=1))/5
    worst = float(cumulative[-1]+3*(len(path)-1)+5)
    region = np.asarray(proof['polygon'])
    source_hull = baseline.exact_hull(belief.P)
    quadrature = np.asarray([g for g in samples(region)
                            if baseline.distance_squared(g, source_hull) == 0]).reshape((-1, 2))
    uncovered = 0
    fallback = not len(quadrature)
    if len(quadrature):
        hits = np.linalg.norm(quadrature[:, None, :]-path, axis=2) <= 20
        uncovered = int((~hits.any(axis=1)).sum())
        fallback = bool(uncovered)
        first = hits.argmax(axis=1)
        mean = worst if fallback else float(np.mean(cumulative[first]+3*first+5))
    else:
        mean = worst
    return dict(kind='ordered_optical', mean_s=mean, worst_s=worst,
        score=MEAN_WEIGHT*mean+RISK_WEIGHT*worst,
        original_points=len(plan['path']), kept_points=len(path),
        removed_points=len(proof['removed_indices']), contraction_applied=proof['info']['applied'],
        sample_count=len(quadrature), sampled_uncovered=uncovered,
        mean_fallback_to_worst=fallback, conditional_polygon=region.tolist(),
        kept_indices=proof['kept_indices'], original_path=np.asarray(plan['path']).tolist(),
        contraction_reason=proof['info']['reason'])


def compact_forecast(forecast):
    return {k:v for k, v in forecast.items()
            if k not in ('conditional_polygon', 'kept_indices', 'original_path')}


def score_measure(belief, start, q, channel, current_channel, prepared=None, diagnostics=True):
    """Keep unexcluded negative outcomes in risk, even at sampled signal mass 1."""
    q, start = np.asarray(q, float), np.asarray(start, float)
    if any(np.linalg.norm(q-p) <= .1 for p in belief.measured):
        return None
    gs, scenarios = belief.scenarios() if prepared is None else prepared
    if not scenarios:
        return None
    all_weights, hit_weights = np.zeros(len(gs)), np.zeros(len(gs))
    total_weight = sum(row[4] for row in scenarios)
    for index, g, heading, radius, weight in scenarios:
        mass = weight/total_weight
        all_weights[index] += mass
        if np.linalg.norm(q-g) <= radius and (heading is None or (q-g)@heading >= 0):
            hit_weights[index] += mass
    raw_probability = float(hit_weights.sum())
    guarantee = receipt_certificate(belief.P, belief.positives, q)
    if guarantee['certified']:
        # A continuous certificate overrides finite-grid/radius rounding.
        hit_weights = all_weights
    probability = float(hit_weights.sum())
    if probability < .05:
        return None
    # None means proven impossible; otherwise also keep sampling-missed negatives.
    negative = None
    if not guarantee['certified']:
        hypothetical = copy.deepcopy(belief)
        hypothetical.update(q, {'measure_result': 'no_signal'})
        negative = forecast_clear(hypothetical, q)
    positive_mean, branch_worst, branches = 0., [], []
    if negative is not None:
        branch_worst.append(negative['worst_s'])
    for index in np.flatnonzero(hit_weights):
        g = gs[index]
        readings = [{'measure_result': 'near'}] if np.linalg.norm(g-q) <= 5 else [
            {'measure_result': 'direction', 'svd_deg': round((
                math.degrees(math.atan2(g[1]-q[1], g[0]-q[0]))+error) % 360, 2) % 360}
            for error in (-1., 0., 1.)]
        values = []
        for reply in readings:
            hypothetical = copy.deepcopy(belief)
            hypothetical.update(q, reply)
            outcome = forecast_clear(hypothetical, q)
            values.append(outcome['mean_s'])
            branch_worst.append(outcome['worst_s'])
            if diagnostics:
                branches.append(dict(position_index=int(index), feedback=reply,
                    work_mass=float(hit_weights[index])/len(readings), **compact_forecast(outcome)))
        positive_mean += float(hit_weights[index])*sum(values)/len(values)
    negative_mass = max(0., 1-probability) if negative is not None else 0.
    expected = positive_mean+(negative_mass*negative['mean_s'] if negative is not None else 0.)
    risk = max(branch_worst)
    immediate = float(np.linalg.norm(q-start))/5+5+int(channel != current_channel)
    return dict(q=q, score=immediate+MEAN_WEIGHT*expected+RISK_WEIGHT*risk,
        immediate_s=immediate, expected_tail_s=expected, risk_tail_s=risk,
        signal_mass=probability, sampled_signal_mass=raw_probability, scenarios=len(scenarios),
        receipt_certificate=guarantee, negative_in_risk=negative is not None,
        negative_work_mass=negative_mass,
        negative_forecast=compact_forecast(negative) if negative is not None else None,
        positive_branches=branches)


def candidate_points(belief, start):
    """The unchanged ten geometric points in localization.choose_measure."""
    center, r = core.mec(belief.P)
    _, (i, j) = core.diameter(belief.P)
    u = belief.P[j]-belief.P[i]
    u = u/np.linalg.norm(u) if np.linalg.norm(u) > 1e-9 else np.array([1., 0.])
    v = np.array([-u[1], u[0]])
    candidates = [center, (center+start)/2]
    for along in (0., -.35*r):
        for offset in (-100., -40., 40., 100.):
            candidates.append(center+along*u+offset*v)
    return candidates


def choose_feedback_action(belief, start, channel, current_channel, plan, trace):
    prepared = belief.scenarios()
    optical = forecast_clear(belief, start, plan=plan)
    best, candidates = None, []
    for q in candidate_points(belief, start):
        value = score_measure(belief, start, q, channel, current_channel, prepared=prepared)
        if value is None:
            continue
        candidates.append({k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in value.items()})
        if best is None or value['score'] < best['score']:
            best = value
    selected = best is not None and best['score'] < optical['score']
    trace.append(dict(phase='feedback_cost_prediction', model=MODEL_ID, channel=channel,
        current_optical=compact_forecast(optical), candidates=candidates,
        selected='measure' if selected else 'optical',
        selected_q=best['q'].tolist() if selected else None,
        mean_weight=MEAN_WEIGHT, risk_weight=RISK_WEIGHT))
    return best, optical['score']


def make_solver():
    source = Path(baseline.__file__).read_bytes()
    require(hashlib.sha256(source).hexdigest() == BASELINE_SHA256, 'baseline changed')
    tree = ast.parse(source.decode('utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    untouched = {n.name: ast.dump(n) for n in function.body
                 if isinstance(n, ast.FunctionDef) and n.name != 'service'}
    service = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    changed = 0
    for index, node in enumerate(service.body):
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'action':
            service.body[index:index+1] = ast.parse('''
if local_steps[c] < max_active_measures:
    action, decision_optical_score = choose_feedback_action(
        belief, np.asarray(api.position), c, api.channel, plan, trace)
else:
    action, decision_optical_score = None, plan['score']
''').body
            changed += 1
            break
    for node in ast.walk(service):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "action is not None and action['score'] < plan['score']":
            node.test = ast.parse("action is not None and action['score'] < decision_optical_score", mode='eval').body
            changed += 1
        if isinstance(node, ast.keyword) and node.arg == 'optical_s' and ast.unparse(node.value) == "plan['score']":
            node.value = ast.Name(id='decision_optical_score', ctx=ast.Load())
            changed += 1
    require(changed == 3, 'service edit scope changed')
    require(untouched == {n.name:ast.dump(n) for n in function.body
                         if isinstance(n, ast.FunctionDef) and n.name != 'service'}, 'other functions changed')
    namespace = baseline.solve_multi.__globals__.copy()
    namespace['choose_feedback_action'] = choose_feedback_action
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


solve_multi = make_solver()
