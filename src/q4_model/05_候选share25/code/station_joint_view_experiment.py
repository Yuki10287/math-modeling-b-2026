"""Local-only candidate: an unvisited scan station may be the next source view.

The original service decision remains the fallback. Compare it with a joint
view/scan using the same frozen remaining task order, after forecasting the
selected source's optical completion and its exit point. The existing finite
position/type/heading quadrature is a heuristic, not an official probability
distribution. No hidden source, future response, or elapsed time is read.

Only a private copy of the frozen solver function is transformed. Production
modules, the 25 stations, six-view limit, observations, and optical proofs are
unchanged. This module has no official-client entry point.
"""
import ast
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

BASELINE_SHA256 = '2854b5d7e38ec5a56f9eb02bdbb126bb71b9c494030d0b3ae7e803e344d0f92a'
MODEL_ID = 'station_joint_view_v1'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def continuation_cost(points, fees, exits):
    """Fixed open route from each possible service exit; no return to origin."""
    points = np.asarray(points, float).reshape((-1, 2))
    exits = np.asarray(exits, float).reshape((-1, 2))
    require(len(points) == len(fees), 'route points and fee count differ')
    if not len(points):
        return np.zeros(len(exits))
    internal = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())/5
    return np.linalg.norm(exits-points[0], axis=1)/5+internal+sum(fees)


def optical_forecast(P, start, remaining_points, remaining_fees, quadrature=None, plan=None):
    """Preserve the original .8 mean/.2 terminal mixture also for route exits."""
    gs = samples(P) if quadrature is None else quadrature
    plan = optical_plan(P, start, quadrature) if plan is None else plan
    path = np.asarray(plan['path'])
    hits = np.linalg.norm(np.asarray(gs)[:, None, :]-path, axis=2) <= 20
    require(hits.any(axis=1).all(), 'forecast optical cover misses quadrature point')
    exits = path[hits.argmax(axis=1)]
    future = .8*float(continuation_cost(remaining_points, remaining_fees, exits).mean())
    future += .2*float(continuation_cost(remaining_points, remaining_fees, path[-1:])[0])
    return dict(local_s=float(plan['score']), continuation_s=future)


def measure_forecast(belief, start, q, channel, current_channel,
                     remaining_points, remaining_fees, prepared):
    """The original mixture score, extended beyond each predicted clear point."""
    gs, scenarios = prepared
    if not scenarios or any(np.linalg.norm(q-p) <= .1 for p in belief.measured):
        return None
    hit_weights = np.zeros(len(gs))
    total_weight = sum(row[4] for row in scenarios)
    for index, g, heading, radius, weight in scenarios:
        if np.linalg.norm(q-g) <= radius and (heading is None or (q-g)@heading >= 0):
            hit_weights[index] += weight/total_weight
    probability = float(hit_weights.sum())
    if probability < .05:
        return None
    negative = optical_forecast(belief.P, q, remaining_points, remaining_fees, gs)
    local = (1-probability)*negative['local_s']
    future = (1-probability)*negative['continuation_s']
    for index in np.flatnonzero(hit_weights):
        g = gs[index]
        if np.linalg.norm(g-q) <= 5:
            local += hit_weights[index]*5
            future += hit_weights[index]*float(continuation_cost(remaining_points, remaining_fees, [q])[0])
            continue
        angle = math.degrees(math.atan2(g[1]-q[1], g[0]-q[0])) % 360
        branches = []
        for error in (-1., 0., 1.):
            bearing = round((angle+error) % 360, 2) % 360
            P = core.disk_clip(core.wedge(belief.P, q, bearing), q)
            require(bool(len(P)), 'empty predictive branch')
            branches.append(optical_forecast(P, q, remaining_points, remaining_fees))
        local += hit_weights[index]*sum(b['local_s'] for b in branches)/3
        future += hit_weights[index]*sum(b['continuation_s'] for b in branches)/3
    local += float(np.linalg.norm(q-start))/5+5+int(channel != current_channel)
    return dict(local_s=float(local), continuation_s=float(future),
                score=float(local+future), signal_mass=probability, scenarios=len(scenarios))


def select_joint_measure(belief, start, channel, current_channel, original_action,
                         original_plan, remaining_tasks, remaining_points,
                         unknown_channels, cover, trace):
    """Consider only original pending stations; do not change cover from predictions.

    Future scan fees freeze currently unresolved channels: 6 seconds each,
    charged once per remaining station. The joint station's same fee moves to
    the immediate stage. This ignores future discoveries and sharing benefits;
    it is not an exact forecast of future scan fees, nor an extra reward.
    """
    if not unknown_channels or not any(t[0] == 'scan' for t in remaining_tasks):
        return None
    prepared = belief.scenarios()
    if not prepared[1]:
        return None
    points = np.asarray(remaining_points, float).reshape((-1, 2))
    require(len(points) == len(remaining_tasks), 'remaining task coordinates differ')
    fees = []
    for task, point in zip(remaining_tasks, points):
        required = [] if task[0] != 'scan' else [
            c for c in unknown_channels
            if not any(np.linalg.norm(point-p) < 1e-7 for p in cover.negative[c])]
        fees.append(6*len(required))
    # Preserve the original service decision; a new forecast cannot switch
    # between old actions unless an actual joint station replaces that decision.
    measuring = original_action is not None and original_action['score'] < original_plan['score']
    if measuring:
        original = measure_forecast(belief, start, original_action['q'], channel,
                                    current_channel, points, fees, prepared)
        require(original is not None, 'original view missing from forecast')
        expected_local = original_action['score']
    else:
        original = optical_forecast(belief.P, start, points, fees, plan=original_plan)
        original['score'] = original['local_s']+original['continuation_s']
        expected_local = original_plan['score']
    require(abs(original['local_s']-expected_local) <= 1e-7,
            'extended forecast changed the frozen local score')
    candidates, best = [], None
    for index, (task, q) in enumerate(zip(remaining_tasks, points)):
        if task[0] != 'scan' or not fees[index]:
            continue
        mask = [j for j in range(len(points)) if j != index]
        value = measure_forecast(belief, start, q, channel, current_channel,
                                 points[mask], [fees[j] for j in mask], prepared)
        if value is None:
            continue
        value.update(station=int(task[1]), q=q.copy(), scan_fee_s=fees[index])
        value['score'] += fees[index]
        candidates.append({k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in value.items()})
        if value['score'] < original['score']-1e-7 and (best is None or value['score'] < best['score']-1e-7):
            best = value
    original_prune_diagnostic = None
    if best is not None and not measuring:
        proof = baseline.prune_path(belief.P, belief.positives, belief.negatives, original_plan['path'])
        original_prune_diagnostic = dict(original_points=len(original_plan['path']),
            removable_points=len(proof['removed_indices']), kept_points=len(proof['kept_indices']))
    trace.append(dict(phase='station_joint_prediction', model=MODEL_ID, channel=channel,
        original_action='measure' if measuring else 'optical', original=original,
        local_cost_surrogate='Original unpruned rectangle, as in the frozen service decision; not exact pruned execution cost.',
        original_prune_diagnostic=original_prune_diagnostic,
        unknown_channels=list(unknown_channels), frozen_tasks=[list(t) for t in remaining_tasks],
        frozen_points=points.tolist(), frozen_scan_fees_s=fees, candidates=candidates,
        selected=None if best is None else best['station']))
    return best


def make_solver():
    source = Path(baseline.__file__).read_bytes()
    require(hashlib.sha256(source).hexdigest() == BASELINE_SHA256, 'baseline changed')
    tree = ast.parse(source.decode('utf-8-sig'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    preserved = {n.name: ast.dump(n) for n in function.body if isinstance(n, ast.FunctionDef) and n.name != 'service'}
    service = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    joint_statements = ast.parse('''
joint = (select_joint_measure(
    belief, np.asarray(api.position), c, api.channel, action, plan,
    [tasks[k] for k in order[1:]], [points[k] for k in order[1:]],
    unknown(), cover, trace) if local_steps[c] < max_active_measures else None)
if joint is not None:
    measure(joint['q'], c, 'joint_source_measure')
    local_steps[c] += 1
    if not scan(np.asarray(api.position).copy()):
        return 'failed'
    return 'joint_measured'
''').body
    insertions = 0
    for index, statement in enumerate(service.body):
        if isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == 'action':
            service.body[index+1:index+1] = joint_statements
            insertions += 1
            break
    require(insertions == 1, 'service insertion point changed')
    changes = 0
    for node in ast.walk(function):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "variant == 'share25' and (not share_at_stop(set()))":
            node.test = ast.parse("variant == 'share25' and status != 'joint_measured' and not share_at_stop(set())", mode='eval').body
            changes += 1
    require(changes == 1, 'post-service sharing condition changed')
    require(preserved == {n.name: ast.dump(n) for n in function.body
                         if isinstance(n, ast.FunctionDef) and n.name != 'service'}, 'other local functions changed')
    namespace = baseline.solve_multi.__globals__.copy()
    namespace['select_joint_measure'] = select_joint_measure
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(Path(__file__)), 'exec'), namespace)
    return namespace['solve_multi']


solve_multi = make_solver()
