"""Local research route objective: travel plus expected discovery-delay fees.

Scoring prior only: N is uniform over the allowed integers 10..16; conditional
on N, channel subsets are uniform and source position/radius/type/heading are
independent, uniform in the target disk / [1000,1500] / two types / circle.
The fixed Sobol cloud never supplies absence evidence or an actual source count.
Only the global route call is replaced. Local sensing and optical logic stay
byte-for-byte identical at the function-AST level.
"""
import ast
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import qmc

Q4 = next(p for p in Path(__file__).resolve().parents if (p/'polar_cover.py').is_file())
sys.path.insert(0, str(Q4))
import task_sharing_solver as frozen
from route_planning import open_route, route_length

CLOUD_SIZE = 2048
CLOUD_SEED = 104729
START_TOURS = 4
TWO_OPT_SWEEPS = 6


@lru_cache(maxsize=1)
def cloud():
    z = qmc.Sobol(d=5, scramble=True, seed=CLOUD_SEED).random_base2(11)
    positions = 1800.*np.sqrt(z[:, 0])[:, None]*np.c_[np.cos(2*np.pi*z[:, 1]), np.sin(2*np.pi*z[:, 1])]
    radii = 1000.+500.*z[:, 2]
    omni = z[:, 3] < .5
    directions = np.c_[np.cos(2*np.pi*z[:, 4]), np.sin(2*np.pi*z[:, 4])]
    return positions, radii, omni, directions, hashlib.sha256(z.tobytes()).hexdigest()


@lru_cache(maxsize=4096)
def visible_mask(point):
    positions, radii, omni, directions, _ = cloud()
    delta = np.asarray(point)-positions
    visible = ((delta*delta).sum(axis=1) <= radii*radii) & (omni | ((delta*directions).sum(axis=1) >= 0))
    return int.from_bytes(np.packbits(visible, bitorder='little').tobytes(), 'little')


@lru_cache(maxsize=4096)
def surviving_cloud(negative_points):
    alive = (1 << CLOUD_SIZE)-1
    for point in negative_points:
        alive &= ~visible_mask(point)
    return alive


def expected_remaining(discovered, alive_count):
    """Finite total-count posterior after known labels and unknown negatives.

For a given total n, P(specified D labels are present) is C(n,D)/C(20,D).
The denominator cancels, leaving C(n,D)*w0**(n-D). Positive-observation
likelihoods of those known labels are common across n under this working prior.
"""
    assert 0 <= discovered <= 16 and 0 < alive_count <= CLOUD_SIZE
    totals = tuple(range(max(10, discovered), 17))
    logw = np.array([math.log(math.comb(n, discovered))+(n-discovered)*math.log(alive_count/CLOUD_SIZE) for n in totals])
    weights = np.exp(logw-logw.max())
    weights /= weights.sum()
    mean = float(sum((n-discovered)*weight for n, weight in zip(totals, weights)))
    assert -1e-10 <= mean <= 16-discovered+1e-10
    return mean, {str(n):float(w) for n,w in zip(totals, weights)}


def prefix_state(order, masks, initial_alive):
    states, costs = [initial_alive], [0]
    for node in order:
        mask = masks[node]
        alive, cost = states[-1], costs[-1]
        if mask is not None:
            cost += alive.bit_count()  # pay current station before discovering
            alive &= ~mask
        states.append(alive)
        costs.append(cost)
    return states, costs


def fee_route(points, start, tasks, unknown_channels, discovered_count, cover, trace):
    points, start = np.asarray(points, float), np.asarray(start, float)
    original = open_route(points, start)
    U = len(unknown_channels)
    if not U:
        return original
    assert U == 20-discovered_count
    ledgers = [tuple(tuple(float(x) for x in q) for q in cover.negative[c]) for c in unknown_channels]
    assert all(ledger == ledgers[0] for ledger in ledgers)
    alive = surviving_cloud(ledgers[0])
    m0 = alive.bit_count()
    if m0 == 0:
        trace.append(dict(phase='discovery_fee_route', applied=False,
            reason='scoring_cloud_empty_original_route_preserved', unknown=U, discovered=discovered_count))
        return original
    h, posterior = expected_remaining(discovered_count, m0)
    masks = []
    for node, (kind, index) in enumerate(tasks):
        if kind == 'scan':
            assert np.array_equal(points[node], cover.stations[index])
            masks.append(visible_mask(tuple(points[node])))
        else:
            assert kind == 'source'
            masks.append(None)
    scan_count = sum(mask is not None for mask in masks)
    if not scan_count:
        return original
    # At each scan: E[unknown] = U - h*(1 - surviving_after_prefix/m0).
    # Six seconds per still-unknown channel is the detection + switch proxy.
    fixed_fee = 6*(U-h)*scan_count
    coefficient = 6*h/m0
    distances = np.linalg.norm(points[:, None, :]-points[None, :, :], axis=2)
    initial = np.linalg.norm(points-start, axis=1)
    def full_score(order):
        length = route_length(points, start, order)
        _, counts = prefix_state(order, masks, alive)
        fee = fixed_fee+coefficient*counts[-1]
        return length/5+fee, length, fee
    original_score, original_length, original_fee = full_score(original)
    best = (original_score, original.copy())
    tours = [original.copy()]
    for first in np.argsort(initial, kind='stable'):
        order, remaining = [int(first)], set(range(len(points)))-{int(first)}
        while remaining:
            node = min(remaining, key=lambda j:(distances[order[-1], j], j))
            order.append(node)
            remaining.remove(node)
        if order not in tours:
            tours.append(order)
        if len(tours) == START_TOURS:
            break
    evaluations = 0
    for initial_order in tours:
        order = initial_order.copy()
        current_score, length, _ = full_score(order)
        for _ in range(TWO_OPT_SWEEPS):
            states, prefix = prefix_state(order, masks, alive)
            change, next_score, next_length = None, current_score, length
            for i in range(len(order)-1):
                for j in range(i+1, len(order)):
                    old_edge = initial[order[i]] if i == 0 else distances[order[i-1], order[i]]
                    new_edge = initial[order[j]] if i == 0 else distances[order[i-1], order[j]]
                    if j+1 < len(order):
                        old_edge += distances[order[j], order[j+1]]
                        new_edge += distances[order[i], order[j+1]]
                    proposed_length = length+new_edge-old_edge
                    remaining, reversed_cost = states[i], 0
                    for k in range(j, i-1, -1):
                        mask = masks[order[k]]
                        if mask is not None:
                            reversed_cost += remaining.bit_count()
                            remaining &= ~mask
                    # The same set of scans has been visited after the reversed
                    # segment, so suffix survival and its cost are unchanged.
                    assert remaining == states[j+1]
                    count = prefix[-1]-(prefix[j+1]-prefix[i])+reversed_cost
                    score = proposed_length/5+fixed_fee+coefficient*count
                    evaluations += 1
                    if score < next_score-1e-8:
                        change, next_score, next_length = (i,j), score, proposed_length
            if change is None:
                break
            i, j = change
            order[i:j+1] = order[i:j+1][::-1]
            current_score, length = next_score, next_length
        actual_score, _, _ = full_score(order)
        assert abs(actual_score-current_score) < 1e-6
        if actual_score < best[0]-1e-8:
            best = (actual_score, order.copy())
    score, length, fee = full_score(best[1])
    assert score <= original_score+1e-7
    trace.append(dict(phase='discovery_fee_route', applied=best[1] != original,
        unknown=U, discovered=discovered_count, surviving_hypotheses=m0,
        expected_remaining_sources=h, total_count_posterior=posterior,
        original_proxy_s=original_score, selected_proxy_s=score,
        original_travel_m=original_length, selected_travel_m=length,
        original_scan_fee_proxy_s=original_fee, selected_scan_fee_proxy_s=fee,
        original_order=original, selected_order=best[1],
        first_task_changed=best[1][0] != original[0], route_nodes=len(tasks),
        scan_nodes=scan_count, two_opt_evaluations=evaluations,
        cloud_is_scoring_only=True))
    return best[1]


_tree = ast.parse((Q4/'task_sharing_solver.py').read_text(encoding='utf-8-sig'))
_function = next(node for node in _tree.body if isinstance(node, ast.FunctionDef) and node.name == 'solve_multi')
_original_ast = ast.dump(_function, include_attributes=False)
_matches = [node for node in ast.walk(_function) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open_route']
assert len(_matches) == 1
_call = _matches[0]
_call.func.id = 'fee_route'
_call.args += [ast.Name(id='tasks', ctx=ast.Load()), ast.Call(func=ast.Name(id='unknown', ctx=ast.Load()), args=[], keywords=[]),
              ast.Call(func=ast.Name(id='len', ctx=ast.Load()), args=[ast.Name(id='discovered', ctx=ast.Load())], keywords=[]),
              ast.Name(id='cover', ctx=ast.Load()), ast.Name(id='trace', ctx=ast.Load())]
_namespace = dict(frozen.__dict__)
_namespace['fee_route'] = fee_route
exec(compile(ast.fix_missing_locations(ast.Module(body=[_function], type_ignores=[])), str(Path(__file__)), 'exec'), _namespace)
_solver = _namespace['solve_multi']
_call.func.id = 'open_route'
_call.args = _call.args[:2]
assert ast.dump(_function, include_attributes=False) == _original_ast


def solve_multi(api, variant='share25', trace=None, max_active_measures=6):
    return _solver(api, variant=variant, trace=trace, max_active_measures=max_active_measures)
