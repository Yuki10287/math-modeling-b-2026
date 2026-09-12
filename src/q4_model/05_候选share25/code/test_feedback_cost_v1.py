"""Independent bounded checks of the frozen local feedback-cost experiment.

No full game is solved. Saved optical blocks supply observed geometry; small
legal work-scenario mixtures exercise support handling and private updates.
"""
import ast
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import socket
import time
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RESULTS = HERE.parent / 'results'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def binding(path):
    return dict(path=path.relative_to(ROOT).as_posix(),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def close(a, b):
    require(math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-8), f'Costs differ: {a}, {b}')


def snapshot(belief):
    return json.dumps(belief.__dict__, sort_keys=True,
                      default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value)


def structure_check(model):
    original = ast.parse(Path(model.baseline.__file__).read_text(encoding='utf-8-sig'))
    original = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')
    captured = []
    fix = ast.fix_missing_locations
    def capture(tree):
        captured.append(copy.deepcopy(tree))
        return fix(tree)
    with mock.patch.object(model.ast, 'fix_missing_locations', capture):
        model.make_solver()
    changed = captured[-1].body[0]
    old_service = next(n for n in original.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    new_service = next(n for n in changed.body if isinstance(n, ast.FunctionDef) and n.name == 'service')
    old_assignment = next(n for n in old_service.body if isinstance(n, ast.Assign)
                          and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'action')
    normalized = 0
    for i, node in enumerate(new_service.body):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'local_steps[c] < max_active_measures':
            require(ast.unparse(node.body[0].value.func) == 'choose_feedback_action', 'Unexpected scorer')
            require(ast.unparse(node.orelse[0]) == "action, decision_optical_score = (None, plan['score'])",
                    'Lifetime-limit fallback changed')
            new_service.body[i] = copy.deepcopy(old_assignment)
            normalized += 1
    for node in ast.walk(new_service):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "action is not None and action['score'] < decision_optical_score":
            node.test = ast.parse("action is not None and action['score'] < plan['score']", mode='eval').body
            normalized += 1
        if isinstance(node, ast.keyword) and node.arg == 'optical_s' and ast.unparse(node.value) == 'decision_optical_score':
            node.value = ast.parse("plan['score']", mode='eval').body
            normalized += 1
    require(normalized == 3, 'Unexpected number of service differences')
    require(ast.dump(changed) == ast.dump(original), 'Other solve_multi behavior changed')
    old = ast.parse(inspect.getsource(model.baseline.choose_measure)).body[0].body
    start = next(i for i, n in enumerate(old) if isinstance(n, ast.Assign)
                 and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'core.mec')
    end = next(i for i, n in enumerate(old[start:], start) if isinstance(n, ast.Assign)
               and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'best')
    new = ast.parse(inspect.getsource(model.candidate_points)).body[0].body[1:-1]
    require([ast.dump(n) for n in old[start:end]] == [ast.dump(n) for n in new], 'Ten-point construction differs')
    return dict(passed=True, service_changes=normalized,
                entire_solver_ast_equal_after_only_three_normalizations=True,
                ten_point_pool_construction_ast_identical=True)


def small_belief(model, wide=False):
    belief = model.baseline.Belief.__new__(model.baseline.Belief)
    belief.P = np.array([[-50., -2.], [250. if wide else 50., -2.],
                         [250. if wide else 50., 2.], [-50., 2.]])
    belief.positives = [np.array([500., 1.]), np.array([500., -1.])]
    belief.negatives, belief.failed_clears = [], []
    belief.measured = [p.copy() for p in belief.positives]
    belief.near = None
    return belief


def scoring_checks(model):
    records = []
    for name, q, positions, wide in (
        ('sample_mass_one_unproved', [500., 2.], [[0., 0.]], False),
        ('exact_receipt_certificate', [500., 0.], [[0., 0.]], False),
        ('mixed_half_signal', [1100., 0.], [[0., 0.], [200., 0.]], True)):
        belief = small_belief(model, wide)
        before = snapshot(belief)
        gs = np.array(positions)
        prepared = (gs, [(i, g, None, 1000., 1.) for i, g in enumerate(gs)])
        value = model.score_measure(belief, belief.measured[0], q, 7, 3, prepared=prepared)
        require(value is not None, 'Unexpected filtered test point')
        require(snapshot(belief) == before, 'Hypothetical update leaked into actual belief')
        expected = sum(b['work_mass']*b['mean_s'] for b in value['positive_branches'])
        worst = [b['worst_s'] for b in value['positive_branches']]
        if value['negative_in_risk']:
            expected += value['negative_work_mass']*value['negative_forecast']['mean_s']
            worst.append(value['negative_forecast']['worst_s'])
        immediate = math.dist(belief.measured[0], q)/5+5+1
        close(expected, value['expected_tail_s'])
        close(max(worst), value['risk_tail_s'])
        close(immediate+.8*expected+.2*max(worst), value['score'])
        if name == 'sample_mass_one_unproved':
            require(value['signal_mass'] == 1 and value['negative_work_mass'] == 0,
                    'Test did not reach sampling-missed-negative case')
            require(not value['receipt_certificate']['certified'] and value['negative_in_risk'],
                    'Unproved negative was excluded from risk')
        elif name == 'exact_receipt_certificate':
            require(value['receipt_certificate']['certified'] and not value['negative_in_risk'],
                    'Receipt certificate not respected')
        else:
            close(value['signal_mass'], .5)
        records.append(dict(name=name, passed=True, state_unchanged=True,
                            score=value['score'], expected_tail_s=expected, risk_tail_s=max(worst),
                            signal_mass=value['signal_mass'], negative_in_risk=value['negative_in_risk'],
                            negative_work_mass=value['negative_work_mass']))
    return records


def optical_checks(model):
    checks, inputs = [], []
    for folder, name, channel in (
        ('station_joint_view_dev54000_v1', '54100-boundary-hashed-baseline_share25_prune', 14),
        ('station_joint_view_dev54000_v1', '54100-boundary-hashed-station_joint_view', 14),
        ('station_joint_view_holdout55000_v1', '55101-boundary-hashed-station_joint_view', 11)):
        path = RESULTS / folder / (name+'.json')
        inputs.append(binding(path))
        case = json.loads(path.read_text(encoding='utf-8'))
        row = next(t for t in case['trace'] if t.get('phase') == 'ordered_optical_prune' and t['channel'] == channel)
        belief = model.baseline.Belief.__new__(model.baseline.Belief)
        belief.P = np.array(row['original_polygon'])
        belief.positives = [np.array(p) for p in row['positives']]
        belief.negatives = [np.array(p) for p in row['negatives']]
        belief.measured, belief.failed_clears, belief.near = [], [], None
        start = case['events'][row['event']-1]['position'] if row['event'] else [0., 0.]
        plan = dict(path=np.array(row['original_path']))
        before = snapshot(belief)
        forecast = model.forecast_clear(belief, start, plan)
        require(snapshot(belief) == before, 'Forecast altered real belief')
        require(forecast['kept_indices'] == row['kept_indices'], 'Forecast and actual pruning differ')
        kept = [row['original_path'][i] for i in row['kept_indices']]
        cumulative, travelled, last = [], 0., start
        for point in kept:
            travelled += math.dist(last, point)/5
            cumulative.append(travelled)
            last = point
        full = cumulative[-1]+3*(len(kept)-1)+5
        close(full, forecast['worst_s'])
        actual = [t for t in case['trace'] if t.get('reason') == 'optical_cover' and t['channel'] == channel]
        require([a['position'] for a in actual] == kept[:len(actual)], 'Actual optical prefix differs')
        prefix = cumulative[len(actual)-1]+3*(len(actual)-1)+5
        observed = case['events'][actual[-1]['event']-1]['time_s']-case['events'][row['event']-1]['time_s']
        close(prefix, observed)
        # Independently sum distances and action fees at each retained sample.
        quadrature = [g for g in model.samples(np.array(forecast['conditional_polygon']))
                      if model.baseline.distance_squared(g, model.baseline.exact_hull(belief.P)) == 0]
        first_indices = [next((i for i,p in enumerate(kept) if math.dist(g,p) <= 20), None)
                         for g in quadrature]
        mean = full if not first_indices or any(i is None for i in first_indices) else sum(
            cumulative[i]+3*i+5 for i in first_indices)/len(first_indices)
        close(mean, forecast['mean_s'])
        close(.8*mean+.2*full, forecast['score'])
        # A deliberately empty retained quadrature must never create zero cost.
        with mock.patch.object(model, 'samples', lambda P: np.array([[1e9, 1e9]])):
            empty = model.forecast_clear(belief, start, plan)
        require(empty['mean_fallback_to_worst'] and empty['sample_count'] == 0,
                'Empty quadrature not recorded')
        close(empty['mean_s'], empty['worst_s'])
        checks.append(dict(input=path.relative_to(ROOT).as_posix(), channel=channel,
                           passed=True, forecast_matches_actual_prune=True,
                           full_kept_path_s=full, actual_success_prefix_s=prefix,
                           mean_s=mean, full_points=len(kept), actual_points=len(actual),
                           empty_quadrature_falls_back_to_worst=True))
    return checks, inputs


def main():
    begin = time.perf_counter()
    network_attempts = []
    def forbidden(*args, **kwargs):
        network_attempts.append(True)
        raise RuntimeError('No network permitted in these local checks')
    with mock.patch.object(socket.socket, 'connect', forbidden), \
         mock.patch.object(socket.socket, 'connect_ex', forbidden), \
         mock.patch.object(socket, 'create_connection', forbidden):
        import feedback_cost_experiment as model
        sources = [binding(HERE / 'feedback_cost_experiment.py'),
                   binding(HERE / 'receive_guarantee_v1.py'), binding(Path(model.baseline.__file__)),
                   binding(Path(__file__))]
        structure = structure_check(model)
        scoring = scoring_checks(model)
        optical, inputs = optical_checks(model)
    require(all(binding(ROOT / item['path']) == item for item in sources), 'A checked source changed')
    report = dict(passed=True, source_files=sources, input_files=inputs,
                  ast_structure=structure, branch_scoring=scoring, saved_optical_prefixes=optical,
                  network_guard=dict(attempts=len(network_attempts), blocked_calls=['connect', 'connect_ex', 'create_connection']),
                  no_full_game_run=True, elapsed_wall_s=time.perf_counter()-begin,
                  limits=['Finite positive quadrature remains a work model, not a continuous worst-case guarantee.',
                          'Mean position integration differs from an official posterior and ignores some nonconvex exclusions.',
                          'Local checks do not establish overall runtime improvement.'])
    output = RESULTS / 'feedback_cost_v1_independent_checks.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
