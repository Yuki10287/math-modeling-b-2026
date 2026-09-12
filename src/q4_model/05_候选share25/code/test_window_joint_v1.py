"""Small local structural/state tests, never a complete world or official run.

Controlled public-feedback transitions test task replanning and shared budget
accounting. One real copied macro tests cloning/clear fees; a handcrafted work
hypothesis tests fixed feedback. These are mechanism checks, not performance or
physical complete-game evidence. Production/research model files stay unchanged.
"""
import argparse
import ast
import builtins
import copy
import hashlib
import inspect
import json
import math
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(HERE))
import window_joint_planner_v1 as model
import window_worlds_v1 as worlds


def captured(compiler):
    output = []
    def capture(tree, *args, **kwargs):
        output.append(copy.deepcopy(tree))
        return builtins.compile(tree, *args, **kwargs)
    with patch.object(model, 'compile', capture, create=True):
        compiler()
    if len(output) != 1:
        raise ValueError('expected one private compilation')
    return output[0].body[0]


def strip_ledger(function):
    count = 0
    for child in function.body:
        if isinstance(child, ast.FunctionDef) and child.name in ('measure', 'clear'):
            original = len(child.body)
            child.body = [n for n in child.body if not (
                isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and ast.unparse(n.value.func) in ('record_measure', 'record_clear'))]
            count += original-len(child.body)
    if count != 2:
        raise ValueError('ledger edit scope changed')


class Cover:
    """Small deterministic task map, not a directional-cover certificate."""
    def __init__(self):
        self.stations = np.array([[0., 2000.], [2000., 0.], [-2000., 0.]])
        self.indices = np.empty((0, 3), int)
        self.triangles = np.empty((0, 3, 2))
        self.spacing = 0.
        self.negative = {c: [] for c in range(1, 21)}
        self.covered = {c: np.zeros(0, bool) for c in range(1, 21)}
        self.witnesses = {c: {} for c in range(1, 21)}

    def complete(self, c):
        return False

    def needed_stations(self, channels):
        return list(range(3)) if channels else []

    def observe_negative(self, c, q):
        self.negative[c].append(np.asarray(q).copy())


def state_fixture():
    a = model.baseline.Belief(np.array([-1000., 0.]), {'measure_result': 'direction', 'svd_deg': 0.})
    b = model.baseline.Belief(np.array([0., 150.]), {'measure_result': 'near'})
    cover = Cover()
    cover.observe_negative(20, [0., -100.])
    ledger = [dict(action='measure', position=[-1000., 0.], channel=1, measure_result='direction', svd_deg=0.),
              dict(action='measure', position=[0., 150.], channel=2, measure_result='near'),
              dict(action='measure', position=[0., -100.], channel=20, measure_result='no_signal')]
    return model.PublicState(np.array([0., -100.]), 20, {1: a, 2: b}, {1: 0, 2: 0},
                             cover, {1, 2}, set(), ledger)


def full_state_snapshot(state):
    # Cover fixed geometry is checked in addition to model.public_digest.
    return json.dumps(dict(public=model.public_payload(state), all_cover_fields=vars(state.cover)),
                      sort_keys=True, default=model.json_value, allow_nan=False)


def fake_worlds():
    return [dict(reply_kind='direction' if i % 2 else 'no_signal', world_identity=i,
                 hidden_metadata={'arbitrary_position': [9999+i, -8888]}, weight=1/6) for i in range(6)]


def fake_transition(state, task, world, budget):
    """Controlled transition returns only public state to the next-task policy."""
    budget.transition()
    budget.api_call('measure')
    after = copy.deepcopy(state)
    if task == ('source', 1):
        feedback = {'measure_result': world['reply_kind']}
        if world['reply_kind'] == 'direction':
            feedback['svd_deg'] = 45.
        after.beliefs[1].update(np.array([0., -100.]), feedback)
        after.local_steps[1] += 1
        after.channel = 1
        model.record_measure(after.ledger, after.position, 1, feedback)
    else:
        # Distinct public history with an unchanged route proxy.
        after.cover.observe_negative(20, after.position)
        model.record_measure(after.ledger, after.position, 20, {'measure_result': 'no_signal'})
    cost = 1000. if task == ('source', 2) else 1.
    return after, cost, dict(events=[], optical_blocks=0, controlled_feedback_only=True)


def sampler(*args, **kwargs):
    return dict(worlds=fake_worlds(), diagnostics={'controlled_unit_input': True})


class WindowChecks(unittest.TestCase):
    def test_ast_only_ledger_and_first_task_hook(self):
        old = model._source_function()
        macro, solver = captured(model._compile_macro), captured(model._compile_solver)
        strip_ledger(macro)
        strip_ledger(solver)
        old_nested = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef)}
        for version in (macro, solver):
            self.assertEqual({n.name: ast.dump(n) for n in version.body if isinstance(n, ast.FunctionDef)}, old_nested)
        ledger = [i for i, n in enumerate(solver.body) if isinstance(n, ast.Assign)
                  and ast.unparse(n.targets[0]) == '(planner_ledger, planner_calls)']
        self.assertEqual(len(ledger), 1)
        del solver.body[ledger[0]]
        loop = next(n for n in solver.body if isinstance(n, ast.For))
        hook_indices = [i for i, n in enumerate(loop.body) if isinstance(n, ast.If)
                        and ast.unparse(n.test) == 'planner_calls < MAX_WINDOW_DECISIONS and beliefs and pending']
        self.assertEqual(len(hook_indices), 1)
        hook = loop.body.pop(hook_indices[0])
        self.assertEqual(sum(isinstance(n, ast.Call) and ast.unparse(n.func) == 'plan_window' for n in ast.walk(hook)), 1)
        self.assertEqual(ast.unparse(hook.body[-1]), 'if selected is not None:\n    task, index = selected')
        self.assertEqual(solver.args.args.pop().arg, 'rollout_depth')
        self.assertEqual(solver.args.defaults.pop().value, 2)
        self.assertEqual(ast.dump(solver), ast.dump(old))
        self.assertEqual(model.MAX_WINDOW_DECISIONS, 4)
        self.assertNotIn('plan_window', model.baseline.solve_multi.__globals__)
        # This exact comparison includes full optical proof+path and unchanged
        # single-task execution. Predicted second actions are never dispatched.

    def test_feedback_changes_public_successor_and_scope(self):
        state = state_fixture()
        roots = model.window_tasks(state)
        self.assertIn(model.baseline_task(state), roots)
        self.assertLessEqual(sum(t[0] == 'source' for t in roots), 2)
        self.assertLessEqual(sum(t[0] == 'scan' for t in roots), 3)
        negative, positive = copy.deepcopy(state), copy.deepcopy(state)
        negative.beliefs[1].update(state.position, {'measure_result': 'no_signal'})
        positive.beliefs[1].update(state.position, {'measure_result': 'direction', 'svd_deg': 45.})
        self.assertEqual(model.baseline_task(negative, roots), ('source', 2))
        self.assertEqual(model.baseline_task(positive, roots), ('source', 1))
        self.assertNotEqual(model.public_digest(negative), model.public_digest(positive))
        self.assertEqual(list(inspect.signature(model.baseline_task).parameters), ['state', 'fixed_tasks'])

    def test_new_discovery_does_not_add_fixed_terminal_debt(self):
        before = state_fixture()
        roots = model.window_tasks(before)
        after = copy.deepcopy(before)
        after.beliefs[3] = model.baseline.Belief(after.position, {'measure_result': 'near'})
        after.discovered.add(3)
        after.local_steps[3] = 0
        first, second = model.terminal_value(before, roots), model.terminal_value(after, roots)
        self.assertEqual(first['service_s'], second['service_s'])
        self.assertEqual(first['travel_s'], second['travel_s'])
        self.assertLessEqual(second['total_s'], first['total_s']+1e-9)
        self.assertNotIn(('source', 3), model.tasks_and_points(after, roots)[0])
        self.assertIn(model.baseline_task(after, roots), roots)
        self.assertGreater(model.terminal_value(after)['service_s'], model.terminal_value(before)['service_s'])

    def test_same_roots_worlds_caps_and_public_only_successor(self):
        state = state_fixture()
        before = full_state_snapshot(state)
        with patch.object(model, 'sample_worlds', sampler), patch.object(model, 'transition', fake_transition):
            _, one = model.plan_window(state, depth=1)
            _, two = model.plan_window(state, depth=2)
        self.assertEqual(one['reason'], 'complete_root_comparison')
        self.assertEqual(two['reason'], 'complete_root_comparison')
        self.assertEqual(one['root_tasks'], two['root_tasks'])
        self.assertEqual(one['world_digest'], two['world_digest'])
        self.assertEqual(one['budget']['max_transitions'], two['budget']['max_transitions'])
        self.assertEqual((one['budget']['used_transitions'], two['budget']['used_transitions']), (30, 60))
        self.assertEqual((one['budget']['predicted_measure'], two['budget']['predicted_measure']), (30, 60))
        roots, mapping = set(map(tuple, two['root_tasks'])), {}
        for row in two['rows']:
            for item in row['samples']:
                self.assertIn(tuple(item['second_task']), roots)
                mapping.setdefault(item['first_public_digest'], set()).add(tuple(item['second_task']))
        self.assertTrue(all(len(tasks) == 1 for tasks in mapping.values()))
        self.assertEqual(full_state_snapshot(state), before)
        changed_worlds = fake_worlds()[::-1]
        for world in changed_worlds:
            world['hidden_metadata'] = {'changed_private_metadata': True}
            world['world_identity'] += 100
        with patch.object(model, 'sample_worlds', return_value=dict(worlds=changed_worlds, diagnostics={})), \
                patch.object(model, 'transition', fake_transition):
            _, changed = model.plan_window(state, depth=2)
        for row in changed['rows']:
            for item in row['samples']:
                self.assertEqual(mapping[item['first_public_digest']], {tuple(item['second_task'])})

    def test_budget_and_conflict_fallback_never_use_partial_minimum(self):
        state = state_fixture()
        original = model.baseline_task(state)
        before = full_state_snapshot(state)
        with patch.object(model, 'sample_worlds', sampler), patch.object(model, 'transition', fake_transition):
            chosen, low = model.plan_window(state, depth=2, transition_cap=59)
        self.assertEqual(chosen, original)
        self.assertEqual(low['budget']['used_transitions'], 0)
        self.assertEqual(low['rows'], [])
        budget_class = model.PredictionBudget
        with patch.object(model, 'sample_worlds', sampler), patch.object(model, 'transition', fake_transition), \
                patch.object(model, 'PredictionBudget', side_effect=lambda cap: budget_class(cap=cap, api_cap=13)):
            chosen, partial = model.plan_window(state, depth=1)
        self.assertEqual(chosen, original)
        self.assertFalse(partial['applied'])
        self.assertEqual(partial['reason'], 'prediction_rejected_original_policy')
        self.assertEqual(len(partial['rows']), 2)
        self.assertNotEqual(tuple(min(partial['rows'], key=lambda row: row['mean_s'])['task']), original)
        self.assertEqual(partial['budget']['predicted_measure'], 13)
        with patch.object(model, 'sample_worlds', sampler), \
                patch.object(model, 'transition', side_effect=RuntimeError('controlled predictive conflict')):
            chosen, conflict = model.plan_window(state, depth=2)
        self.assertEqual(chosen, original)
        self.assertFalse(conflict['applied'])
        self.assertEqual(full_state_snapshot(state), before)

    def test_real_macro_clone_clear_fee_and_sibling_isolation(self):
        state = state_fixture()
        state.beliefs = {1: model.baseline.Belief(np.zeros(2), {'measure_result': 'near'})}
        state.position, state.channel = np.array([30., 40.]), 7
        state.discovered, state.local_steps = {1}, {1: 0}
        state.ledger = [dict(action='measure', position=[0., 0.], channel=1, measure_result='near')]
        world = dict(sources={'1': dict(position=[0., 0.], radius=1000., type='omni', heading=None)},
                     base_cleared=[], historical_measure_cache={}, error_salt='fixed-unit-world')
        before, original_world = full_state_snapshot(state), copy.deepcopy(world)
        a, ca, ea = model.transition(state, ('source', 1), world, model.PredictionBudget())
        b, cb, eb = model.transition(state, ('source', 1), world, model.PredictionBudget())
        self.assertEqual(ca, math.hypot(30., 40.)/5+5)
        self.assertEqual((ca, ea), (cb, eb))
        self.assertEqual(a.channel, 7)
        self.assertEqual(a.cleared, {1})
        self.assertNotIn(1, a.beliefs)
        self.assertEqual(a.ledger[-1]['clear_result'], 'success')
        a.cover.negative[1].append(np.array([888., 777.]))
        a.cover.stations[0] += 1
        self.assertEqual(full_state_snapshot(state), before)
        self.assertFalse(any(np.array_equal(p, [888., 777.]) for p in b.cover.negative[1]))
        np.testing.assert_array_equal(b.cover.stations, state.cover.stations)
        self.assertEqual(world, original_world)

    def test_fixed_feedback_and_recorded_reading_consistency(self):
        q, old_q = [500., 100.], [500., 0.]
        cache_key = '1:'+float(old_q[0]).hex()+':'+float(old_q[1]).hex()
        world = dict(sources={'1': dict(position=[0., 0.], radius=1000., type='omni', heading=None)},
                     base_cleared=[], historical_measure_cache={cache_key: {'measure_result': 'direction', 'svd_deg': 180.}},
                     error_salt='fixed-unit-error')
        original = copy.deepcopy(world)
        first = worlds.predict_measure(world, q, 1)
        worlds.predict_measure(world, [250., 250.], 1)
        worlds.predict_clear(world, [100., 0.], 1)
        self.assertEqual(first, worlds.predict_measure(world, q, 1))
        self.assertEqual(worlds.predict_measure(world, old_q, 1), world['historical_measure_cache'][cache_key])
        self.assertEqual(worlds.predict_measure(world, old_q, 1, cleared={1}), {'measure_result': 'no_signal'})
        self.assertEqual(world, original)


def binding(path):
    path = path.resolve()
    return dict(path=path.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE.parent/'results/window_joint_v1_checks.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('refuse to overwrite existing check evidence')
    paths = [Path(__file__), Path(model.__file__), Path(worlds.__file__), Path(model.baseline.__file__),
             ROOT/'src/q4_model/localization.py', ROOT/'src/q4_model/route_planning.py', ROOT/'src/q3_model_v2/geometry.py']
    hashes = [binding(path) for path in paths]
    started = time.perf_counter()
    with patch.object(socket, 'create_connection', side_effect=AssertionError('network forbidden in local unit check')):
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WindowChecks))
    stable = hashes == [binding(path) for path in paths]
    report = dict(passed=result.wasSuccessful() and stable, tests=result.testsRun, runtime_s=time.perf_counter()-started,
        failures=[dict(test=str(t), traceback=tb) for t, tb in result.failures],
        errors=[dict(test=str(t), traceback=tb) for t, tb in result.errors],
        source_bindings=hashes, source_stable=stable,
        scope='Seven bounded synthetic mechanism checks. No full game or simulator connection. Controlled transitions test planning control flow; one real macro tests state/fee semantics.',
        limitations=['Budget comparison uses the same cap, not identical consumed work: depth1=30, depth2=60 controlled macro transitions.',
                     'Fixed-error checks use a handcrafted hypothesis; the complete world sampler is tested independently elsewhere.',
                     'Local fixed-task terminal is a heuristic. This check does not claim complete exploration value or global two-step optimality.'])
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=str(args.out), passed=report['passed'], tests=result.testsRun), ensure_ascii=False))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
