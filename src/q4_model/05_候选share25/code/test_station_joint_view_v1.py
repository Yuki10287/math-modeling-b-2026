"""Small local structural/model checks; no simulator or network connection.

The synthetic state machine tests control flow only. Its toy optical points
are not geometry evidence. Full geometric validation belongs to the paired
experiment's independent audits. Existing production modules are never edited.
"""
import argparse
import ast
import builtins
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import station_joint_view_experiment as model


def generated_ast():
    saved = []

    def capture(tree, *args, **kwargs):
        saved.append(copy.deepcopy(tree))
        return builtins.compile(tree, *args, **kwargs)

    with patch.object(model, 'compile', capture, create=True):
        model.make_solver()
    if len(saved) != 1:
        raise AssertionError('expected exactly one private compilation')
    return saved[0]


def baseline_function():
    tree = ast.parse(Path(model.baseline.__file__).read_text(encoding='utf-8-sig'))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'solve_multi')


def toy_run(joint=True, budget=1, candidate=True):
    """A negative active view, then two optical attempts; no hidden source API."""
    views, shares, instances = [], [], []

    class Cover:
        def __init__(self):
            self.stations = np.array([[0., 0.], [1000., 0.]])
            self.negative = {c: [] for c in range(1, 21)}
            self.spacing = 1000.
            self.indices = np.empty((0, 3), dtype=int)

        def observe_negative(self, c, q):
            self.negative[c].append(np.asarray(q).copy())

        def complete(self, c):
            return any(np.array_equal(p, self.stations[1]) for p in self.negative[c])

        def needed_stations(self, channels):
            return [1] if channels else []

        def certificate(self, c):
            return {'synthetic_channel': c}

    class Belief:
        def __init__(self, q, feedback):
            self.P = np.array([[2000., -100.], [2200., -100.], [2200., 100.], [2000., 100.]])
            self.near = None
            self.measured = [np.asarray(q).copy()]
            self.positives = [np.asarray(q).copy()]
            self.negatives, self.failed_clears = [], []
            instances.append(self)

        def update(self, q, feedback):
            self.measured.append(np.asarray(q).copy())
            if feedback['measure_result'] == 'no_signal':
                self.negatives.append(np.asarray(q).copy())
            else:
                self.positives.append(np.asarray(q).copy())

    class Public:
        def __init__(self):
            self.position = np.zeros(2)
            self.channel = 1
            self.events = []
            self.clear_count = 0

        def measure(self, q, c):
            self.position, self.channel = np.asarray(q).copy(), c
            result = 'direction' if c == 1 and not self.events else 'no_signal'
            reply = dict(measure_result=result, svd_deg=0.)
            self.events.append(('measure', c, tuple(q), result))
            return reply

        def clear(self, q, c):
            self.position = np.asarray(q).copy()
            self.clear_count += 1
            result = 'success' if self.clear_count == 2 else 'no_target_in_range'
            self.events.append(('clear', c, tuple(q), result))
            return dict(clear_result=result)

    def choose(*args):
        views.append('original')
        return dict(q=np.array([500., 0.]), score=100., signal_mass=.5, scenarios=1)

    def select(*args):
        views.append('joint_selector')
        return dict(q=np.array([1000., 0.]), station=1) if joint else None

    def plan(*args):
        return dict(path=np.array([[2000., 0.], [2100., 0.]]), score=1000.)

    tree = generated_ast() if candidate else ast.Module(body=[baseline_function()], type_ignores=[])
    fn = tree.body[0]
    share = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == 'share_at_stop')
    share.body.insert(0, ast.parse('_share_hook(tuple(api.position))').body[0])
    ns = model.solve_multi.__globals__.copy()
    ns.update(PolarCover=Cover, Belief=Belief, choose_measure=choose,
              select_joint_measure=select, optical_plan=plan, score_at=lambda *a, **k: None,
              core=SimpleNamespace(nearest_certified_clear=lambda *a: None,
                                   mec=lambda P: (P.mean(axis=0), 150.)),
              open_route=lambda points, start: [len(points)-1]+list(range(len(points)-1)),
              _prune_path=lambda P, pos, neg, path: dict(kept_indices=[0, 1], original_path=path.tolist()),
              _share_hook=shares.append)
    exec(compile(ast.fix_missing_locations(tree), '<synthetic-control-flow>', 'exec'), ns)
    api, trace = Public(), []
    result = ns['solve_multi'](api, trace=trace, max_active_measures=budget)
    return dict(result=result, trace=trace, events=api.events, channel=api.channel,
                views=views, shares=shares, beliefs=instances)


class JointViewChecks(unittest.TestCase):
    def test_only_three_structural_changes_and_contiguous_optical_proof(self):
        old, new = baseline_function(), generated_ast().body[0]
        old_nested = {n.name: n for n in old.body if isinstance(n, ast.FunctionDef)}
        new_nested = {n.name: n for n in new.body if isinstance(n, ast.FunctionDef)}
        self.assertEqual(set(old_nested), set(new_nested))
        for name in old_nested.keys()-{'service'}:
            self.assertEqual(ast.dump(old_nested[name]), ast.dump(new_nested[name]), name)
        service = new_nested['service']
        joint_index = next(i for i, n in enumerate(service.body)
                           if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'joint')
        self.assertEqual(ast.unparse(service.body[joint_index+1].test), 'joint is not None')
        del service.body[joint_index:joint_index+2]
        self.assertEqual(ast.dump(service), ast.dump(old_nested['service']))
        modified = [n for n in ast.walk(new) if isinstance(n, ast.If)
                    and "status != 'joint_measured'" in ast.unparse(n.test)]
        self.assertEqual(len(modified), 1)
        modified[0].test = ast.parse("variant == 'share25' and not share_at_stop(set())", mode='eval').body
        self.assertEqual(ast.dump(new), ast.dump(old))
        # This full-AST equality also binds the original prune proof, complete
        # optical_plan and uninterrupted for/clear block, rather than names only.
        self.assertNotIn('select_joint_measure', model.baseline.solve_multi.__globals__)

    def test_fixed_open_route_geometry_and_once_only_fees(self):
        points = np.array([[4., 0.], [4., 3.], [0., 3.]])
        np.testing.assert_allclose(model.continuation_cost(points, [6, 0, 18], [[0, 0]]), [11/5+24])
        np.testing.assert_allclose(model.continuation_cost(points[1:], [0, 18], [[0, 0]]), [9/5+18])
        np.testing.assert_array_equal(model.continuation_cost([], [], [[123, 456], [0, 0]]), [0, 0])

    def test_optical_route_attaches_to_each_success_exit(self):
        path = np.array([[-100., 0.], [100., 0.], [200., 0.]])
        value = model.optical_forecast(None, np.array([999., 999.]), [[0., 1.]], [12],
                                      quadrature=path[:2], plan=dict(path=path, score=17.))
        expected = (.8*math.sqrt(10001)+.2*math.sqrt(40001))/5+12
        self.assertAlmostEqual(value['continuation_s'], expected, places=12)
        self.assertEqual(value['local_s'], 17.)
        self.assertGreater(abs(expected-(math.hypot(40., 1.)/5+12)), 10.)

    def test_no_pending_unknown_or_scenarios_returns_original(self):
        belief = SimpleNamespace(scenarios=lambda: (np.array([[0., 0.]]), []))
        args = (belief, np.zeros(2), 1, 1, None, {'score': 10}, [], [], [2], None, [])
        self.assertIsNone(model.select_joint_measure(*args))
        self.assertIsNone(model.select_joint_measure(*args[:6], [('scan', 0)], [[1, 0]], [], None, []))
        self.assertIsNone(model.select_joint_measure(*args[:6], [('scan', 0)], [[1, 0]], [2], None, []))

    def test_selector_removes_only_chosen_station_and_keeps_scan_fees(self):
        belief = SimpleNamespace(P=np.zeros((1, 2)),
                                 scenarios=lambda: (np.zeros((1, 2)), [(0, np.zeros(2), None, 1000., 1.)]))
        points = np.array([[4., 0.], [4., 3.], [0., 3.]])
        tasks = [('scan', 7), ('source', 9), ('scan', 8)]
        cover = SimpleNamespace(negative={2: [points[0]], 3: []})
        calls = []

        def forecast(b, start, q, c, current, route, fees, prepared):
            calls.append((np.array(q), np.array(route), list(fees)))
            local = 100. if np.array_equal(q, [99., 99.]) else 1.
            future = float(model.continuation_cost(route, fees, [q])[0])
            return dict(local_s=local, continuation_s=future, score=local+future, signal_mass=.5, scenarios=1)

        trace = []
        with patch.object(model, 'measure_forecast', forecast):
            chosen = model.select_joint_measure(belief, np.zeros(2), 1, 1,
                dict(q=np.array([99., 99.]), score=100.), dict(score=200.),
                tasks, points, [2, 3], cover, trace)
        self.assertIsNotNone(chosen)
        self.assertEqual(trace[0]['frozen_scan_fees_s'], [6, 0, 12])
        self.assertEqual([c[2] for c in calls], [[6, 0, 12], [0, 12], [6, 0]])
        np.testing.assert_array_equal(calls[1][1], points[[1, 2]])
        np.testing.assert_array_equal(calls[2][1], points[[0, 1]])
        for candidate in trace[0]['candidates']:
            self.assertEqual(candidate['scan_fee_s']+calls[1 if candidate['station'] == 7 else 2][2][0]
                             +calls[1 if candidate['station'] == 7 else 2][2][1], 18)
        with patch.object(model, 'measure_forecast', lambda *a: None):
            # Optical baseline is unchanged when every joint station is inadmissible.
            with patch.object(model, 'optical_forecast', return_value=dict(local_s=200., continuation_s=18.)):
                self.assertIsNone(model.select_joint_measure(belief, np.zeros(2), 1, 1, None,
                    dict(score=200.), tasks, points, [2, 3], cover, []))

    def test_negative_joint_view_spends_budget_and_shares_once(self):
        run = toy_run(joint=True, budget=1)
        self.assertTrue(run['result']['complete'])
        self.assertEqual(run['views'], ['original', 'joint_selector'])
        self.assertEqual(run['shares'], [(0., 0.), (1000., 0.), (2100., 0.)])
        joint = [t for t in run['trace'] if t.get('reason') == 'joint_source_measure']
        self.assertEqual(len(joint), 1)
        self.assertEqual(joint[0]['result'], 'no_signal')
        self.assertTrue(any(np.array_equal(p, [1000., 0.]) for p in run['beliefs'][0].negatives))
        self.assertEqual(sum(e[0] == 'measure' and e[2] == (1000., 0.) for e in run['events']), 20)
        self.assertEqual(run['channel'], 20)  # Optical clear never changes radar channel.
        i = next(i for i, t in enumerate(run['trace']) if t['phase'] == 'ordered_optical_prune')
        self.assertEqual([t['phase'] for t in run['trace'][i:i+4]],
                         ['ordered_optical_prune', 'optical_plan', 'actual_action', 'actual_action'])
        self.assertEqual([t.get('result') for t in run['trace'][i+2:i+4]], ['no_target_in_range', 'success'])

    def test_no_joint_winner_and_exhausted_budget_preserve_baseline(self):
        for budget in (0, 1):
            with self.subTest(budget=budget):
                candidate = toy_run(joint=False, budget=budget)
                baseline = toy_run(joint=False, budget=budget, candidate=False)
                for key in ('events', 'result', 'trace', 'shares', 'channel'):
                    self.assertEqual(candidate[key], baseline[key], key)
                if budget == 0:
                    self.assertEqual(candidate['views'], [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, help='optional NEW output directory; never overwrite')
    args = parser.parse_args()
    if args.out:
        args.out.mkdir(parents=True, exist_ok=False)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(JointViewChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    summary = dict(passed=result.wasSuccessful(), tests=result.testsRun,
                   failures=[str(t) for t, _ in result.failures], errors=[str(t) for t, _ in result.errors],
                   scope='Local synthetic control flow, exact AST scope, hand-computed route/fee checks; not geometry or performance evidence.',
                   source_hashes={str(p.relative_to(HERE.parents[3])): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path(__file__), Path(model.__file__), Path(model.baseline.__file__)]})
    if args.out:
        (args.out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
