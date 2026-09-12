"""Local semantic checks for closed-policy rollouts; no official or HTTP use."""
from __future__ import annotations

import copy
import hashlib
import pickle
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np

from bootstrap import baseline, FeedbackModel
from environment import LocalArena
from validate_model import PublicAPI
import closed_preview
import rollout_solver
from scenario_model import clone_for_world


def layouts():
    ten = [dict(channel=c, position=[0., 0.], radius=1000.) for c in range(1, 11)]
    sixteen = [dict(channel=c, position=[0., 0.], radius=1000.) for c in range(1, 16)]
    sixteen.append(dict(channel=20, position=[1200., 400.], radius=1000.))
    return ten, sixteen


class NoRealActions:
    """Planning may read public receiver state, but cannot perform real actions."""
    @property
    def position(self):
        return np.array([123., -40.])

    @property
    def channel(self):
        return 7

    def measure(self, *args):
        raise AssertionError('forecast attempted a real measurement')

    def clear(self, *args):
        raise AssertionError('forecast attempted a real clearance')


def absence_checkpoint():
    model = FeedbackModel(NoRealActions(), [], range_cuts=True, search_fraction=.30)
    model.discovered = set(range(1, 11))
    model.cleared = set(range(1, 11))
    for channel in range(11, 21):
        model.coverage.observe(channel, np.zeros(2))
        model.positions[channel].append(np.zeros(2))
    return model


def evidence_digest(model):
    evidence = (model.beliefs, model.discovered, model.cleared, model.positions,
                model.coverage.__dict__, model.trace, model.use_negatives,
                model.range_cuts, model.search_fraction,
                np.asarray(model.position), model.channel)
    return hashlib.sha256(pickle.dumps(evidence, protocol=5)).hexdigest()


class ClosedLoopChecks(unittest.TestCase):
    def test_complete_reference_matches_frozen_events(self):
        for sources in layouts():
            with self.subTest(source_count=len(sources)):
                old = LocalArena(copy.deepcopy(sources), seed=0, field='constant')
                new = LocalArena(copy.deepcopy(sources), seed=0, field='constant')
                original = baseline.solve_multi(PublicAPI(old), trace=[])
                reference = rollout_solver.solve_multi(PublicAPI(new), variant='reference', trace=[])
                self.assertTrue(original['complete'] and reference['complete'])
                self.assertTrue(old.evaluation()['all_cleared'] and new.evaluation()['all_cleared'])
                self.assertEqual(old.events, new.events)
                self.assertEqual(original['completion_certificate'], reference['completion_certificate'])

    def test_clone_matches_real_suffix_with_fifteen_cleared(self):
        sources = layouts()[1]
        arena = LocalArena(sources, seed=0, field='constant')
        captured = {}

        def checkpoint(model, iteration, tasks, route):
            if captured or model.cleared != set(range(1, 16)) or model.beliefs:
                return
            clone, simulated = clone_for_world(model, {20: (np.array([1200., 400.]), 1000.)},
                                               field='constant', scenario_index=0)
            captured.update(clone=clone, arena=simulated, iteration=iteration,
                            first_task=copy.deepcopy(route[0]), prefix=len(arena.events),
                            offset=arena.time_s, before_counts=arena.counts.copy(),
                            before_distance=arena.distance_m,
                            negatives=copy.deepcopy(model.coverage._points))
            self.assertEqual(clone.discovered, set(range(1, 16)))
            self.assertEqual(clone.cleared, set(range(1, 16)))
            self.assertEqual(clone.coverage._points, model.coverage._points)
            self.assertGreater(len(clone.coverage._points[20]), 0)
            self.assertNotEqual(clone.channel, 1)
            self.assertEqual(simulated.time_s, 0.)

        result = rollout_solver.solve_multi(PublicAPI(arena), variant='reference', checkpoint=checkpoint)
        self.assertTrue(result['complete'])
        self.assertTrue(captured)
        continued = rollout_solver.continue_lean(captured['clone'], start_iteration=captured['iteration'],
                                                first_task=captured['first_task'])
        simulated = captured['arena']
        self.assertTrue(continued['complete'] and simulated.evaluation()['all_cleared'])
        self.assertEqual(continued['cleared'], 16)
        suffix = arena.events[captured['prefix']:]
        self.assertEqual(len(suffix), len(simulated.events))
        for actual, predicted in zip(suffix, simulated.events):
            self.assertEqual({k: v for k, v in actual.items() if k != 'time_s'},
                             {k: v for k, v in predicted.items() if k != 'time_s'})
            self.assertAlmostEqual(actual['time_s'] - captured['offset'], predicted['time_s'], places=8)
        self.assertAlmostEqual(arena.time_s - captured['offset'], simulated.time_s, places=8)
        self.assertAlmostEqual(arena.distance_m - captured['before_distance'], simulated.distance_m, places=8)
        self.assertEqual({k: arena.counts[k] - captured['before_counts'][k] for k in arena.counts},
                         simulated.counts)
        self.assertEqual(continued['completion_certificate'], result['completion_certificate'])

    def test_adaptive_boundary_59_60_and_forced_first_consumes_iteration(self):
        model = absence_checkpoint()
        clone, arena = clone_for_world(model, {})
        with patch.object(rollout_solver, 'build_scan_tasks', wraps=rollout_solver.build_scan_tasks) as build:
            _, adaptive, _ = rollout_solver.plan(clone, 59)
            self.assertEqual(build.call_count, 1)
            _, fixed, _ = rollout_solver.plan(clone, 60)
            self.assertEqual(build.call_count, 1)
        self.assertTrue(all(isinstance(task['key'], str) for task in adaptive))
        self.assertTrue(all(isinstance(task['key'], int) for task in fixed))
        with patch.object(rollout_solver, 'build_scan_tasks', wraps=rollout_solver.build_scan_tasks) as build:
            result = rollout_solver.continue_lean(clone, start_iteration=59, first_task=adaptive[0])
            self.assertEqual(build.call_count, 1)
        self.assertTrue(result['complete'])
        iterations = [r['iteration'] for r in clone.trace if r.get('phase') == 'plan']
        self.assertGreater(len(iterations), 1)
        self.assertEqual(iterations, list(range(59, 59 + len(iterations))))
        self.assertGreater(len(arena.events), 0)

    def test_iteration_99_truncates_and_100_does_no_actions(self):
        for start in (99, 100):
            with self.subTest(start=start):
                clone, arena = clone_for_world(absence_checkpoint(), {})
                result = rollout_solver.continue_lean(clone, start_iteration=start)
                self.assertFalse(result['complete'])
                self.assertEqual(result['status'], 'iteration_limit')
                self.assertEqual([r['iteration'] for r in clone.trace if r.get('phase') == 'plan'],
                                 [99] if start == 99 else [])
                self.assertEqual(bool(arena.events), start == 99)

    def test_forecast_is_order_invariant_and_does_not_touch_real_evidence(self):
        model = absence_checkpoint()
        _, route, _ = rollout_solver.plan(model, 3)
        candidates = route[:2]
        before = evidence_digest(model)
        with patch.object(closed_preview, 'scenario_worlds', return_value=[{}]):
            forward = closed_preview.forecast(model, 3, candidates)
            self.assertEqual(evidence_digest(model), before)
            backward = closed_preview.forecast(model, 3, list(reversed(candidates)))
            self.assertEqual(evidence_digest(model), before)
        def costs(prediction):
            return {row['task']['key']: row['costs_s'] for row in prediction['candidates']}
        self.assertEqual(costs(forward), costs(backward))
        self.assertTrue(all(row['costs_s'][0] is not None for row in forward['candidates']))

    def test_truncated_and_failed_forecasts_are_invalid_not_cheap(self):
        model = absence_checkpoint()
        _, route, _ = rollout_solver.plan(model, 99)
        with patch.object(closed_preview, 'scenario_worlds', return_value=[{}]):
            truncated = closed_preview.forecast(model, 99, route[:1])
        row = truncated['candidates'][0]
        self.assertEqual(row['costs_s'], [None])
        self.assertIn('incomplete scenario', row['errors'][0])
        self.assertEqual(closed_preview.decision(truncated), (0, []))
        with patch.object(closed_preview, 'scenario_worlds', return_value=[{}]), \
                patch.object(closed_preview, 'continue_lean', side_effect=RuntimeError('rejected rollout')):
            failed = closed_preview.forecast(model, 99, route[:1])
        self.assertEqual(failed['candidates'][0]['costs_s'], [None])
        self.assertEqual(closed_preview.decision(failed), (0, []))
        prediction = dict(candidates=[dict(costs_s=[100., 100.]),
                                      dict(costs_s=[0., None]),
                                      dict(costs_s=[80., 80.])])
        chosen, scores = closed_preview.decision(prediction)
        self.assertEqual(chosen, 2)
        self.assertFalse(scores[1]['valid'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
