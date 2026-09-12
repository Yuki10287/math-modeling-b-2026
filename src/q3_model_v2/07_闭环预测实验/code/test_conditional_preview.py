"""Fast local checks for the one-decision known-source extension."""
from __future__ import annotations

import copy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
import numpy as np

from bootstrap import baseline
from environment import LocalArena
from validate_model import PublicAPI
from conditional_worlds import LedgerModel
import conditional_preview as preview


class ReadOnlyReceiver:
    @property
    def position(self):
        return np.zeros(2)

    @property
    def channel(self):
        return 1

    def measure(self, *args):
        raise AssertionError('selector attempted an actual measurement')

    def clear(self, *args):
        raise AssertionError('selector attempted an actual clearance')


def known_checkpoint():
    arena = LocalArena([dict(channel=1, position=[600., 0.], radius=1000.)], field='constant')
    model = LedgerModel(PublicAPI(arena), [], range_cuts=True, search_fraction=.30)
    model.measure(np.zeros(2), 1)
    model.api = ReadOnlyReceiver()
    return model


def joint_tasks():
    # Deliberately share key=1 across two kinds: identity must include kind.
    return [dict(kind='source', key=1, position=[600., 0.]),
            dict(kind='source', key=2, position=[200., 100.]),
            dict(kind='scan', key=1, position=[1200., 0.]),
            dict(kind='scan', key=2, position=[-1200., 0.])]


def fake_arena(cost):
    return SimpleNamespace(time_s=cost, counts={}, position=np.zeros(2),
                           evaluation=lambda: dict(all_cleared=True))


class ConditionalPreviewChecks(unittest.TestCase):
    def test_candidates_preserve_baseline_and_include_other_kind(self):
        tasks = joint_tasks()
        for route in (tasks, [tasks[2], tasks[3], tasks[0], tasks[1]]):
            with self.subTest(first_kind=route[0]['kind']):
                selected = preview.joint_candidates(tasks, route)
                self.assertEqual(selected[0], route[0])
                self.assertEqual(len(selected), 3)
                self.assertEqual({task['kind'] for task in selected}, {'source', 'scan'})
                self.assertEqual(len({(task['kind'], task['key']) for task in selected}), 3)
                self.assertTrue(all(task in tasks for task in selected))

    def test_gate_waits_then_only_one_joint_choice_is_permitted(self):
        model, tasks = known_checkpoint(), joint_tasks()
        selector = preview.ConditionalSelector()
        beliefs = model.beliefs
        model.beliefs = {}
        self.assertIsNone(selector(model, 0, tasks, tasks))
        self.assertFalse(selector.used)
        model.beliefs = beliefs
        self.assertIsNone(selector(model, 1, tasks[:1], tasks[:1]))
        self.assertFalse(selector.used)
        prediction = dict(candidates=[dict(costs_s=[100., 100.]),
                                      dict(costs_s=[80., 80.]),
                                      dict(costs_s=[95., 95.])])
        with patch.object(preview, 'conditional_forecast', return_value=prediction) as forecast:
            chosen = selector(model, 2, tasks, tasks)
            self.assertEqual(chosen['kind'], 'scan')
            self.assertEqual(chosen, tasks[2])
            self.assertTrue(selector.used)
            self.assertIsNone(selector(model, 3, tasks, tasks))
            self.assertIsNone(selector(model, 60, list(reversed(tasks)), list(reversed(tasks))))
            self.assertEqual(forecast.call_count, 1)
        rows = [r for r in model.trace if r.get('phase') == 'conditional_forecast']
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]['changed'])

    def test_insufficient_samples_keep_baseline_and_consume_single_attempt(self):
        model, tasks = known_checkpoint(), joint_tasks()
        selector = preview.ConditionalSelector()
        with patch.object(preview, 'conditional_worlds', return_value=[]) as worlds, \
                patch.object(preview, 'clone_conditional') as clone:
            self.assertIsNone(selector(model, 2, tasks, tasks))
            self.assertIsNone(selector(model, 3, tasks, tasks))
            self.assertEqual(worlds.call_count, 1)
            clone.assert_not_called()
        self.assertTrue(selector.used)
        row = model.trace[-1]
        self.assertTrue(row['unavailable'])
        self.assertFalse(row['changed'])
        self.assertIsNone(row['prediction'])

    def test_ledger_reference_and_unavailable_forecast_match_frozen_events(self):
        sources = [dict(channel=c, position=[0., 0.], radius=1000.) for c in range(1, 10)]
        sources.append(dict(channel=20, position=[1200., 400.], radius=1000.))
        old = LocalArena(copy.deepcopy(sources), seed=0, field='constant')
        original = baseline.solve_multi(PublicAPI(old), trace=[])
        self.assertTrue(original['complete'] and old.evaluation()['all_cleared'])
        for reference in (True, False):
            with self.subTest(reference=reference):
                arena = LocalArena(copy.deepcopy(sources), seed=0, field='constant')
                trace = []
                with patch.object(preview, 'conditional_worlds', return_value=[]) as worlds:
                    result = preview.solve_conditional(PublicAPI(arena), reference=reference, trace=trace)
                self.assertTrue(result['complete'] and arena.evaluation()['all_cleared'])
                self.assertEqual(arena.events, old.events)
                self.assertEqual(result['completion_certificate'], original['completion_certificate'])
                self.assertEqual(worlds.call_count, 0 if reference else 1)
                self.assertEqual(sum(r.get('phase') == 'conditional_forecast' for r in trace),
                                 0 if reference else 1)

    def test_partial_failed_world_does_not_make_candidate_cheap(self):
        model, candidates = known_checkpoint(), joint_tasks()[::2]
        worlds = [{1: (np.array([600., 0.]), 1000.)}, {1: (np.array([700., 0.]), 1000.)}]
        clone_results = [(object(), fake_arena(100.)), (object(), fake_arena(100.)),
                         (object(), fake_arena(1.)), ValueError('incompatible history')]
        with patch.object(preview, 'conditional_worlds', return_value=worlds), \
                patch.object(preview, 'clone_conditional', side_effect=clone_results) as clone, \
                patch.object(preview, 'continue_lean', return_value=dict(complete=True)) as run:
            prediction = preview.conditional_forecast(model, 59, candidates)
        self.assertEqual(prediction['candidates'][0]['costs_s'], [100., 100.])
        self.assertEqual(prediction['candidates'][1]['costs_s'], [1., None])
        chosen, scores = preview.decision(prediction)
        self.assertEqual(chosen, 0)
        self.assertFalse(scores[1]['valid'])
        # Both candidate orders use the same worlds and spatial-field identities.
        self.assertEqual([c.kwargs['scenario_index'] for c in clone.call_args_list], [0, 1, 0, 1])
        self.assertEqual([c.kwargs['field'] for c in clone.call_args_list],
                         ['smooth', 'extreme', 'smooth', 'extreme'])
        self.assertTrue(all(c.kwargs['start_iteration'] == 59 for c in run.call_args_list))
        self.assertTrue(all('selector' not in c.kwargs for c in run.call_args_list))
        with patch.object(preview, 'conditional_worlds', return_value=worlds[:1]), \
                patch.object(preview, 'clone_conditional', return_value=(object(), fake_arena(0.))), \
                patch.object(preview, 'continue_lean', return_value=dict(complete=False, status='iteration_limit')):
            truncated = preview.conditional_forecast(model, 99, candidates[:1])
        self.assertEqual(truncated['candidates'][0]['costs_s'], [None])
        self.assertEqual(preview.decision(truncated), (0, []))
        for cost in (float('inf'), float('nan')):
            with self.subTest(nonfinite_cost=cost):
                with patch.object(preview, 'conditional_worlds', return_value=worlds[:1]), \
                        patch.object(preview, 'clone_conditional', return_value=(object(), fake_arena(cost))), \
                        patch.object(preview, 'continue_lean', return_value=dict(complete=True)):
                    nonfinite = preview.conditional_forecast(model, 59, candidates[:1])
                self.assertEqual(nonfinite['candidates'][0]['costs_s'], [None])
                self.assertEqual(preview.decision(nonfinite), (0, []))

    def test_cache_key_changes_with_history_and_belief_geometry(self):
        model, tasks = known_checkpoint(), joint_tasks()
        original = preview.conditional_key(model, 3, tasks, 6)
        model.history[0]['reply']['svd_deg'] += .01
        changed_reading = preview.conditional_key(model, 3, tasks, 6)
        self.assertNotEqual(original, changed_reading)
        model.beliefs[1]['P'][0, 0] += .01
        self.assertNotEqual(changed_reading, preview.conditional_key(model, 3, tasks, 6))


if __name__ == '__main__':
    unittest.main(verbosity=2)
