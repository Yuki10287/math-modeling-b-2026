"""Public-action equivalence and conditional-world boundary checks."""
import copy
import pickle
import unittest
import numpy as np

import bootstrap
from belief_model import FeedbackModel
from environment import LocalArena
from validate_model import PublicAPI
from conditional_worlds import (LedgerModel, conditional_worlds, clone_conditional,
                                _inside_polygon, _radius_interval)


class ScriptAPI:
    def __init__(self, reply=None):
        self.position, self.channel = np.zeros(2), 1
        self.reply = reply

    def measure(self, q, channel):
        self.position, self.channel = np.asarray(q).copy(), channel
        return copy.deepcopy(self.reply)

    def clear(self, q, channel):
        self.position = np.asarray(q).copy()
        return copy.deepcopy(self.reply)


class SealedState:
    def __init__(self, position, channel):
        object.__setattr__(self, '_p', np.asarray(position).copy())
        object.__setattr__(self, '_c', channel)

    def __getattribute__(self, name):
        if name == 'position':
            return object.__getattribute__(self, '_p').copy()
        if name == 'channel':
            return object.__getattribute__(self, '_c')
        raise AssertionError(f'forbidden real API access: {name}')


def state(model):
    return pickle.dumps({k: v for k, v in model.__dict__.items() if k != 'api'})


class ConditionalWorldTests(unittest.TestCase):
    def model(self, reply=None):
        api = ScriptAPI(reply)
        model = LedgerModel(api, [], range_cuts=True, search_fraction=.30)
        model.discovered.update(range(1, 11))
        model.cleared.update(range(1, 11))
        return model, api

    def test_ledger_preserves_actions_responses_and_rejects_failed_evidence(self):
        sources = [dict(channel=11, position=[100., 0.], radius=1000.)]
        a, b = LocalArena(sources, seed=5), LocalArena(sources, seed=5)
        baseline = FeedbackModel(PublicAPI(a), [], range_cuts=True)
        ledger = LedgerModel(PublicAPI(b), [], range_cuts=True)
        for action, q in [('measure', [0., 0.]), ('measure', [100., 0.]),
                          ('clear', [200., 0.]), ('clear', [100., 0.]),
                          ('measure', [0., 0.])]:
            self.assertEqual(getattr(baseline, action)(q, 11), getattr(ledger, action)(q, 11))
        self.assertEqual(a.events, b.events)
        self.assertEqual(baseline.trace, ledger.trace)
        self.assertEqual(len(ledger.history), 5)
        model, api = self.model(dict(accepted=False, measure_result='no_signal'))
        with self.assertRaises(RuntimeError):
            model.measure([0., 0.], 11)
        self.assertEqual(model.history, [])
        api.reply = dict(accepted=False, clear_result='success')
        with self.assertRaises(RuntimeError):
            model.clear([0., 0.], 11)
        self.assertEqual(model.history, [])

    def test_wrapped_bearing_replay_cost_and_independent_optical_state(self):
        original_reply = dict(accepted=True, measure_result='direction', svd_deg=359.5)
        model, _ = self.model(original_reply)
        model.measure([0., 0.], 11)
        model.beliefs[11]['state'] = {'path': [np.array([70., -1.])], 'index': 0}
        model.api = SealedState([0., 0.], 11)
        before = state(model)
        theta = np.radians(.4)
        point = 100. * np.array([np.cos(theta), np.sin(theta)])
        clone, arena = clone_conditional(model, {11: (point, 1000.)}, field='hash', scenario_index=9)
        self.assertFalse(np.shares_memory(clone.beliefs[11]['state']['path'][0],
                                         model.beliefs[11]['state']['path'][0]))
        clone.beliefs[11]['state']['path'][0][0] = 999.
        self.assertEqual(state(model), before)
        self.assertEqual(clone.measure([0., 0.], 11), original_reply)
        self.assertEqual(arena.time_s, 5.)
        self.assertEqual(arena.events[-1]['svd_deg'], 359.5)
        first = clone.measure([0., 10.], 11)
        self.assertAlmostEqual(arena.time_s, 12.)
        self.assertEqual(clone.measure([0., 0.], 11), original_reply)
        self.assertEqual(clone.measure([0., 10.], 11), first)
        clone.beliefs[11]['anchor'][0] = 777.
        clone.history[0]['reply']['svd_deg'] = 12.
        self.assertEqual(state(model), before)
        clone.clear(point, 11)
        reply = clone.measure([0., 0.], 11)
        self.assertEqual(reply['measure_result'], 'no_signal')
        self.assertNotIn('svd_deg', arena.events[-1])
        self.assertEqual(clone.measure([0., 0.], 1)['measure_result'], 'no_signal')

    def test_near_outer_polygon_is_filtered_by_actual_five_meter_disk(self):
        model, _ = self.model(dict(measure_result='near'))
        model.measure([0., 0.], 11)
        polygon = model.beliefs[11]['P']
        point = polygon[np.argmax(np.linalg.norm(polygon, axis=1))] * .999999
        self.assertGreater(np.linalg.norm(point), 5.)
        self.assertTrue(_inside_polygon(point, polygon))
        self.assertIsNone(_radius_interval(model, 11, point))
        with self.assertRaises(ValueError):
            clone_conditional(model, {11: (point, 1000.)})
        worlds = conditional_worlds(model)
        self.assertEqual(len(worlds), 6)
        self.assertTrue(all(np.linalg.norm(world[11][0]) <= 5. for world in worlds))

    def test_failed_clear_hole_and_strict_fixed_radius_upper_bound(self):
        model, api = self.model(dict(measure_result='direction', svd_deg=0.))
        model.measure([0., 0.], 11)
        api.reply = dict(clear_result='no_target_in_range')
        model.clear([500., 0.], 11)
        self.assertTrue(_inside_polygon(np.array([500., 0.]), model.beliefs[11]['P']))
        self.assertIsNone(_radius_interval(model, 11, np.array([500., 0.])))
        with self.assertRaises(ValueError):
            clone_conditional(model, {11: (np.array([520., 0.]), 1000.)})
        self.assertIsNotNone(_radius_interval(model, 11, np.array([521., 0.])))
        # R must be strictly less than distance to an old silent point.
        other, api = self.model(dict(measure_result='no_signal'))
        other.measure([0., 0.], 11)
        api.reply = dict(measure_result='direction', svd_deg=0.)
        other.measure([500., 0.], 11)
        point = np.array([1200., 0.])
        lower, upper = _radius_interval(other, 11, point)
        self.assertEqual(lower, 1000.)
        self.assertLess(upper, 1200.)
        clone_conditional(other, {11: (point, upper)})
        with self.assertRaises(ValueError):
            clone_conditional(other, {11: (point, 1200.)})

    def test_world_counts_history_and_source_beliefs_are_independent(self):
        model, api = self.model(dict(measure_result='no_signal'))
        for c in range(11, 21):
            model.measure([0., 0.], c)
        api.reply = dict(measure_result='direction', svd_deg=0.)
        model.measure([500., 0.], 11)
        before = state(model)
        worlds = conditional_worlds(model)
        self.assertEqual(len(worlds), 6)
        for world in worlds:
            self.assertIn(11, world)
            self.assertTrue(10 <= len(world) + len(model.cleared) <= 16)
            clone, _ = clone_conditional(model, world)
            self.assertEqual(clone.discovered, model.discovered)
            for point, radius in world.values():
                self.assertGreater(np.linalg.norm(point), radius)
        self.assertEqual(state(model), before)
        # A convex relaxation may retain only incompatible representatives.
        model.beliefs[11]['P'] = np.array([[500., 0.]])
        self.assertEqual(conditional_worlds(model), [])

    def test_inconsistent_fixed_point_or_incomplete_history_is_rejected(self):
        model, _ = self.model(dict(measure_result='direction', svd_deg=0.))
        model.measure([0., 0.], 11)
        model.history.append(copy.deepcopy(model.history[0]))
        model.history[-1]['reply']['svd_deg'] = .5
        self.assertEqual(conditional_worlds(model), [])
        with self.assertRaisesRegex(ValueError, 'fixed-point'):
            clone_conditional(model, {11: (np.array([100., 0.]), 1000.)})
        model.history = []
        self.assertEqual(conditional_worlds(model), [])


if __name__ == '__main__':
    unittest.main()
