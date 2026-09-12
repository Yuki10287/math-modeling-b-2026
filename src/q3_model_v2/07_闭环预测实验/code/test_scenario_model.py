"""Isolation, conditional-history and public-cost tests for design worlds."""
import unittest

import numpy as np

import bootstrap
from belief_model import FeedbackModel
from scenario_model import clone_for_world, scenario_worlds


class SealedAPI:
    """Fail loudly if cloning introspects the real simulator or copies it."""
    def __init__(self, position=(0., 0.), channel=7):
        object.__setattr__(self, '_position', np.asarray(position, float))
        object.__setattr__(self, '_channel', channel)

    def __getattribute__(self, name):
        if name == 'position':
            return object.__getattribute__(self, '_position').copy()
        if name == 'channel':
            return object.__getattribute__(self, '_channel')
        raise AssertionError(f'forbidden real API access: {name}')


def snapshot(model):
    tracker = model.coverage
    return (tracker._excluded.tobytes(), tracker._counts.tobytes(),
            repr(tracker._points), repr(tracker._seen), repr(tracker._cache),
            repr(model.positions), sorted(model.discovered), sorted(model.cleared),
            repr(model.beliefs), repr(model.trace), model.use_negatives,
            model.range_cuts, model.search_fraction)


class ScenarioModelTests(unittest.TestCase):
    def model(self):
        model = FeedbackModel(SealedAPI(), [], range_cuts=True, search_fraction=.30)
        model.discovered.update(range(1, 11))
        model.cleared.update(range(1, 11))
        return model

    def test_clone_isolated_and_original_api_has_only_two_readable_members(self):
        model = self.model()
        model.coverage.observe(11, [0., 0.])
        model.positions[11].append(np.zeros(2))
        before = snapshot(model)
        world = {11: (np.array([1400., 0.]), 1000.)}
        clone, arena = clone_for_world(model, world, scenario_index=3)
        self.assertEqual(arena.time_s, 0.)
        self.assertEqual(arena.channel, 7)
        self.assertEqual(clone.discovered, model.discovered)
        self.assertEqual(clone.cleared, model.cleared)
        self.assertFalse(np.shares_memory(clone.coverage._excluded, model.coverage._excluded))
        self.assertFalse(np.shares_memory(clone.positions[11][0], model.positions[11][0]))
        self.assertEqual(clone.measure([0., 0.], 11)['measure_result'], 'no_signal')
        clone.measure([1400., 0.], 11)
        clone.clear([1400., 0.], 11)
        clone.coverage.observe(12, [100., 0.])
        self.assertEqual(snapshot(model), before)
        self.assertEqual(len(clone.cleared), 11)
        self.assertEqual(arena.evaluation()['cleared'], 1)

    def test_direction_near_silence_and_clear_fees(self):
        clone, arena = clone_for_world(self.model(), {11: (np.array([100., 0.]), 1000.)})
        self.assertEqual(clone.measure([0., 0.], 11)['measure_result'], 'direction')
        self.assertEqual(arena.time_s, 6.)
        self.assertEqual(clone.measure([100., 0.], 11)['measure_result'], 'near')
        self.assertEqual(arena.time_s, 31.)
        self.assertEqual(clone.measure([1200., 0.], 11)['measure_result'], 'no_signal')
        self.assertEqual(arena.time_s, 256.)
        self.assertEqual(clone.clear([100., 0.], 11)['clear_result'], 'success')
        self.assertEqual(arena.time_s, 481.)
        self.assertEqual(clone.clear([100., 0.], 12)['clear_result'], 'no_target_in_range')
        self.assertEqual(arena.time_s, 484.)
        self.assertEqual(arena.channel, 11)  # clear never switches the receiver

    def test_history_consistency_and_post_clear_silence(self):
        model = self.model()
        model.coverage.observe(11, [0., 0.])
        model.positions[11].append(np.zeros(2))
        with self.assertRaisesRegex(ValueError, 'historical no_signal'):
            clone_for_world(model, {11: (np.array([1000., 0.]), 1000.)})
        clone, arena = clone_for_world(model, {11: (np.array([1400., 0.]), 1000.)})
        for _ in range(2):
            self.assertEqual(clone.measure([0., 0.], 11)['measure_result'], 'no_signal')
        self.assertEqual(clone.measure([0., 0.], 1)['measure_result'], 'no_signal')
        self.assertNotIn(1, clone.beliefs)

    def test_fixed_field_repeated_point_and_reproducibility(self):
        world = {11: (np.array([100., 0.]), 1000.)}
        for field in ('smooth', 'hash', 'extreme'):
            a, _ = clone_for_world(self.model(), world, field=field, scenario_index=5)
            b, _ = clone_for_world(self.model(), world, field=field, scenario_index=5)
            first = a.measure([0., 0.], 11)
            a.measure([0., 100.], 11)
            self.assertEqual(first, a.measure([0., 0.], 11))
            self.assertEqual(first, b.measure([0., 0.], 11))

    def test_count_gate_selection_and_world_copy(self):
        model = self.model()
        for c in range(11, 21):
            model.coverage.observe(c, [0., 0.])
            model.positions[c].append(np.zeros(2))
        before = snapshot(model)
        six, twelve = scenario_worlds(model), scenario_worlds(model, count=12)
        self.assertEqual(len(six), 6)
        self.assertEqual(len(twelve), 12)
        for a, b in zip(six, twelve[::2]):
            self.assertEqual(set(a), set(b))
            self.assertTrue(10 <= len(a) + len(model.discovered) <= 16)
            for c in a:
                np.testing.assert_array_equal(a[c][0], b[c][0])
                self.assertEqual(a[c][1], b[c][1])
                self.assertGreater(np.linalg.norm(a[c][0]), a[c][1])
        self.assertEqual(snapshot(model), before)
        for invalid in (0, 3, True, 6.5):
            with self.assertRaises(ValueError):
                scenario_worlds(model, invalid)
        model.beliefs[11] = {'P': np.array([[100., 0.]])}
        self.assertEqual(scenario_worlds(model), [])
        with self.assertRaises(ValueError):
            clone_for_world(model, {11: (np.array([1400., 0.]), 1000.)})


if __name__ == '__main__':
    unittest.main()
