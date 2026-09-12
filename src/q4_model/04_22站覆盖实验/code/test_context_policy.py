"""Small geometry/action checks; no simulator or external connection."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from localization import Belief, choose_measure, optical_plan
from context_policy import (choose_context_measure, choose_context_optical,
                            _candidates, _context_optical)


class ContextChecks(unittest.TestCase):
    def test_empty_continuation_preserves_original_policy(self):
        belief = Belief(np.zeros(2), {'measure_result': 'direction', 'svd_deg': 0.})
        start = np.array([30., -10.])
        original = choose_measure(belief, start, 2, 1)
        context = choose_context_measure(belief, start, 2, 1, [])
        np.testing.assert_array_equal(original['q'], context['q'])
        self.assertEqual(original['score'], context['score'])
        old_plan = optical_plan(belief.P, start)
        plan = choose_context_optical(belief, start, [])
        np.testing.assert_array_equal(plan['path'], old_plan['path'])
        self.assertEqual(plan['score'], old_plan['score'])

    def test_full_cover_and_continuation_accounting(self):
        P = np.array([[0., 0.], [140., 0.], [140., 20.], [0., 20.]])
        start = np.array([70., -35.]); after = np.array([[220., 5.], [-250., 15.]])
        plan = choose_context_optical(SimpleNamespace(P=P), start, after)
        original = optical_plan(P, start)
        # All cell centers are retained even if another snake direction wins.
        self.assertEqual(len(plan['path']), len(original['path']))
        distances = np.linalg.norm(plan['path'][:, None, :] - original['path'], axis=2)
        self.assertLess(float(distances.min(axis=1).max()), 1e-9)
        self.assertLess(plan['cell_radius'], 20.)
        self.assertAlmostEqual(plan['score'], plan['local_score']+plan['continuation_s'])
        expected_last = np.linalg.norm(after-plan['path'][-1], axis=1).min()/5
        self.assertAlmostEqual(plan['continuation_worst_s'], expected_last)

    def test_route_context_can_change_snake_direction(self):
        # Symmetric endpoint quadrature: the onward task breaks the service tie.
        P = np.array([[0., 0.], [120., 0.]])
        start = np.array([60., 0.]); points = np.array([[0., 0.], [120., 0.]])
        right = _context_optical(P, start, np.array([[250., 0.]]), points)
        left = _context_optical(P, start, np.array([[-130., 0.]]), points)
        self.assertGreater(right['path'][-1, 0], right['path'][0, 0])
        self.assertLess(left['path'][-1, 0], left['path'][0, 0])
        np.testing.assert_allclose(right['path'], left['path'][::-1])
        self.assertAlmostEqual(right['score'], left['score'])

    def test_both_feedback_branches_include_continuation(self):
        # One visible and one hidden scenario produce equal branch masses.
        P = np.array([[0., -10.], [40., -10.], [40., 10.], [0., 10.]])
        q = np.array([-100., 0.]); g = np.array([20., 0.])
        rows = [(0, g, None, 1000., .5), (0, g, np.array([1., 0.]), 1000., .5)]
        belief = SimpleNamespace(P=P, measured=[], scenarios=lambda: (np.array([g]), rows))
        observed_polygons = []
        def tail(polygon, *args, **kwargs):
            observed_polygons.append(polygon.copy())
            return dict(score=80., continuation_s=30.)
        with patch('context_policy._candidates', return_value=[q]), \
                patch('context_policy._context_optical', side_effect=tail):
            action = choose_context_measure(belief, q, 2, 1, [[300., 0.]])
        self.assertAlmostEqual(action['signal_mass'], .5)
        self.assertAlmostEqual(action['score'], 6.+80.)
        self.assertAlmostEqual(action['continuation_s'], 30.)
        self.assertEqual(len(observed_polygons), 4)
        # The negative branch uses the entire original uncertainty polygon.
        np.testing.assert_array_equal(observed_polygons[0], P)

    def test_near_branch_continuation_and_bounded_candidates(self):
        g = np.array([20., 0.]); q = np.array([21., 0.])
        belief = SimpleNamespace(P=np.array([[0., -1.], [40., -1.], [40., 1.], [0., 1.]]),
                                 measured=[], scenarios=lambda: (np.array([g]), [(0,g,None,1000.,1.)]))
        with patch('context_policy._candidates', return_value=[q]):
            action = choose_context_measure(belief, q, 1, 1, [[121., 0.]])
        self.assertAlmostEqual(action['score'], 5.+5.+20.)
        self.assertAlmostEqual(action['continuation_s'], 20.)
        self.assertLessEqual(len(_candidates(belief, q, np.array([[121., 0.]]))), 12)

    def test_inputs_and_empty_scenarios(self):
        belief = SimpleNamespace(P=np.array([[0., 0.]]), scenarios=lambda: ([], []))
        self.assertIsNone(choose_context_measure(belief, np.zeros(2), 1, 1, [[20., 0.]]))
        with self.assertRaises(ValueError):
            choose_context_optical(belief, np.zeros(2), [[float('nan'), 0.]])


if __name__ == '__main__':
    unittest.main()
