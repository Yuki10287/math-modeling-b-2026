"""Safety and accounting tests for global value proposals; no network."""
import copy
import unittest
from types import SimpleNamespace
import numpy as np
import geometry as core
from coverage_model import CoverageTracker
from global_value_policy import GlobalContext
from value_validation import trace_truth_audit


def fixture(channels=()):
    coverage = CoverageTracker()
    P = np.array([[-2., -2.], [2., -2.], [2., 2.], [-2., 2.]])
    belief = dict(P=P, witness=np.array([-100., 0.]), state=None, near=None)
    tasks = [dict(kind='source', key=1, position=[0., 0.]),
             dict(kind='source', key=2, position=[100., 0.])]
    if channels:
        tasks += [dict(kind='scan', key=i, position=q.tolist()) for i, q in enumerate(core.coverage_points())]
    return SimpleNamespace(coverage=coverage, beliefs={1: belief}, trace=[dict(phase='plan', route=tasks)],
        channel=1, position=np.array([-100., 0.]), unknown=lambda: list(channels),
        recorded=lambda c, q: False, search_fraction=.3)


class GlobalValueTests(unittest.TestCase):
    def test_route_correction_does_not_charge_local_approach_twice(self):
        context = GlobalContext(fixture(), 1)
        at_other = context.correction(dict(kind='measure', q=np.array([100., 0.])), 'route')
        self.assertAlmostEqual(at_other['route_correction_s'], -20.)
        final = context.correction(dict(kind='clear', certified=True, q=np.array([10., 0.])), 'route')
        self.assertAlmostEqual(final['route_correction_s'], -2.)

    def test_negative_branch_preserves_all_channels_without_changing_evidence(self):
        model = fixture((3, 4))
        model.search_fraction = .1
        model.coverage.observe(3, [0, 0])
        before = copy.deepcopy(model.coverage.__dict__)
        context = GlobalContext(model, 1)
        action = dict(kind='measure', q=np.array([1200., 0.]))
        score = context.correction(action, 'coverage')
        self.assertEqual(score['hypothetical_scan_channels'], [3, 4])
        self.assertTrue(np.isfinite(score['correction_s']))
        self.assertTrue(np.array_equal(before['_excluded'], model.coverage._excluded))
        self.assertTrue(np.array_equal(before['_counts'], model.coverage._counts))
        self.assertEqual(before['_points'], model.coverage._points)
        failed = context.correction(dict(kind='clear', certified=False, q=action['q']), 'shared')
        self.assertEqual(failed['coverage_correction_s'], 0.)
        self.assertEqual(failed['shared_credit_s'], 0.)

    def test_new_waypoints_are_reception_certified_and_unused(self):
        model = fixture((3, 4))
        model.beliefs[1]['P'] = np.array([[300., -20.], [900., -20.], [900., 20.], [300., 20.]])
        context = GlobalContext(model, 1)
        points = context.waypoint_candidates(np.array([600., 30.]))
        self.assertTrue(points)
        for q in points:
            self.assertTrue(core.reception_certified(model.beliefs[1]['P'], q, model.beliefs[1]['witness']))
            self.assertFalse(model.recorded(1, q))

    def test_audit_rejects_truth_in_a_real_concave_gap(self):
        arena = SimpleNamespace(_sources={1: dict(position=[1.5, 1.5])})
        trace = [dict(channel=1, polygon=[[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]])]
        with self.assertRaises(AssertionError):
            trace_truth_audit(arena, trace)


if __name__ == '__main__':
    unittest.main()
