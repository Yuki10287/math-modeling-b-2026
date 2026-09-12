"""Read-only, count-bound and feedback-branch checks for scan preview."""
import pickle
import time
import unittest
from types import SimpleNamespace

import numpy as np

from bootstrap import FeedbackModel, core
from scan_preview import _rollout, _worlds, choose_scan_task, rank_scan_tasks
from v1_solver import rolling_route


class ScanPreviewTests(unittest.TestCase):
    def model(self):
        # This API deliberately exposes no simulator actions or hidden truth.
        return FeedbackModel(SimpleNamespace(position=np.array([0., 0.]), channel=1), [])

    def test_full_plan_selection_preserves_all_evidence_and_tasks(self):
        model = self.model()
        tasks = [dict(kind='scan', key=i, position=q.tolist())
                 for i, q in enumerate(core.coverage_points())]
        route, _ = rolling_route(model.position, tasks)
        before = pickle.dumps((model.__dict__, tasks, route))
        began = time.perf_counter()
        ranked = rank_scan_tasks(model, tasks, route)
        elapsed = time.perf_counter() - began
        self.assertEqual(len(ranked), len(tasks))
        self.assertTrue(all(np.isfinite(score) for score, _ in ranked))
        self.assertTrue(all(any(task is original for original in tasks) for _, task in ranked))
        self.assertEqual(before, pickle.dumps((model.__dict__, tasks, route)))
        print(f'full seven-scan preview: {elapsed:.3f} s')

    def test_scenarios_obey_counts_and_actual_negative_feedback(self):
        model = self.model()
        model.discovered.update(range(1, 11))
        model.cleared.update(range(1, 11))
        for channel in range(11, 21):
            model.coverage.observe(channel, [0., 0.])
        channels = model.unknown()
        worlds = _worlds(model, channels)
        self.assertIsNotNone(worlds)
        for world in worlds:
            self.assertLessEqual(10 + len(world), 16)
            self.assertGreaterEqual(10 + len(world), 10)
            for channel, (point, radius) in world.items():
                self.assertIn(channel, channels)
                self.assertGreater(np.linalg.norm(point), radius)
                self.assertLessEqual(np.linalg.norm(point), 1800.)

    def test_positive_branch_exits_scan_negative_branch_continues(self):
        points = np.array([[0., 0.], [100., 0.]])
        gains = [np.array([True]), np.array([True])]
        absent = _rollout(np.zeros(2), [0, 1], points, gains, [1], {}, 15)
        present = _rollout(np.zeros(2), [0, 1], points, gains, [1],
                           {1: (np.zeros(2), 1000.)}, 15)
        self.assertEqual(absent, 32.)  # Two scans plus the 100 m move.
        self.assertEqual(present, 6.)  # Sixteenth source stops further search.

    def test_incomplete_plan_or_pending_source_keeps_existing_policy(self):
        model = self.model()
        tasks = [dict(kind='scan', key=0, position=[0., 0.]),
                 dict(kind='scan', key=1, position=[10., 0.])]
        self.assertIsNone(choose_scan_task(model, tasks, tasks))
        model.beliefs[1] = {'P': np.array([[0., 0.]])}
        self.assertIsNone(choose_scan_task(model, tasks, tasks))


if __name__ == '__main__':
    unittest.main()
