"""Geometric and evidence-preservation checks for joint route proposals."""
import unittest
import numpy as np

from coverage_model import CoverageTracker
from joint_planning import project_disks, shortest_visit, joint_route, route_cost
from scan_planning import build_scan_tasks
from v1_solver import rolling_route
from flexible_coverage import flexible_route


class JointGeometryTests(unittest.TestCase):
    def test_lens_projection_has_two_active_constraints(self):
        centers = np.array([[-600., 0.], [600., 0.]])
        q = project_disks([0.,1600.],centers,1000.,[0.,0.])
        np.testing.assert_allclose(q,[0.,800.],atol=2e-5,rtol=0)
        self.assertLessEqual(np.linalg.norm(centers-q,axis=1).max(),1000.)

    def test_entry_and_exit_share_a_feasible_line(self):
        q = shortest_visit([-50.,10.],[50.,10.],np.array([[0.,0.]]),20.,[0.,0.])
        self.assertAlmostEqual(np.linalg.norm(q-[-50.,10.])+np.linalg.norm(q-[50.,10.]),100.,places=5)
        self.assertLessEqual(np.linalg.norm(q),20.)

    def test_reject_infeasible_projection_seed(self):
        with self.assertRaises(ValueError):
            project_disks([0.,0.],np.array([[100.,0.]]),10.,[0.,0.])

    def test_random_disk_intersections_against_dense_feasible_oracle(self):
        rng = np.random.default_rng(58123)
        for _ in range(50):
            centers = rng.uniform(-.55,.55,(8,2))
            target = rng.uniform(-3,3,2)
            q = project_disks(target,centers,1.,np.zeros(2))
            self.assertLessEqual(np.linalg.norm(centers-q,axis=1).max(),1.+1e-9)
            samples = rng.uniform(-1,1,(2000,2))
            keep = np.max(np.linalg.norm(samples[:,None]-centers[None],axis=2),axis=1)<=1.
            if np.any(keep):
                self.assertLessEqual(np.linalg.norm(q-target),np.linalg.norm(samples[keep]-target,axis=1).min()+1e-5)

    def test_route_preserves_evidence_and_whole_cell_coverage(self):
        tracker = CoverageTracker(channels=[1,2])
        for channel in (1,2):
            for q in ([0,0],[-1200,0],[-600,-1039.23048454]):
                tracker.observe(channel,q)
        tracker.observe(1,[600,1039.23048454])
        before = tracker._excluded.copy()
        certificates = [tracker.certificate(c) for c in (1,2)]
        sources = [dict(kind='source',key=3,position=[800,1300]),
                   dict(kind='source',key=4,position=[1500,-350])]
        start = np.array([-400.,1000.])
        old = build_scan_tasks(tracker,[1,2],start,sources)
        _, old_distance = rolling_route(start,old+sources)
        for planner in (joint_route, flexible_route):
            route,distance = planner(tracker,[1,2],start,sources)
            self.assertLessEqual(distance,old_distance+1e-6)
            self.assertAlmostEqual(distance,route_cost(start,route))
            masks = [tracker._mask(x['position']) for x in route if x['kind']=='scan']
            for c in (1,2):
                self.assertTrue(np.all(tracker._excluded[tracker._index(c)]|np.any(masks,axis=0)))
        np.testing.assert_array_equal(before,tracker._excluded)
        self.assertEqual(certificates,[tracker.certificate(c) for c in (1,2)])


if __name__ == '__main__':
    unittest.main()
