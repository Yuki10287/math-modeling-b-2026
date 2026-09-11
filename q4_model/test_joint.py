import math
import unittest
import numpy as np
from polar_cover import PolarCover
from directional_cover import inside
from route_planning import open_route,route_length
from localization import Belief,choose_measure
from guarded_policy import score_at
from benchmark import LocalArena,public_api
from joint_solver import solve_multi


class JointChecks(unittest.TestCase):
    def test_polar_mesh_continuous_construction(self):
        m=PolarCover()
        self.assertEqual(len(m.stations),25);self.assertEqual(len(m.triangles),36)
        self.assertLess(m.spacing,1000)
        self.assertGreater(m.outer_radius*math.cos(math.pi/12),1800)
        # Positive orientation, total triangulated area equals outer polygon area.
        a=m.triangles[:,1]-m.triangles[:,0];b=m.triangles[:,2]-m.triangles[:,0]
        twice=a[:,0]*b[:,1]-a[:,1]*b[:,0]
        self.assertTrue(np.all(twice>0))
        self.assertAlmostEqual(float(twice.sum()/2),3*m.outer_radius**2,places=7)
        angles=np.linspace(0,2*math.pi,2048,endpoint=False)
        points=1800*np.column_stack((np.cos(angles),np.sin(angles)))
        covered=np.zeros(len(points),bool)
        for t in m.triangles:covered|=inside(t,points)
        self.assertTrue(covered.all())
        for q in m.stations:m.observe_negative(1,q)
        self.assertTrue(m.complete(1))

    def test_open_route_covers_tasks_and_improves_seed(self):
        rng=np.random.default_rng(992)
        points=rng.normal(size=(30,2))*1200;start=np.array([130.,-300.])
        order=open_route(points,start)
        self.assertEqual(sorted(order),list(range(30)))
        greedy=[];pending=set(range(30));q=start
        while pending:
            k=min(pending,key=lambda k:(np.linalg.norm(points[k]-q),k))
            greedy.append(k);pending.remove(k);q=points[k]
        self.assertLessEqual(route_length(points,start,order),route_length(points,start,greedy)+1e-7)
        self.assertEqual(open_route(np.array([[1.,0.],[2.,0.]]),np.zeros(2)),[0,1])

    def test_type_guard_is_worse_conditional_cost(self):
        b=Belief(np.zeros(2),{'measure_result':'direction','svd_deg':0.})
        old=choose_measure(b,np.zeros(2),1,1)
        mixture=score_at(b,np.zeros(2),old['q'],1,1,robust=False)
        guarded=score_at(b,np.zeros(2),old['q'],1,1,robust=True)
        self.assertAlmostEqual(old['score'],mixture['score'],places=7)
        self.assertGreaterEqual(guarded['score']+1e-8,mixture['score'])

    def test_near_feedback_and_shared_clear(self):
        sources=[dict(channel=1,position=[2.,0.],radius=1000.,orientation=None)]
        # Noncontest toy count is used only for a near-feedback branch check.
        for k in range(2,17):sources.append(dict(channel=k,position=[0.,0.],radius=1000.,orientation=None))
        arena=LocalArena(sources)
        result=solve_multi(public_api(arena),'shared')
        self.assertTrue(result['complete']);self.assertEqual(arena.evaluation()['cleared'],16)
        self.assertEqual(arena.counts['clear_fail'],0)


if __name__=='__main__':unittest.main()
