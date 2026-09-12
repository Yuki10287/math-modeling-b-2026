import math
import unittest
import numpy as np
from directional_cover import DirectionalCover,inside
from localization import Belief,optical_plan
from benchmark import LocalArena,public_api
from solver import accepted
from shared import core
from validation import polygon_contains


class ModelChecks(unittest.TestCase):
    def test_degenerate_audit_is_bounded(self):
        self.assertTrue(polygon_contains([[0,0],[1,0]],[.5,0]))
        self.assertFalse(polygon_contains([[0,0],[1,0]],[2,0]))
        self.assertFalse(polygon_contains([[0,0],[.5,0],[1,0]],[2,0]))

    def test_continuous_mesh_construction(self):
        mesh=DirectionalCover()
        edges=np.roll(mesh.triangles,-1,axis=1)-mesh.triangles
        self.assertLess(float(np.linalg.norm(edges,axis=2).max()),1000)
        angles=np.linspace(0,2*math.pi,1440,endpoint=False)
        boundary=1800*np.column_stack((np.cos(angles),np.sin(angles)))
        rng=np.random.default_rng(7331)
        points=rng.uniform(-1800,1800,(10000,2))
        points=np.vstack((boundary,points[np.linalg.norm(points,axis=1)<=1800]))
        covered=np.zeros(len(points),bool)
        for triangle in mesh.triangles:covered|=inside(triangle,points)
        self.assertTrue(covered.all())
        for q in mesh.stations:mesh.observe_negative(1,q)
        self.assertTrue(mesh.complete(1))

    def test_outward_boundary_source_defeats_q3_stations(self):
        g=np.array([1800.,0.]);n=np.array([1.,0.])
        angle=np.arange(6)*math.pi/3
        old=np.vstack(([0.,0.],1200*np.column_stack((np.cos(angle),np.sin(angle)))))
        self.assertTrue(np.all((old-g)@n<0))
        mesh=DirectionalCover()
        for theta in np.linspace(0,2*math.pi,721):
            n=np.array([math.cos(theta),math.sin(theta)])
            self.assertTrue(np.any((np.linalg.norm(mesh.stations-g,axis=1)<=1000)&((mesh.stations-g)@n>=-1e-8)))

    def test_negative_does_not_delete_nearby_directional_source(self):
        arena=LocalArena([dict(channel=1,position=[100.,0.],radius=1000.,orientation=math.pi)])
        first=arena.measure([0.,0.],1);b=Belief(np.zeros(2),first);old=b.P.copy()
        feedback=arena.measure([110.,0.],1)
        self.assertEqual(feedback['measure_result'],'no_signal')
        b.update(np.array([110.,0.]),feedback)
        self.assertTrue(np.array_equal(old,b.P))
        self.assertEqual(arena.clear([110.,0.],1)['clear_result'],'success')

    def test_optical_rectangle_and_hypothesis_separation(self):
        for angle in (0.,89.99,180.,359.99):
            P=core.initial_belief(np.zeros(2),angle)
            plan=optical_plan(P,np.zeros(2))
            self.assertLess(plan['cell_radius'],20)
            self.assertEqual(len(plan['path']),math.prod(plan['cells']))
        b=Belief(np.zeros(2),{'measure_result':'direction','svd_deg':0.})
        b.update(np.array([750.,100.]),{'measure_result':'no_signal'})
        _,rows=b.scenarios()
        for _,g,n,radius,_ in rows:
            for p in b.positives:
                self.assertLessEqual(np.linalg.norm(p-g),radius+1e-7)
                self.assertTrue(n is None or (p-g)@n>=-1e-7)
            for p in b.negatives:
                self.assertTrue(np.linalg.norm(p-g)>radius or (n is not None and (p-g)@n<0))

    def test_public_feedback_boundary(self):
        api=public_api(LocalArena([]))
        for name in ('_sources','evaluation','_removed','time_s'):
            with self.assertRaises(AttributeError):getattr(api,name)
        with self.assertRaises(ValueError):accepted({'accepted':False,'measure_result':'no_signal'},'measure_result',('no_signal',))
        with self.assertRaises(ValueError):accepted({'measure_result':'direction','svd_deg':float('nan')},'measure_result',('direction',))


if __name__=='__main__':unittest.main()
