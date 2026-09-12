"""Checks the geometry and action rules used in cost scenarios."""
import unittest
import numpy as np
import geometry as core
from forecast_policy import segment_clear,continuation,optical_tail
from local_policy import area_scenarios
from validate_model import independent_disk_cover


class ForecastChecks(unittest.TestCase):
    def test_segment_formula_matches_existing_bisection(self):
        rng=np.random.default_rng(7341)
        for _ in range(100):
            angles=np.sort(rng.uniform(0,2*np.pi,7))
            P=np.column_stack((np.cos(angles),np.sin(angles)))*rng.uniform(1,19)
            s=rng.uniform(-1500,1500,2)
            c,r=core.mec(P)
            q=segment_clear(P,s,c)
            self.assertLessEqual(np.linalg.norm(P-q,axis=1).max(),20.)
            np.testing.assert_allclose(q,core.nearest_certified_clear(P,s),rtol=0,atol=5e-7)

    def test_small_optical_tail_covers_entire_polygon(self):
        for P in (np.array([[-32,-10],[32,-10],[32,10],[-32,10]]),
                  np.array([[-20,-20],[20,-20],[20,20],[-20,20]])):
            path=optical_tail(P,np.array([-100.,60.]))
            self.assertTrue(independent_disk_cover(P,path)['complete'])

    def test_future_measurements_are_certified_and_unrepeated(self):
        P=core.initial_belief(np.zeros(2),18.)
        first=core.mec(P)[0]
        for g in area_scenarios(P,4):
            reading=np.degrees(np.arctan2(*(g-first)[::-1]))
            region=core.wedge(P,first,reading)
            for error in (-1.,0.,1.):
                audit=[]
                cost=continuation(region,first,g,error,[np.zeros(2),first],variant='bounded',audit=audit)
                self.assertTrue(np.isfinite(cost))
                seen=[np.zeros(2),first]
                for event in audit:
                    if event['kind']=='measure':
                        q=np.asarray(event['q'])
                        self.assertTrue(core.reception_certified(np.asarray(event['polygon']),q,np.asarray(event['witness'])))
                        self.assertTrue(all(np.linalg.norm(q-old)>.1 for old in seen))
                        seen.append(q)


if __name__=='__main__':unittest.main()
