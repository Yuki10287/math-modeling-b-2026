"""Check inclusive task thresholds and bearing-endpoint containment explicitly."""
import math
import unittest
import numpy as np
from stress_check import EndpointArena
from localization import Belief
from validation import polygon_contains


class ThresholdChecks(unittest.TestCase):
    def arena(self, radius=1000, orientation=0):
        return EndpointArena([dict(channel=2, position=[0., 0.], radius=radius,
                                   orientation=orientation)], field='plus_one')

    def test_inclusive_radius_and_semicircle_boundary(self):
        for radius in (1000, 1500):
            arena = self.arena(radius)
            self.assertEqual(arena.measure([radius, 0], 2)['measure_result'], 'direction')
            self.assertEqual(arena.measure([radius+1e-6, 0], 2)['measure_result'], 'no_signal')
            self.assertEqual(arena.measure([0, 999], 2)['measure_result'], 'direction')
            self.assertEqual(arena.measure([-1e-6, 999], 2)['measure_result'], 'no_signal')

    def test_near_and_optical_radius_include_boundary_and_clear_ignores_heading(self):
        arena = self.arena()
        self.assertEqual(arena.measure([5, 0], 2)['measure_result'], 'near')
        self.assertEqual(arena.measure([5+1e-6, 0], 2)['measure_result'], 'direction')
        self.assertEqual(arena.measure([-5, 0], 2)['measure_result'], 'no_signal')
        arena.measure([0, 0], 1)
        self.assertEqual(arena.clear([-20-1e-6, 0], 2)['clear_result'], 'no_target_in_range')
        self.assertEqual(arena.clear([-20, 0], 2)['clear_result'], 'success')
        self.assertEqual(arena.channel, 1)

    def test_exact_error_endpoints_wrap_and_repeat_without_excluding_truth(self):
        for bearing in (0.001, 0.9951, 179.9951, 359.0049, 359.999):
            angle = math.radians(bearing)
            g = 1000*np.array([math.cos(angle), math.sin(angle)])
            for field in ('plus_one', 'minus_one', 'checker_one'):
                arena = EndpointArena([dict(channel=1, position=g.tolist(), radius=1500,
                                           orientation=None)], field=field)
                reply = arena.measure([0, 0], 1)
                self.assertEqual(reply, arena.measure([0, 0], 1))
                belief = Belief(np.zeros(2), reply)
                self.assertTrue(polygon_contains(belief.P, g))


if __name__ == '__main__':
    unittest.main()
