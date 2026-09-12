from copy import deepcopy
import math
import unittest

import numpy as np

from cell_cover import CellCover
from cell_validation import audit_canonical_geometry, validate_cell_run


class CellCoverChecks(unittest.TestCase):
    def test_canonical_exact_geometry_and_distinct_station_view(self):
        cover = CellCover()
        report = audit_canonical_geometry(cover)
        self.assertTrue(report['passed'])
        self.assertEqual(len(cover.stations), 22)
        self.assertEqual(cover.triangles.shape, (108, 3, 2))
        self.assertGreater(len(cover.vertices), 22)
        self.assertTrue(np.array_equal(cover.triangles, cover.vertices[cover.indices]))
        self.assertEqual(cover.geometry_certificate()['station_count'], 22)

    def test_all_channels_full_scan_gives_real_negative_certificates(self):
        cover = CellCover()
        for c in range(1, 21):
            self.assertFalse(cover.complete(c))
            for k in [21]+list(range(21)):
                self.assertIn(k, cover.needed_stations([c]))
                cover.observe_negative(c, cover.stations[k])
            self.assertTrue(cover.complete(c))
            self.assertEqual(cover.needed_stations([c]), [])
            cert = cover.certificate(c)
            self.assertEqual(cert['covered'], 108)
            self.assertEqual(len(cert['negative_points']), 22)
            self.assertEqual(len(cert['triangle_witnesses']), 108)
            self.assertTrue(all(0 <= i < 22 for w in cert['triangle_witnesses'].values() for i in w))

    def test_insufficient_points_and_preview_are_not_evidence(self):
        cover = CellCover()
        cover.observe_negative(1, cover.stations[21])
        cover.observe_negative(1, cover.stations[12])
        before = deepcopy(cover.certificate(1))
        gain = cover.preview_negative(1, cover.stations[13])
        self.assertTrue(gain.any())
        self.assertEqual(before, cover.certificate(1))
        self.assertFalse(cover.complete(1))
        cover.observe_negative(1, cover.stations[13])
        self.assertTrue(np.array_equal(cover.covered[1], gain))
        self.assertFalse(cover.preview_negative(1, cover.stations[13]).any())

    def test_extra_actual_negative_point_can_certify_a_cell(self):
        cover = CellCover()
        ids = next(i for i, w in enumerate(cover.pre_witnesses)
                   if len(w) >= 4 and 21 not in w)
        # The pre-proof hull has interior cells: a nonstation point near one
        # witness is valid only after it has actually been observed negative.
        witnesses = cover.stations[list(cover.pre_witnesses[ids])]
        center = witnesses.mean(axis=0)
        extras = witnesses*.999+center*.001
        self.assertFalse(any(np.array_equal(extras[0], q) for q in cover.stations))
        for q in extras:
            cover.observe_negative(1, q)
        self.assertTrue(cover.covered[1][ids])
        for i in cover.witnesses[1][ids]:
            self.assertTrue(any(np.array_equal(cover.negative[1][i], q) for q in extras))

    def test_outward_boundary_source_cannot_be_certified_absent(self):
        cover = CellCover()
        source = np.array([1800., 0.])
        # Inside stations are shadowed by an outward-facing source. An omni
        # disk-only negative inference would be false here.
        negative_count = 0
        positive_count = 0
        for q in cover.stations:
            receives = np.linalg.norm(q-source) <= 1000 and q[0]-source[0] >= 0
            if receives:
                positive_count += 1
            else:
                cover.observe_negative(1, q)
                negative_count += 1
        self.assertGreater(positive_count, 0)
        self.assertGreater(negative_count, 2)
        self.assertFalse(cover.complete(1))
        for t in np.flatnonzero(cover.covered[1]):
            triangle = cover.triangles[t]
            edges = np.roll(triangle, -1, axis=0)-triangle
            rel = source-triangle
            self.assertFalse(np.all(edges[:, 0]*rel[:, 1]-edges[:, 1]*rel[:, 0] >= -1e-7))

    def test_absent_world_adapter_checks_ledger_and_rejects_fabrication(self):
        # Zero-source toy solely tests the certificate/ledger adapter; it is
        # not a contest performance scene or a claimed stress-test result.
        cover = CellCover()
        events = []
        position = np.zeros(2)
        channel = 1
        time_s = 0.
        for q in cover.stations:
            for c in range(1, 21):
                time_s += np.linalg.norm(q-position)/5+5+int(c != channel)
                events.append(dict(action='measure', position=q.tolist(), channel=c,
                                   measure_result='no_signal', time_s=float(time_s)))
                cover.observe_negative(c, q)
                position, channel = q, c
        certificate = cover.geometry_certificate()
        certificate.update(cleared=[], absent=list(range(1, 21)),
                           channels={str(c): cover.certificate(c) for c in range(1, 21)})
        result = dict(complete=True, cleared=[], certificate=certificate)
        report = validate_cell_run([], events, [], result, cover)
        self.assertTrue(report['passed'])
        self.assertEqual(report['triangle_certificates'], 2160)
        self.assertEqual(report['actual_measurement_stations'], 22)
        corrupted = deepcopy(result)
        corrupted['certificate']['channels']['1']['negative_points'][0][0] += .01
        with self.assertRaises(AssertionError):
            validate_cell_run([], events, [], corrupted, cover)
        corrupted = deepcopy(result)
        corrupted['certificate']['stations'] = corrupted['certificate']['vertices']
        with self.assertRaises(AssertionError):
            validate_cell_run([], events, [], corrupted, cover)


if __name__ == '__main__':
    unittest.main()
