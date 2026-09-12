"""Candidate-specific Q4 sessions plus reused frozen transport tests.

Every live endpoint comes from LocalServer: 127.0.0.1 and an OS-assigned port,
with the official default port explicitly excluded by that fixture.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from benchmark import LocalArena, public_api
from local_http_fixture import ArenaScenario, LocalServer
from polar_cover import PolarCover
from shared import ROOT
from stress_check import EndpointArena, stress_cases
from validation import validate_run
import q4_official_client as frozen_client
import q4_share25_client as candidate_client
import task_sharing_solver
import test_q4_client as protocol_tests


HERE = Path(__file__).resolve().parent


class Share25ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = HERE/'tmp'/'share25-http-tests'/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.log = self.directory/'requests.jsonl'
        self.scenario = ArenaScenario(LocalArena([
            dict(channel=2, position=[3., 4.], radius=1000., orientation=0.)]))

    def client(self, server, **options):
        self.assertEqual(server.server.server_address[0], '127.0.0.1')
        self.assertNotIn(server.server.server_address[1], (0, 2026))
        api = candidate_client.Q4Client('local-test-team', server.url, self.log, **options)
        self.addCleanup(api.close_log)
        return api

    def records(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def assert_recorded_identity(self, summary):
        records = self.records()
        starts = [r for r in records if r.get('type') == 'session_start']
        endings = [r for r in records if r.get('type') == 'session_summary']
        self.assertEqual(len(starts), 1)
        self.assertEqual(len(endings), 1)
        snapshot = candidate_client.source_hashes()
        for record in (starts[0], endings[0], summary):
            self.assertEqual(record['problem'], 4)
            self.assertEqual(record['variant'], 'share25')
            self.assertEqual(record['source_sha256'], snapshot)

    def assert_full_replay(self, case, arena, scenario, summary, trace):
        self.assertTrue(summary['complete'])
        self.assertTrue(summary['exit_accepted'])
        self.assertTrue(arena.evaluation()['all_cleared'])
        self.assertEqual(summary['successful_clear_count'], len(case['sources']))
        self.assertAlmostEqual(summary['total_virtual_s'], arena.time_s, places=6)
        self.assertIsNone(summary['official_program_runtime_s'])
        direct = EndpointArena(case['sources'], case['seed'], 'plus_one')
        task_sharing_solver.solve_multi(public_api(direct), variant='share25')
        self.assertEqual(arena.events, direct.events)
        self.assertEqual(scenario.executions['/measure'], arena.counts['measure'])
        mesh = PolarCover()
        audit = validate_run(case['sources'], arena.events, trace, summary,
                             mesh.stations, mesh.indices)
        self.assertTrue(audit['passed'])
        self.assert_recorded_identity(summary)
        evidence = dict(problem=4, variant='share25', case=case['name'],
                        local_only=True, official_simulator_contacted=False,
                        direct_http_actions_equal=True, summary=summary, audit=audit,
                        events=arena.events, sources=case['sources'])
        with (self.directory/f'{case["name"]}-evidence.json').open('x', encoding='utf-8') as stream:
            json.dump(evidence, stream, ensure_ascii=False, indent=2)

    def test_reused_surface_and_actual_dependency_hashes(self):
        for name in ('Q4Client', 'ClientError', 'solver_api', 'check_completion'):
            self.assertIs(getattr(candidate_client, name), getattr(frozen_client, name))
        snapshot = candidate_client.source_hashes()
        expected = {'shared.py', 'directional_cover.py', 'polar_cover.py', 'localization.py',
                    'solver.py', 'route_planning.py', 'guarded_policy.py', 'geometry.py',
                    'official_client.py', 'q4_official_client.py', 'q4_share25_client.py',
                    'task_sharing_solver.py'}
        self.assertTrue(expected <= {Path(name).name for name in snapshot})
        for name, digest in snapshot.items():
            self.assertEqual(digest, hashlib.sha256((ROOT/name).read_bytes()).hexdigest())

    def test_default_share25_dispatch(self):
        # Real complete CLI sessions are independently exercised by
        # validate_share25_http.py; here isolate the default entry selection.
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            with patch.object(task_sharing_solver, 'solve_multi', return_value={
                    'complete': False, 'reason': 'dispatch_fixture'}) as actual:
                summary = candidate_client.run_session(api)
            actual.assert_called_once()
            self.assertEqual(actual.call_args.kwargs['variant'], 'share25')
            public = actual.call_args.args[0]
            self.assertTrue(callable(public.measure) and callable(public.clear))
            self.assertFalse(hasattr(public, 'time_s'))
            self.assertFalse(hasattr(public, 'detected_channels'))
            self.assertTrue(api._log_stream.closed)
            self.assertFalse(summary['complete'])
            self.assertTrue(summary['exit_accepted'])
            self.assert_recorded_identity(summary)

    def test_incomplete_candidate_exit_is_not_reported_as_completion(self):
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            def incomplete(public, **options):
                self.assertEqual(options['variant'], 'share25')
                for hidden in ('negative_points', 'time_s', 'detected_channels',
                               'source_count', 'cleared_channels', 'scan_points'):
                    self.assertFalse(hasattr(public, hidden))
                options['trace'].append(dict(phase='incomplete_candidate'))
                return dict(complete=False, reason='candidate_local_test')
            summary = candidate_client.run_session(api, incomplete)
            self.assertFalse(summary['complete'])
            self.assertTrue(summary['exit_accepted'])
            self.assertEqual(summary['successful_clear_count'], 0)
            self.assertIsNone(summary['average_virtual_s_per_cleared'])
            self.assertTrue(api._log_stream.closed)
            self.assert_recorded_identity(summary)

    def test_candidate_exceptions_keep_original_trace_and_cleanup(self):
        for index, (original, fail_exit) in enumerate([
                (ValueError('candidate geometry failure'), False),
                (ValueError('candidate failure with exit failure'), True),
                (KeyboardInterrupt('local interruption fixture'), False)]):
            with self.subTest(index=index):
                self.log = self.directory/f'exception-{index}.jsonl'
                scenario = ArenaScenario(LocalArena([]))
                if fail_exit:
                    scenario.close_paths.add('/exit')
                with LocalServer(scenario) as server:
                    api = self.client(server)
                    def broken(public, **options):
                        self.assertEqual(options['variant'], 'share25')
                        options['trace'].append(dict(phase='candidate_failure', fixture=index))
                        raise original
                    with self.assertRaises(type(original)) as caught:
                        candidate_client.run_session(api, broken)
                    self.assertIs(caught.exception, original)
                    self.assertTrue(api._log_stream.closed)
                    trace = json.loads(api.trace_path.read_text(encoding='utf-8'))
                self.assertEqual(trace, [dict(phase='candidate_failure', fixture=index)])
                exit_ids = {r[2]['request_id'] for r in scenario.requests if r[0] == '/exit'}
                self.assertEqual(len(exit_ids), 1)
                self.assertTrue(any(r.get('type') == 'client_error' for r in self.records()))
                self.assertFalse(any(r.get('type') == 'session_summary' for r in self.records()))
                if fail_exit:
                    self.assertTrue(original.__notes__)

    def test_failed_enter_or_measure_preserves_log_without_blind_exit(self):
        for path in ('/enter', '/measure'):
            with self.subTest(path=path):
                self.log = self.directory/f'closed-{path[1:]}.jsonl'
                scenario = ArenaScenario(LocalArena([]))
                scenario.close_paths.add(path)
                with LocalServer(scenario) as server:
                    api = self.client(server)
                    def broken(public, **options):
                        if path == '/enter':
                            self.fail('solver called after failed enter')
                        options['trace'].append(dict(phase='before_disconnected_measure'))
                        public.measure([0., 0.], 1)
                    with self.assertRaises(frozen_client.transport.TransportFailure):
                        candidate_client.run_session(api, broken)
                    self.assertTrue(api._log_stream.closed)
                    trace = json.loads(api.trace_path.read_text(encoding='utf-8'))
                expected = [] if path == '/enter' else [dict(phase='before_disconnected_measure')]
                self.assertEqual(trace, expected)
                self.assertNotIn('/exit', [r[0] for r in scenario.requests])
                self.assertEqual(self.records()[0]['variant'], 'share25')

    def test_candidate_preserves_existing_log_and_trace(self):
        with LocalServer(self.scenario) as server:
            self.log.write_text('previous candidate log', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                candidate_client.Q4Client('local-test-team', server.url, self.log)
            self.assertEqual(self.log.read_text(encoding='utf-8'), 'previous candidate log')
            other = self.directory/'existing-trace.jsonl'
            trace = Path(str(other)+'.beliefs.json')
            trace.write_text('previous candidate trace', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                candidate_client.Q4Client('local-test-team', server.url, other)
            self.assertFalse(other.exists())
            self.assertEqual(trace.read_text(encoding='utf-8'), 'previous candidate trace')
            self.assertEqual(self.scenario.requests, [])

    def test_q3_seven_negative_scan_points_cannot_certify_q4_absence(self):
        # One-source toy demonstrates the directional shadow counterexample;
        # it is a transport/certificate test, not a contest performance scene.
        scenario = ArenaScenario(LocalArena([
            dict(channel=1, position=[1800., 0.], radius=1000., orientation=0.)]))
        mesh = PolarCover()
        with LocalServer(scenario) as server:
            api = self.client(server)
            def forged_q3_stop(public, **options):
                angles = np.arange(6)*np.pi/3
                q3_points = np.vstack(([0., 0.], 1200*np.column_stack((np.cos(angles), np.sin(angles)))))
                for q in q3_points:
                    for c in range(1, 21):
                        self.assertEqual(public.measure(q, c)['measure_result'], 'no_signal')
                options['trace'].append(dict(phase='q3_negative_scan_is_insufficient'))
                return dict(complete=True, cleared=[], certificate=dict(
                    basis='directional_triangle_cover', cleared=[], absent=list(range(1, 21)),
                    stations=mesh.stations.tolist(), triangles=mesh.indices.tolist()))
            with self.assertRaises(candidate_client.ClientError):
                candidate_client.run_session(api, forged_q3_stop)
            self.assertTrue(all(len(points) == 7 for points in api.negative_points.values()))
            self.assertTrue(all(not ids for ids in api.negative_scan_indices.values()))
            self.assertEqual(api.cleared_channels, set())
            self.assertTrue(api._log_stream.closed)
        self.assertEqual(scenario.executions['/exit'], 1)
        self.assertFalse(any(r.get('type') == 'session_summary' for r in self.records()))


def protocol_suite(loader):
    # Keep the unchanged protocol cases; complete sessions have a dedicated
    # candidate CLI harness, so do not repeat the old version's full runs.
    return unittest.TestSuite(test for test in loader.loadTestsFromTestCase(protocol_tests.ClientTests)
        if test._testMethodName != 'test_full_cli_and_count_bound_sessions_match_direct_actions')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite([
        protocol_suite(loader),
        loader.loadTestsFromTestCase(Share25ClientTests)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'results'/'share25_http_validation.json')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'Test report already exists; choose a new --out: {args.out}')
    snapshot = candidate_client.source_hashes()
    protocol_snapshot = frozen_client.source_hashes()
    loader = unittest.defaultTestLoader
    suite = load_tests(loader, None, None)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, passed=result.wasSuccessful(),
                  reused_protocol_tests=protocol_suite(loader).countTestCases(),
                  candidate_specific_tests=loader.loadTestsFromTestCase(Share25ClientTests).countTestCases(),
                  local_only=True, official_simulator_contacted=False,
                  source_sha256=snapshot, protocol_source_sha256=protocol_snapshot,
                  source_snapshot_stable=candidate_client.source_hashes() == snapshot,
                  protocol_source_snapshot_stable=frozen_client.source_hashes() == protocol_snapshot,
                  test_sha256={name: hashlib.sha256((HERE/name).read_bytes()).hexdigest()
                               for name in ('test_share25_client.py', 'test_q4_client.py',
                                            'local_http_fixture.py', 'stress_check.py')},
                  failures=[dict(test=str(t), detail=d) for t, d in result.failures],
                  errors=[dict(test=str(t), detail=d) for t, d in result.errors])
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    raise SystemExit(0 if result.wasSuccessful() and report['source_snapshot_stable']
                     and report['protocol_source_snapshot_stable'] else 1)
