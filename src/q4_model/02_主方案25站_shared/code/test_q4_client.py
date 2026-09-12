"""Q4 transport and complete CLI checks, using ephemeral local fixtures only."""
import copy
import hashlib
import json
import subprocess
import sys
import time
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from benchmark import LocalArena, public_api
from joint_solver import solve_multi
from polar_cover import PolarCover
from validation import validate_run
from stress_check import EndpointArena, stress_cases
from local_http_fixture import ArenaScenario, LocalServer
from q4_official_client import Q4Client, check_completion, run_session, source_hashes, transport

HERE = Path(__file__).resolve().parent


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.directory = HERE/'tmp'/'q4-http-tests'/uuid.uuid4().hex
        self.directory.mkdir(parents=True)
        self.log = self.directory/'requests.jsonl'
        self.scenario = ArenaScenario(LocalArena([
            dict(channel=2, position=[3., 4.], radius=1000., orientation=0.)]))

    def client(self, server, **kwargs):
        api = Q4Client('local-test-team', server.url, self.log, **kwargs)
        self.addCleanup(api.close_log)
        return api

    def records(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_rejected_measure_preserves_state_and_evidence(self):
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            api.enter()
            before = (api.position.tolist(), api.channel, api.time_s, copy.deepcopy(api.negative_points))
            self.scenario.reject_next = True
            with self.assertRaises(transport.ActionRejected):
                api.measure([100, 200], 3)
            self.assertEqual(before, (api.position.tolist(), api.channel, api.time_s, api.negative_points))
            self.assertFalse(api.can_exit)

    def test_same_id_retry_executes_once_and_clear_preserves_channel(self):
        self.scenario.drop_first_measure = True
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            api.enter()
            api.measure([3, 4], 1)
            requests = [r for r in self.scenario.requests if r[0] == '/measure']
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0][1], requests[1][1])
            self.assertEqual(self.scenario.executions['/measure'], 1)
            self.assertEqual(len(api.negative_points[1]), 1)
            api.clear([3, 4], 2)
            self.assertEqual(api.channel, 1)
            self.assertEqual(api.time_s, 11)
            self.assertEqual(api.cleared_channels, {2})
            api.exit()

    def test_budget_stops_new_measure_but_allows_reserved_exit(self):
        with LocalServer(self.scenario) as server:
            api = self.client(server, exit_reserve_s=5)
            api.enter()
            api.deadline = time.monotonic()+2
            with self.assertRaises(transport.DeadlineBudgetExceeded):
                api.measure([0, 0], 1)
            self.assertNotIn('/measure', [r[0] for r in self.scenario.requests])
            self.assertTrue(api.can_exit)
            api.exit()

    def test_network_wait_is_bounded_by_remaining_time(self):
        self.scenario.measure_delay = .4
        with LocalServer(self.scenario) as server:
            api = self.client(server, exit_reserve_s=0)
            api.enter()
            api.deadline = time.monotonic()+.15
            started = time.monotonic()
            with self.assertRaises(transport.DeadlineBudgetExceeded):
                api.measure([0, 0], 1)
            self.assertLess(time.monotonic()-started, .35)
            self.assertFalse(api.can_exit)

    def test_closed_interface_preserves_trace_without_blind_exit(self):
        self.scenario.close_paths.add('/measure')
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            def broken(public, **options):
                options['trace'].append({'phase': 'before_measure'})
                public.measure([0, 0], 1)
            with self.assertRaises(transport.TransportFailure):
                run_session(api, broken)
            self.assertNotIn('/exit', [r[0] for r in self.scenario.requests])
            self.assertEqual(json.loads(api.trace_path.read_text(encoding='utf-8')), [{'phase':'before_measure'}])

    def test_failed_enter_never_calls_solver(self):
        self.scenario.close_paths.add('/enter')
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            with self.assertRaises(transport.TransportFailure):
                run_session(api, lambda *a, **k: self.fail('solver called'))
            self.assertEqual(json.loads(api.trace_path.read_text(encoding='utf-8')), [])
            self.assertNotIn('/exit', [r[0] for r in self.scenario.requests])

    def test_malformed_response_adds_no_evidence(self):
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            api.enter()
            self.scenario.malformed_next = True
            with self.assertRaises(transport.ResponseProtocolError):
                api.measure([100, 100], 3)
            self.assertEqual(api.negative_points[3], [])
            self.assertEqual(api.time_s, 0)
            self.assertFalse(api.can_exit)

    def test_original_error_survives_exit_failure(self):
        self.scenario.close_paths.add('/exit')
        original = ValueError('geometry failure')
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            def broken(public, **options):
                options['trace'].append({'phase': 'geometry_failure'})
                raise original
            with self.assertRaises(ValueError) as caught:
                run_session(api, broken)
            self.assertIs(caught.exception, original)
            self.assertTrue(original.__notes__)
            self.assertEqual(len({r[1] for r in self.scenario.requests if r[0]=='/exit'}), 1)

    def test_exit_is_not_a_completion_certificate(self):
        with LocalServer(self.scenario) as server:
            api = self.client(server)
            def incomplete(public, **options):
                self.assertEqual(options['variant'], 'shared')
                self.assertFalse(hasattr(public, 'negative_points'))
                self.assertFalse(hasattr(public, 'time_s'))
                return {'complete': False, 'reason': 'test_incomplete'}
            summary = run_session(api, incomplete)
            self.assertFalse(summary['complete'])
            self.assertTrue(summary['exit_accepted'])
            self.assertIsNone(summary['official_program_runtime_s'])

    def test_forged_clears_are_rejected(self):
        api = SimpleNamespace(cleared_channels=set())
        result = dict(complete=True, cleared=list(range(1,17)), certificate=dict(
            basis='count_upper_bound', cleared=list(range(1,17))))
        with self.assertRaises(transport.ClientError):
            check_completion(result, api)

    def test_absence_requires_actual_feedback_and_detected_cannot_be_absent(self):
        mesh = PolarCover()
        api = SimpleNamespace(cleared_channels=set(), detected_channels=set(),
            negative_points={c:[] for c in range(1,21)})
        result = dict(complete=True, cleared=[], certificate=dict(basis='directional_triangle_cover',
            cleared=[], absent=list(range(1,21)), stations=mesh.stations.tolist(), triangles=mesh.indices.tolist()))
        with self.assertRaises(transport.ClientError):
            check_completion(result, api)
        api.negative_points = {c:mesh.stations.tolist() for c in range(1,21)}
        self.assertTrue(check_completion(result, api))  # Certificate unit fixture, not a contest population.
        api.detected_channels = {1}
        with self.assertRaises(transport.ClientError):
            check_completion(result, api)
        api.detected_channels = set()
        result['certificate']['stations'][0][0] += .01
        with self.assertRaises(transport.ClientError):
            check_completion(result, api)

    def test_existing_outputs_are_never_overwritten(self):
        self.log.write_text('old log', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            Q4Client('local-test-team', 'http://127.0.0.1:1', self.log)
        self.assertEqual(self.log.read_text(encoding='utf-8'), 'old log')
        other = self.directory/'other.jsonl'
        Path(str(other)+'.beliefs.json').write_text('old trace', encoding='utf-8')
        with self.assertRaises(FileExistsError):
            Q4Client('local-test-team', 'http://127.0.0.1:1', other)
        self.assertFalse(other.exists())

    def test_non_loopback_url_is_rejected_before_log_creation(self):
        for url in ['http://example.com:2026', 'http://127.0.0.1:2026/measure',
                    'http://user@127.0.0.1:2026', 'http://127.0.0.1:2026?x=1']:
            with self.assertRaises(ValueError):
                Q4Client('local-test-team', url, self.log)
        self.assertFalse(self.log.exists())

    def test_full_cli_and_count_bound_sessions_match_direct_actions(self):
        for index in (0, 9):
            with self.subTest(case=index):
                case = stress_cases()[index]
                arena = EndpointArena(case['sources'], case['seed'], 'plus_one')
                scenario = ArenaScenario(arena)
                scenario.drop_first_measure = True
                self.log = self.directory/f'full-{index}.jsonl'
                with LocalServer(scenario) as server:
                    if index == 0:
                        completed = subprocess.run([sys.executable, '-X', 'utf8', str(HERE/'q4_official_client.py'),
                            '--robot-id', 'local-test-team', '--url', server.url, '--log', str(self.log)],
                            capture_output=True, text=True, encoding='utf-8', timeout=90)
                        self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
                        records = self.records()
                        summary = next(r for r in records if r.get('type') == 'session_summary')
                        trace = json.loads(Path(str(self.log)+'.beliefs.json').read_text(encoding='utf-8'))
                    else:
                        api = self.client(server)
                        summary = run_session(api)
                        trace = json.loads(api.trace_path.read_text(encoding='utf-8'))
                self.assertTrue(summary['complete'])
                self.assertTrue(summary['exit_accepted'])
                self.assertEqual(summary['certificate']['basis'],
                    'directional_triangle_cover' if index == 0 else 'count_upper_bound')
                self.assertTrue(arena.evaluation()['all_cleared'])
                self.assertAlmostEqual(summary['total_virtual_s'], arena.time_s, places=6)
                direct = EndpointArena(case['sources'], case['seed'], 'plus_one')
                solve_multi(public_api(direct), variant='shared')
                self.assertEqual(arena.events, direct.events)
                self.assertEqual(scenario.executions['/measure'], arena.counts['measure'])
                mesh = PolarCover()
                audit = validate_run(case['sources'], arena.events, trace, summary, mesh.stations, mesh.indices)
                self.assertTrue(audit['passed'])
                evidence = dict(problem=4, case=case['name'], local_only=True,
                    direct_http_actions_equal=True, retry_executed_once=True,
                    summary=summary, audit=audit, events=arena.events, sources=case['sources'])
                (self.directory/f'full-{index}-evidence.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')


if __name__ == '__main__':
    snapshot = source_hashes()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(tests_run=result.testsRun, passed=result.wasSuccessful(), local_only=True,
        official_simulator_contacted=False, source_sha256=snapshot,
        source_snapshot_stable=source_hashes()==snapshot,
        test_sha256={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in
                     ('test_q4_client.py','local_http_fixture.py','stress_check.py')},
        failures=[dict(test=str(t), detail=d) for t,d in result.failures],
        errors=[dict(test=str(t), detail=d) for t,d in result.errors])
    output = HERE/'results'/'q4_http_validation.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    raise SystemExit(0 if result.wasSuccessful() and report['source_snapshot_stable'] else 1)
