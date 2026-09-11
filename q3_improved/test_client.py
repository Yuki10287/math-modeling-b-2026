"""客户端本地替身校验。只使用操作系统分配的临时端口，不访问官方 2026 端口。"""
from __future__ import annotations

import collections
import json
import math
import socket
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from official_client import (ActionRejected, ClientError, DeadlineBudgetExceeded,
                             OfficialClient, TransportFailure, run_session)


class Scenario:
    def __init__(self):
        self.cache = {}
        self.requests = []
        self.executions = collections.Counter()
        self.position = (0.0, 0.0)
        self.channel = 1
        self.virtual = 0.0
        self.cleared = set()
        self.reject_next = False
        self.drop_first_measure = False
        self.close_paths = set()
        self.measure_delay = 0.0
        self.remaining = 1200
        self.active = False

    def dispatch(self, path, raw):
        request = json.loads(raw)
        self.requests.append((path, raw, request))
        if path in self.close_paths:
            return None, False
        key = request["request_id"]
        if key in self.cache:
            old_path, old_raw, response = self.cache[key]
            if path != old_path or raw != old_raw:
                return {"conflict": True}, False
            return response, False
        if self.reject_next:
            self.reject_next = False
            return {"accepted": False, "virtual_time_s": 0, "real_timestamp_ms": 0}, False
        self.executions[path] += 1
        if path == "/enter":
            self.active = True
        elif path in ("/measure", "/clear"):
            point = tuple(request["position"][key] for key in ("x", "y"))
            self.virtual += math.dist(self.position, point) / 5.0
            self.position = point
            channel = request["channel"]
            if path == "/measure":
                self.virtual += 5 + int(self.channel != channel)
                self.channel = channel
            else:
                self.virtual += 3 if channel in self.cleared else 5
        elif path == "/exit":
            self.active = False
        response = {"accepted": True, "real_timestamp_ms": int(time.time() * 1000),
                    "virtual_time_s": self.virtual}
        if path == "/enter":
            response.update(max_virtual_duration_s=360000, max_real_duration_s=1200,
                            remaining_real_duration_s=self.remaining)
        elif path == "/measure":
            response.update(measure_result="direction", svd_deg=45.0)
        elif path == "/clear":
            response.update(clear_result="no_target_in_range" if channel in self.cleared else "success")
            self.cleared.add(channel)
        else:
            response["exit_reason"] = "user_exit"
        self.cache[key] = (path, raw, response)
        drop = self.drop_first_measure and path == "/measure"
        if drop:
            self.drop_first_measure = False
        return response, drop


class LocalSimulator:
    def __init__(self, scenario=None):
        self.scenario = scenario or Scenario()
        scenario = self.scenario

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                response, drop = scenario.dispatch(self.path, raw)
                if response is None or drop:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    self.connection.close()
                    return
                if self.path == "/measure" and scenario.measure_delay:
                    time.sleep(scenario.measure_delay)
                body = json.dumps(response).encode("utf-8")
                try:
                    self.send_response(409 if response.get("conflict") else 200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

        # 使用临时端口，不采用官方端口。若本地监听权限被拒绝，直接让测试失败。
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        port = self.server.server_address[1]
        if port == 2026:
            self.server.server_close()
            raise RuntimeError("临时端口意外等于官方端口，停止测试。")
        self.url = f"http://127.0.0.1:{port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       kwargs={"poll_interval": 0.01}, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class ClientTests(unittest.TestCase):
    def setUp(self):
        base = Path(__file__).resolve().parent / 'tmp' / 'http-tests'
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="b-q3-client-test-", dir=base)
        assert Path(self.temp.name).resolve().is_relative_to(base.resolve())
        self.addCleanup(self.temp.cleanup)
        self.log = Path(self.temp.name) / "requests.jsonl"

    def client(self, server, **kwargs):
        api = OfficialClient("local-test-team", server.url, self.log, **kwargs)
        self.addCleanup(api.close_log)
        return api

    def records(self):
        return [json.loads(line) for line in self.log.read_text(encoding="utf-8").splitlines()]

    def test_rejected_action_preserves_position_channel_time(self):
        with LocalSimulator() as server:
            api = self.client(server)
            api.enter()
            api.measure((3, 4), 2)
            before = (tuple(api.position), api.channel, api.time_s)
            server.scenario.reject_next = True
            with self.assertRaises(ActionRejected):
                api.measure((100, 100), 10)
            self.assertEqual((tuple(api.position), api.channel, api.time_s), before)
            self.assertFalse(api.can_exit)
            rows = self.records()
            self.assertEqual(rows[-2]["response"]["accepted"], False)
            self.assertEqual(rows[-1]["type"], "client_error")

    def test_response_loss_retries_identical_body_and_executes_once(self):
        scenario = Scenario()
        scenario.drop_first_measure = True
        with LocalSimulator(scenario) as server:
            api = self.client(server)
            api.enter()
            api.measure((3, 4), 2)
            calls = [r for r in scenario.requests if r[0] == "/measure"]
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertEqual(scenario.executions["/measure"], 1)
            self.assertAlmostEqual(api.time_s, 7.0)
            self.assertEqual(api.channel, 2)
            self.assertTrue(api.can_exit)
            api.clear((3, 4), 3)
            self.assertEqual(api.channel, 2)
            self.assertAlmostEqual(api.time_s, 12.0)
            api.exit()
            self.assertFalse(api.can_exit)
            accepted = [r for r in self.records() if "response" in r and r["response"].get("accepted")]
            self.assertEqual(len({r["request"]["request_id"] for r in accepted}), len(accepted))

    def test_network_timeout_is_limited_by_remaining_budget(self):
        scenario = Scenario()
        scenario.measure_delay = 0.40
        with LocalSimulator(scenario) as server:
            api = self.client(server, http_timeout_s=5, max_attempts=3, exit_reserve_s=0)
            api.enter()
            api.deadline = time.monotonic() + 0.15
            start = time.monotonic()
            with self.assertRaises(DeadlineBudgetExceeded) as caught:
                api.measure((0, 0), 1)
            elapsed = time.monotonic() - start
            self.assertLess(elapsed, 0.35, "网络等待未被剩余预算截短。")
            self.assertIsNotNone(caught.exception.__cause__)
            self.assertFalse(api.can_exit)
            self.assertEqual(api.time_s, 0)

    def test_closed_test_does_not_trigger_blind_exit_and_saves_trace(self):
        scenario = Scenario()
        scenario.close_paths.add("/measure")
        with LocalSimulator(scenario) as server:
            api = self.client(server)

            def broken_solver(api, **options):
                options["trace"].append({"phase": "before_measure"})
                api.measure((0, 0), 1)

            with self.assertRaises(TransportFailure) as caught:
                run_session(api, broken_solver)
            self.assertIsNotNone(caught.exception.__cause__)
            self.assertNotIn("/exit", [r[0] for r in scenario.requests])
            self.assertEqual(json.loads(api.trace_path.read_text(encoding="utf-8")),
                             [{"phase": "before_measure"}])
            self.assertTrue(any(r.get("type") == "client_error" for r in self.records()))

    def test_original_solver_error_survives_cleanup_error(self):
        scenario = Scenario()
        scenario.close_paths.add("/exit")
        original = ValueError("原始几何异常")
        with LocalSimulator(scenario) as server:
            api = self.client(server)

            def broken_solver(api, **options):
                options["trace"].append({"phase": "geometry_failure"})
                raise original

            with self.assertRaises(ValueError) as caught:
                run_session(api, broken_solver)
            self.assertIs(caught.exception, original)
            self.assertTrue(any("结束测试时另有异常" in note for note in original.__notes__))
            self.assertEqual(json.loads(api.trace_path.read_text(encoding="utf-8"))[0]["phase"],
                             "geometry_failure")
            exit_calls = [r for r in scenario.requests if r[0] == "/exit"]
            self.assertEqual(len({r[1] for r in exit_calls}), 1)

    def test_user_exit_is_not_a_completion_certificate(self):
        with LocalSimulator() as server:
            api = self.client(server)

            def incomplete_solver(api, **options):
                self.assertEqual(options["schedule"], "dynamic")
                self.assertEqual(options["max_local_steps"], 30)
                return {"complete": False, "completion_certificate": {"status": "incomplete"}}

            summary = run_session(api, incomplete_solver)
            self.assertFalse(summary["complete"])
            self.assertTrue(summary["exit_accepted"])
            self.assertIsNone(summary["official_program_runtime_s"])
            self.assertGreaterEqual(summary["local_session_wall_time_s"], summary["solver_wall_time_s"])
            self.assertIn("HTTP", summary["local_session_wall_time_definition"])

    def test_completion_requires_certificate_and_observed_clear_successes(self):
        with LocalSimulator() as server:
            api = self.client(server)

            def complete_solver(api, **options):
                for channel in range(1, 17):
                    api.clear((0, 0), channel)
                return {"complete": True, "completion_certificate": {
                    'valid': True, 'basis': 'count_upper_bound', "cleared_channels": list(range(1, 17)),
                    "absent_channels": [], "negative_scan_indices": {}}}

            summary = run_session(api, complete_solver)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["successful_clear_count"], 16)
            self.assertEqual(summary["total_virtual_s"], 80)
            self.assertTrue(summary["exit_accepted"])

    def test_false_completion_claim_is_rejected(self):
        with LocalSimulator() as server:
            api = self.client(server)

            def false_solver(api, **options):
                return {"complete": True, "completion_certificate": {
                    'valid': True, 'basis': 'count_upper_bound', "cleared_channels": list(range(1, 17))}}

            with self.assertRaises(ClientError):
                run_session(api, false_solver)
            self.assertEqual(scenario_exit_count(server), 1)
            self.assertFalse(any(r.get("type") == "session_summary" for r in self.records()))

    def test_existing_log_and_trace_are_never_overwritten(self):
        self.log.write_text("old log", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            OfficialClient("team", "http://127.0.0.1:1", self.log)
        self.assertEqual(self.log.read_text(encoding="utf-8"), "old log")
        second = self.log.with_name("second.jsonl")
        trace = Path(str(second) + ".beliefs.json")
        trace.write_text("old trace", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            OfficialClient("team", "http://127.0.0.1:1", second)
        self.assertFalse(second.exists())
        self.assertEqual(trace.read_text(encoding="utf-8"), "old trace")

    def test_failed_enter_also_saves_empty_trace(self):
        scenario = Scenario()
        scenario.close_paths.add("/enter")
        with LocalSimulator(scenario) as server:
            api = self.client(server)
            with self.assertRaises(TransportFailure):
                run_session(api, lambda *args, **kwargs: self.fail("进入失败不得调用算法。"))
            self.assertEqual(json.loads(api.trace_path.read_text(encoding="utf-8")), [])
            self.assertNotIn("/exit", [r[0] for r in scenario.requests])

    def test_coverage_claim_requires_actual_negative_measurements(self):
        with LocalSimulator() as server:
            api = self.client(server)
            def false_solver(api, **options):
                return {'complete': True, 'completion_certificate': {
                    'valid': True, 'basis': 'seven_point_coverage', 'cleared_channels': [],
                    'absent_channels': list(range(1, 21)),
                    'negative_scan_indices': {str(c): list(range(7)) for c in range(1, 21)}}}
            with self.assertRaises(ClientError):
                run_session(api, false_solver)

    def test_actual_solver_over_temporary_http_arena(self):
        from environment import LocalArena
        from benchmark import multi_case
        from solver import solve_multi

        class ArenaScenario(Scenario):
            def __init__(self):
                super().__init__()
                self.arena = LocalArena(multi_case(3000), seed=3000, field='hash')
            def dispatch(self, path, raw):
                if path not in ('/measure', '/clear'):
                    return super().dispatch(path, raw)
                request = json.loads(raw)
                self.requests.append((path, raw, request))
                key = request['request_id']
                if key in self.cache:
                    old_path, old_raw, response = self.cache[key]
                    if path != old_path or raw != old_raw:
                        return {'conflict': True}, False
                    return response, False
                self.executions[path] += 1
                q = [request['position']['x'], request['position']['y']]
                feedback = (self.arena.measure if path == '/measure' else self.arena.clear)(q, request['channel'])
                self.virtual = self.arena.time_s
                response = {'accepted': True, 'real_timestamp_ms': int(time.time()*1000),
                            'virtual_time_s': self.virtual, **feedback}
                self.cache[key] = path, raw, response
                drop = self.drop_first_measure and path == '/measure'
                if drop:
                    self.drop_first_measure = False
                return response, drop

        for schedule in ('baseline', 'safe', 'dynamic'):
            with self.subTest(schedule=schedule):
                self.log = Path(self.temp.name) / f'{schedule}.jsonl'
                scenario = ArenaScenario()
                scenario.drop_first_measure = True
                with LocalSimulator(scenario) as server:
                    api = self.client(server)
                    summary = run_session(api, solve_multi, schedule=schedule)
                    self.assertTrue(summary['complete'])
                    self.assertTrue(summary['completion_certificate']['valid'])
                    self.assertTrue(scenario.arena.evaluation()['all_cleared'])
                    self.assertTrue(summary['exit_accepted'])
                    self.assertEqual(summary['schedule'], schedule)
                    self.assertIn('solver.py', summary['source_sha256'])
                    self.assertAlmostEqual(summary['total_virtual_s'], scenario.arena.time_s)
                    self.assertEqual(scenario.executions['/measure'], scenario.arena.counts['measure'])


def scenario_exit_count(server):
    return sum(r[0] == "/exit" for r in server.scenario.requests)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ClientTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {"test_kind": "temporary_local_http_simulator", "official_simulator_contacted": False,
              "official_test_started": False, "tests_run": result.testsRun,
              "passed": result.wasSuccessful(),
              "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures],
              "errors": [{"test": str(test), "detail": detail} for test, detail in result.errors],
              "scope": ["accepted=false state preservation", "same-id same-body retry idempotency",
                        "remaining network budget", "no blind exit after closure", "trace on exceptions",
                        "original error preservation", "completion certificate", "exclusive logs",
                        "local wall time separated from official summary"]}
    Path(__file__).with_name("http-validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)
