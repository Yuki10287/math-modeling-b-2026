"""Exercise the real Q3 CLI using a guarded, temporary local HTTP simulator.

Run from the repository root:
    python -X utf8 src/q3_model_v2/tests/test_direct_entry.py

Only the parent test imports archived validation helpers. Client subprocesses
run official_client.py directly, with PYTHONPATH removed. A temporary user-site
contains only a network audit hook, never model modules or import-path fixes.
Artifacts remain under repository tmp/; this is entry regression, not a new
independent performance experiment or an official practice.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
import unittest


MODEL = Path(__file__).resolve().parents[1]
ROOT = MODEL.parents[1]
ARCHIVED_CODE = MODEL / "00_主方案_lean" / "code"
RESULTS = MODEL / "00_主方案_lean" / "results"
RUNTIME_FILES = (
    "official_client.py", "geometry.py", "solver.py", "belief_model.py",
    "local_policy.py", "coverage_model.py", "scan_planning.py", "recovery.py",
    "v1_solver.py", "baseline_solver.py",
)

# These paths are confined to this parent test process. Actual client children
# get neither PYTHONPATH nor an artificial solver search path.
sys.path[:0] = [str(MODEL), str(ARCHIVED_CODE)]
from benchmark import multi_case
from environment import LocalArena, audit_trace
from solver import solve_multi
from test_client import LocalSimulator, Scenario


GUARD = '''"""Test-only network restriction, loaded by normal Python site startup."""
import json
import os
import sys

_port = int(os.environ["Q3_TEST_ALLOWED_PORT"])
_path = os.environ["Q3_TEST_NETWORK_AUDIT"]

def _record(row):
    with open(_path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\\n")

def _guard(event, args):
    if event == "socket.getaddrinfo":
        host, port = args[:2]
    elif event in ("socket.connect", "socket.sendto"):
        address = args[-1]
        host, port = address[:2] if isinstance(address, tuple) else (address, None)
    else:
        return
    allowed = host == "127.0.0.1" and port == _port and _port not in (0, 2026)
    _record({"event": event, "host": host, "port": port, "allowed": allowed})
    if not allowed:
        raise PermissionError("Q3_TEST_NETWORK_BLOCKED: only the temporary test server is permitted")

sys.addaudithook(_guard)
_record({"event": "guard_installed", "allowed_port": _port,
         "pythonpath_present": "PYTHONPATH" in os.environ})
sys._q3_network_guard_installed = True
'''


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


class ArenaScenario(Scenario):
    """Use the already-established seed 1000/hash interface regression case."""

    def __init__(self):
        super().__init__()
        self.arena = LocalArena(multi_case(1000), seed=1000, field="hash")

    def dispatch(self, path, raw):
        if path not in ("/measure", "/clear"):
            return super().dispatch(path, raw)
        request = json.loads(raw)
        self.requests.append((path, raw, request))
        key = request["request_id"]
        if key in self.cache:
            old_path, old_raw, response = self.cache[key]
            return (response if path == old_path and raw == old_raw else {"conflict": True}), False
        self.executions[path] += 1
        point = [request["position"][axis] for axis in ("x", "y")]
        action = self.arena.measure if path == "/measure" else self.arena.clear
        feedback = action(point, request["channel"])
        self.virtual = self.arena.time_s
        response = {"accepted": True, "real_timestamp_ms": int(time.time() * 1000),
                    "virtual_time_s": self.virtual, **feedback}
        self.cache[key] = path, raw, response
        return response, False


class DirectEntryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temp_root = ROOT / "tmp"
        temp_root.mkdir(exist_ok=True)
        cls.artifacts = Path(tempfile.mkdtemp(prefix="q3-direct-entry-", dir=temp_root))
        cls.report = {
            "kind": "real_direct_cli_local_regression",
            "official_simulator_contacted": False,
            "seed": 1000, "field": "hash", "independent_layouts": 1,
            "note": "Reuses an existing interface case; not new performance evidence.",
            "python_version": sys.version,
            "checks": [],
        }
        cls.hashes = {name: sha256(MODEL / name) for name in RUNTIME_FILES}
        frozen = json.loads((RESULTS / "main_solution_freeze.json").read_text(encoding="utf-8"))
        for name, digest in cls.hashes.items():
            if digest != frozen["file_sha256"][name]:
                raise AssertionError(f"Frozen runtime source changed: {name}")
        cls.report["frozen_runtime_sha256"] = cls.hashes
        cls.fixture_hashes = {name: sha256(ARCHIVED_CODE / name)
                              for name in ("benchmark.py", "environment.py")}
        for name, digest in cls.fixture_hashes.items():
            if digest != frozen["file_sha256"][name]:
                raise AssertionError(f"Frozen local fixture changed: {name}")
        cls.report["frozen_local_fixture_sha256"] = cls.fixture_hashes
        cls.userbase = cls.artifacts / "network_guard_userbase"
        guard_site = Path(sysconfig.get_path(
            "purelib", scheme=sysconfig.get_preferred_scheme("user"),
            vars={"userbase": str(cls.userbase)}))
        guard_site.mkdir(parents=True)
        (guard_site / "usercustomize.py").write_text(GUARD, encoding="utf-8")
        cls.foreign_cwd = cls.artifacts / "unrelated-working-directory"
        cls.foreign_cwd.mkdir()
        # This reference is the frozen algorithm over its local public API.
        # Its ten runtime bytes must match the historical freeze before use.
        cls.reference = LocalArena(multi_case(1000), seed=1000, field="hash")
        cls.reference_trace = []
        cls.reference_outcome = solve_multi(
            cls.reference, policy="time", schedule="lean", trace=cls.reference_trace)
        if not cls.reference_outcome["complete"]:
            raise AssertionError("Frozen-source reference did not complete")
        write_json(cls.artifacts / "reference.json", {
            "metrics": cls.reference.evaluation(), "events": cls.reference.events,
            "trace": cls.reference_trace, "outcome": cls.reference_outcome,
        })

    def run_child(self, name, args, *, cwd, port=0, expect_guard=True):
        audit_path = self.artifacts / f"{name}.network.jsonl"
        environment = os.environ.copy()
        for key in ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSAFEPATH"):
            environment.pop(key, None)
        # Normal user-site loading installs only a network restriction. This
        # site has no solver modules and never changes sys.path itself.
        environment.update(PYTHONUSERBASE=str(self.userbase),
                           Q3_TEST_ALLOWED_PORT=str(port),
                           Q3_TEST_NETWORK_AUDIT=str(audit_path),
                           PYTHONDONTWRITEBYTECODE="1", NO_PROXY="127.0.0.1", no_proxy="127.0.0.1")
        for key in list(environment):
            if key.lower() in ("http_proxy", "https_proxy", "all_proxy"):
                environment.pop(key)
        command = [sys.executable, "-X", "utf8", *map(str, args)]
        result = subprocess.run(command, cwd=cwd, env=environment,
                                capture_output=True, text=True, encoding="utf-8", timeout=300)
        (self.artifacts / f"{name}.stdout.txt").write_text(result.stdout, encoding="utf-8")
        (self.artifacts / f"{name}.stderr.txt").write_text(result.stderr, encoding="utf-8")
        if not expect_guard:
            self.assertFalse(audit_path.exists(), "The no-user-site control unexpectedly loaded the guard")
            return result, []
        self.assertTrue(audit_path.is_file(), "Python did not load the mandatory network restriction")
        audit = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(audit[0]["event"], "guard_installed")
        self.assertFalse(audit[0]["pythonpath_present"])
        return result, audit

    def audit_events(self, arena):
        sources = {source["channel"]: source for source in multi_case(1000)}
        removed, position, channel, total, max_error = set(), (0.0, 0.0), 1, 0.0, 0.0
        for event in arena.events:
            total += math.dist(position, event["position"]) / 5.0
            position = event["position"]
            if event["action"] == "measure":
                total += 5 + (channel != event["channel"])
                channel = event["channel"]
            else:
                source = sources.get(event["channel"])
                expected = (source is not None and event["channel"] not in removed
                            and math.dist(source["position"], position) <= 20)
                self.assertEqual(event["clear_result"] == "success", expected)
                total += 5 if expected else 3
                if expected:
                    removed.add(event["channel"])
            max_error = max(max_error, abs(total - event["time_s"]))
        self.assertEqual(removed, set(sources))
        self.assertLess(max_error, 1e-7)
        return max_error

    def check_cli(self, name, entry, cwd):
        scenario = ArenaScenario()
        log = self.artifacts / f"{name}.jsonl"
        with LocalSimulator(scenario) as server:
            port = server.server.server_address[1]
            self.assertNotEqual(port, 2026)
            result, audit = self.run_child(name, [entry, "--robot-id", "local-direct-entry-test",
                "--url", server.url, "--log", log], cwd=cwd, port=port)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("完整清除证书：通过", result.stdout)
        self.assertIn("退出请求：已接受", result.stdout)
        self.assertTrue(all(row["allowed"] for row in audit[1:]))
        self.assertGreater(len(audit), 1)
        self.assertEqual(scenario.executions["/enter"], 1)
        self.assertEqual(scenario.executions["/exit"], 1)
        self.assertFalse(scenario.active)
        records = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        self.assertFalse(any(row.get("type") == "client_error" for row in records))
        summaries = [row for row in records if row.get("type") == "session_summary"]
        self.assertEqual(len(summaries), 1)
        summary = summaries[0]
        self.assertTrue(summary["complete"] and summary["exit_accepted"])
        self.assertEqual((summary["policy"], summary["schedule"]), ("time", "lean"))
        self.assertEqual(summary["source_sha256"], self.hashes)
        self.assertIsNone(summary["official_program_runtime_s"])
        self.assertEqual(summary["completion_certificate"], self.reference_outcome["completion_certificate"])
        self.assertEqual(summary["successful_clear_count"], len(multi_case(1000)))
        self.assertEqual(scenario.arena.events, self.reference.events)
        self.assertEqual(summary["total_virtual_s"], self.reference.time_s)
        actions = [row for row in records if row.get("path") in ("/measure", "/clear")]
        self.assertEqual(len(actions), len(scenario.arena.events))
        for record, event in zip(actions, scenario.arena.events):
            self.assertEqual(record["path"], "/" + event["action"])
            self.assertEqual(record["request"]["channel"], event["channel"])
            self.assertEqual([record["request"]["position"][axis] for axis in ("x", "y")], event["position"])
            self.assertEqual(record["response"]["virtual_time_s"], event["time_s"])
        trace = json.loads(Path(str(log) + ".beliefs.json").read_text(encoding="utf-8"))
        self.assertTrue(trace)
        self.assertEqual(trace, self.reference_trace)
        self.assertEqual(audit_trace(scenario.arena, trace), [])
        independent_error = self.audit_events(scenario.arena)
        self.report["checks"].append({
            "name": name, "passed": True, "entry": str(entry),
            "cwd": str(Path(cwd).relative_to(ROOT)),
            "action_count": len(actions), "trace_records": len(trace),
            "cleared": summary["successful_clear_count"], "total_virtual_s": summary["total_virtual_s"],
            "all_actions_and_trace_equal_frozen_reference": True,
            "completion_certificate_valid": True, "normal_exit": True,
            "source_hashes_equal_frozen_reference": True,
            "independent_max_time_error_s": independent_error,
            "non_test_network_attempts": sum(not row["allowed"] for row in audit[1:]),
        })

    def test_01_relative_entry_from_repository_root(self):
        self.check_cli("relative_from_root", Path("src/q3_model_v2/official_client.py"), ROOT)

    def test_02_absolute_entry_from_other_working_directory(self):
        self.check_cli("absolute_from_other_cwd", MODEL / "official_client.py", self.foreign_cwd)

    def test_03_only_ten_runtime_files_in_empty_directory(self):
        standalone = self.artifacts / "ten-runtime-files-only"
        standalone.mkdir()
        for name in RUNTIME_FILES:
            shutil.copyfile(MODEL / name, standalone / name)
        self.assertEqual(sorted(path.name for path in standalone.iterdir()), sorted(RUNTIME_FILES))
        imports = "; ".join(f"import {Path(name).stem}" for name in RUNTIME_FILES)
        result, audit = self.run_child("ten_file_import", ["-c", imports], cwd=standalone)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(audit), 1, "Imports must make no network attempts")
        result, audit = self.run_child("ten_file_help", [standalone / "official_client.py", "--help"],
                                       cwd=self.foreign_cwd)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--robot-id", result.stdout)
        self.assertEqual(len(audit), 1, "Help must make no network attempts")
        self.report["checks"].append({"name": "ten_runtime_files_import_and_help", "passed": True,
                                      "runtime_file_count": 10, "network_attempts": 0})

    def test_04_network_restriction_blocks_wrong_targets_before_connection(self):
        probe = ("import socket, sys\n"
                 "if not getattr(sys, '_q3_network_guard_installed', False):\n"
                 "    raise RuntimeError('Q3_TEST_GUARD_NOT_INSTALLED')\n"
                 "socket.create_connection((HOST, PORT), timeout=1)")
        for name, host, port in (("official_port_blocked", "127.0.0.1", 2026),
                                 ("other_loopback_blocked", "127.0.0.1", 1),
                                 ("external_address_blocked", "192.0.2.1", 80)):
            with self.subTest(name=name):
                code = probe.replace("HOST", repr(host)).replace("PORT", str(port))
                result, audit = self.run_child(name, ["-c", code], cwd=self.foreign_cwd)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Q3_TEST_NETWORK_BLOCKED", result.stderr)
                self.assertEqual(audit[1]["event"], "socket.getaddrinfo")
                self.assertFalse(audit[1]["allowed"])
                self.assertFalse(any(row["event"] == "socket.connect" for row in audit))
        # If a device disables user-site startup, the probe fails before even
        # resolving an address. It must never depend on a later parent check.
        code = probe.replace("HOST", repr("127.0.0.1")).replace("PORT", "2026")
        result, _ = self.run_child("guard_unavailable_fails_closed", ["-s", "-c", code],
                                   cwd=self.foreign_cwd, expect_guard=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError: Q3_TEST_GUARD_NOT_INSTALLED", result.stderr)
        self.report["checks"].append({"name": "network_restriction_negative_controls", "passed": True,
                                      "blocked_before_socket_connect": 3,
                                      "missing_guard_fails_before_address_resolution": True})

    @classmethod
    def tearDownClass(cls):
        cls.report["runtime_hashes_unchanged"] = cls.hashes == {
            name: sha256(MODEL / name) for name in RUNTIME_FILES}
        write_json(cls.artifacts / "validation.json", cls.report)
        print(f"Local-only direct-entry artifacts: {cls.artifacts}")
        if not cls.report["runtime_hashes_unchanged"]:
            raise AssertionError("Runtime source files changed during the regression")


if __name__ == "__main__":
    unittest.main(verbosity=2)
