"""Two complete share25 CLI sessions against owned, ephemeral loopback fixtures.

This harness never uses the official port 2026. It starts its own fixture on
127.0.0.1:0, supplies that explicit address to a real CLI subprocess, and stops
only that fixture afterward. Source truth is available to the fixture/auditor,
never to the client subprocess or the solver's public action interface.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from urllib.parse import urlsplit

from benchmark import public_api
from local_http_fixture import ArenaScenario, LocalServer
from polar_cover import PolarCover
from stress_check import EndpointArena, stress_cases
from task_sharing_solver import solve_multi
from validation import validate_run


HERE = Path(__file__).resolve().parent
CASE_NAMES = ('outward_min_radius', 'mixed_radius_endpoints')
FIELD = 'plus_one'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def test_hashes():
    paths = [HERE/name for name in ('validate_share25_http.py', 'local_http_fixture.py',
        'stress_check.py', 'benchmark.py', 'validation.py')]
    paths.append(HERE.parent/'q3_model_v2'/'environment.py')
    return {p.relative_to(HERE.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths}


def json_form(value):
    """Normalize only JSON container representation, with no numeric rounding."""
    return json.loads(json.dumps(value, ensure_ascii=False))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run_case(case, directory, source_snapshot):
    directory.mkdir()
    log_path = directory/'client.jsonl'
    arena = EndpointArena(case['sources'], case['seed'], FIELD)
    scenario = ArenaScenario(arena)
    scenario.drop_first_measure = True
    started = time.perf_counter()
    process = None
    command = None
    fixture_url = None
    try:
        with LocalServer(scenario) as server:
            address = urlsplit(server.url)
            require(address.scheme == 'http' and address.hostname == '127.0.0.1'
                    and address.port not in (None, 0, 2026), 'Not an owned ephemeral fixture URL')
            require(address.port == server.server.server_address[1], 'Fixture port mismatch')
            fixture_url = server.url
            command = [sys.executable, '-X', 'utf8', str(HERE/'q4_share25_client.py'),
                '--robot-id', 'local-test-team', '--url', server.url, '--log', str(log_path)]
            # No client in this harness is ever instantiated with its default URL.
            process = subprocess.run(command, cwd=HERE, capture_output=True,
                text=True, encoding='utf-8', timeout=120, check=False)
    finally:
        # Preserve exact event/source values and raw request bodies even on failure.
        write(directory/'server.json', dict(case=case, field=FIELD,
            local_only=True, official_simulator_contacted=False, fixture_url=fixture_url,
            sources=case['sources'], events=arena.events, evaluation=arena.evaluation(),
            requests=[dict(path=path, body_utf8=raw.decode('utf-8'), parsed=decoded)
                      for path,raw,decoded in scenario.requests],
            response_cache={k:dict(path=v[0], body_utf8=v[1].decode('utf-8'), response=v[2])
                            for k,v in scenario.cache.items()},
            executions=dict(scenario.executions), fixture_active_after_client=scenario.active))
        if process is not None:
            (directory/'stdout.txt').write_text(process.stdout, encoding='utf-8')
            (directory/'stderr.txt').write_text(process.stderr, encoding='utf-8')
            write(directory/'subprocess.json', dict(command=command, returncode=process.returncode,
                timeout_s=120, elapsed_wall_s=time.perf_counter()-started,
                owned_loopback_fixture_only=True))
    require(process is not None and process.returncode == 0,
            f'CLI failed: {None if process is None else process.returncode}')
    records = [json.loads(line) for line in log_path.read_text(encoding='utf-8').splitlines()]
    starts = [r for r in records if r.get('type') == 'session_start']
    summaries = [r for r in records if r.get('type') == 'session_summary']
    require(len(starts) == len(summaries) == 1, 'Expected exactly one start and summary')
    summary = summaries[0]
    trace_path = Path(str(log_path)+'.beliefs.json')
    http_trace = json.loads(trace_path.read_text(encoding='utf-8'))
    for record in (starts[0], summary):
        require(record.get('variant') == 'share25' and record.get('problem') == 4,
                'The actual CLI log does not identify problem 4 / share25')
        require(record.get('source_sha256') == source_snapshot, 'Client source snapshot mismatch')
    require(summary.get('complete') is True and summary.get('exit_accepted') is True,
            'CLI did not complete and accept its exit')
    require(not scenario.active, 'Fixture session remained active')
    require(arena.evaluation()['all_cleared'], 'HTTP session did not clear every fixture source')
    require(summary['successful_clear_count'] == len(case['sources']), 'CLI clear count mismatch')
    require(abs(summary['total_virtual_s']-arena.time_s) <= 5.1e-7, 'Rounded interface time mismatch')
    require(summary.get('official_program_runtime_s') is None,
            'Local session claimed an official program runtime')
    expected_basis = ('count_upper_bound' if len(case['sources']) == 16
                      else 'directional_triangle_cover')
    require(summary['certificate']['basis'] == expected_basis, 'Wrong stopping-certificate branch')
    measure_requests = [r for r in scenario.requests if r[0] == '/measure']
    require(len(measure_requests) >= 2 and measure_requests[0][1] == measure_requests[1][1],
            'Dropped response was not retried with exactly the same request body')
    require(len(measure_requests) == arena.counts['measure']+1,
            'Unexpected request/execution difference after one dropped measurement reply')
    require(scenario.executions['/measure'] == arena.counts['measure'],
            'A repeated measure request executed more than once')
    require(scenario.executions['/clear'] == arena.counts['clear_success']+arena.counts['clear_fail'],
            'Clear execution count mismatch')
    require(scenario.executions['/enter'] == scenario.executions['/exit'] == 1,
            'Expected one accepted enter and exit')

    direct = EndpointArena(case['sources'], case['seed'], FIELD)
    direct_trace = []
    direct_started = time.perf_counter()
    direct_result = solve_multi(public_api(direct), variant='share25', trace=direct_trace)
    direct_seconds = time.perf_counter()-direct_started
    direct_trace = json_form(direct_trace)
    write(directory/'direct.json', dict(case=case['name'], field=FIELD, sources=case['sources'],
        result=direct_result, events=direct.events, trace=direct_trace,
        evaluation=direct.evaluation(), solver_wall_s=direct_seconds))
    write(directory/'http_result.json', summary)
    equal_events = arena.events == direct.events
    equal_trace = http_trace == direct_trace
    equal_certificate = summary['certificate'] == json_form(direct_result['certificate'])
    equal_result = all(k in summary and summary[k] == json_form(v) for k,v in direct_result.items())
    comparison = dict(events_exactly_equal=equal_events, trace_exactly_equal=equal_trace,
        certificates_exactly_equal=equal_certificate, solver_result_fields_exactly_equal=equal_result,
        numeric_rounding_or_tolerance_used_for_action_comparison=False,
        retry_executed_once=True, fixed_cli_variant='share25', actual_cli_subprocess=True)
    write(directory/'comparison.json', comparison)
    # Audit each saved path separately, even if a cross-process comparison fails.
    mesh = PolarCover()
    http_audit = validate_run(case['sources'], arena.events, http_trace, summary,
                             mesh.stations, mesh.indices)
    direct_audit = validate_run(case['sources'], direct.events, direct_trace, direct_result,
                               mesh.stations, mesh.indices)
    write(directory/'audits.json', dict(http=http_audit, direct=direct_audit))
    require(all((equal_events, equal_trace, equal_certificate, equal_result)),
            'CLI/direct results differ; exact comparison files and both paths were preserved')
    require(http_audit['passed'] and direct_audit['passed'], 'Independent geometry audit failed')
    return dict(case=case['name'], field=FIELD, passed=True, source_count=len(case['sources']),
        cleared=arena.evaluation()['cleared'], total_virtual_s=arena.time_s,
        certificate_basis=expected_basis, comparison=comparison,
        http_audit=http_audit, direct_audit=direct_audit,
        fixture_url=fixture_url, local_only=True, official_simulator_contacted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='A new output directory; never overwritten.')
    args = parser.parse_args()
    # Importing an entry module makes no connection. Every actual client is
    # launched below with the explicit address of our newly started fixture.
    from q4_share25_client import source_hashes
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    snapshot, fixture_snapshot = source_hashes(), test_hashes()
    all_cases = {c['name']:c for c in stress_cases()}
    cases = [all_cases[name] for name in CASE_NAMES]
    require([len(c['sources']) for c in cases] == [10,16], 'Unexpected declared case sizes')
    write(out/'manifest.json', dict(local_only=True, official_simulator_contacted=False,
        variant='share25', cli_entry='q4_share25_client.py', python=platform.python_version(),
        cases=cases, field=FIELD, source_sha256=snapshot, validation_sha256=fixture_snapshot,
        connection_policy='Only own LocalServer bound to 127.0.0.1:0; reject port 2026; explicit --url for every CLI subprocess.',
        purpose='Full-session transport/certificate equivalence checks, not a performance benchmark.',
        scenario='Drop the first measurement response once to verify same-body retry executes once.'))
    rows = []
    for case in cases:
        try:
            row = run_case(case, out/case['name'], snapshot)
        except Exception as exc:
            row = dict(case=case['name'], passed=False, error_type=type(exc).__name__,
                       message=str(exc), traceback=traceback.format_exc())
        rows.append(row)
        print(f'{case["name"]}: passed={row["passed"]}; '
              f'{row.get("cleared", "?")}/{len(case["sources"])}; '
              f'{row.get("message", row.get("certificate_basis", ""))}', flush=True)
    stable = source_hashes() == snapshot and test_hashes() == fixture_snapshot
    branches = {r.get('certificate_basis') for r in rows if r['passed']}
    report = dict(passed=stable and all(r['passed'] for r in rows), tests_run=len(rows),
        cli_subprocess_sessions=sum(bool(r['passed']) for r in rows),
        source_snapshot_stable=stable, source_sha256=snapshot, validation_sha256=fixture_snapshot,
        both_stopping_branches_verified=branches == {'count_upper_bound','directional_triangle_cover'},
        local_only=True, official_simulator_contacted=False, rows=rows)
    report['passed'] &= report['both_stopping_branches_verified']
    write(out/'summary.json', report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_sha256','validation_sha256')},
                     ensure_ascii=False), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
