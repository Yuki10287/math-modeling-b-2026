"""Archive already-passed local evidence and a separate candidate release.

No client or simulator is imported. Existing release records are never replaced.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
RESULTS = HERE/'results'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_map(values, base):
    for name, expected in values.items():
        assert digest(base/name) == expected, f'File changed: {name}'


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def main():
    ready_path = RESULTS/'share25_readiness.json'
    release_path = RESULTS/'share25_client_release.json'
    assert not ready_path.exists() and not release_path.exists(), 'Release records already exist'
    protocol = read(RESULTS/'share25_http_validation.json')
    full = read(RESULTS/'share25_http_full/summary.json')
    selection = read(RESULTS/'cell_selection.json')
    assessment = read(RESULTS/'task_candidate_assessment.json')
    original_release = read(RESULTS/'client_release.json')
    assert protocol['passed'] and protocol['tests_run'] == 20
    assert protocol['source_snapshot_stable'] and protocol['protocol_source_snapshot_stable']
    assert full['passed'] and full['tests_run'] == 2 and full['cli_subprocess_sessions'] == 2
    assert full['source_snapshot_stable'] and full['both_stopping_branches_verified']
    assert protocol['source_sha256'] == full['source_sha256']
    for report in (protocol, full):
        assert report['local_only'] and not report['official_simulator_contacted']
    assert selection['selected_variant'] == 'share25'
    assert assessment['meets_predeclared_replacement_criteria']
    verify_map(selection['code_hashes'], HERE.parent)
    verify_map(protocol['source_sha256'], HERE.parent)
    verify_map(protocol['protocol_source_sha256'], HERE.parent)
    verify_map(protocol['test_sha256'], HERE)
    verify_map(full['validation_sha256'], HERE.parent)
    verify_map(original_release['source_sha256'], HERE.parent)

    ps1 = HERE/'run_share25.ps1'
    quoted = str(ps1).replace("'", "''")
    parse_code = f"""$parseTokens = $null
$parseErrors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile('{quoted}', [ref]$parseTokens, [ref]$parseErrors)
@{{error_count=@($parseErrors).Count; errors=@($parseErrors | ForEach-Object {{ $_.Message }})}} | ConvertTo-Json -Compress
if (@($parseErrors).Count -gt 0) {{ exit 1 }}
"""
    parsed = subprocess.run(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', parse_code],
        capture_output=True, text=True, encoding='utf-8', check=True)
    parser_result = json.loads(parsed.stdout)
    assert parser_result['error_count'] == 0
    assert ps1.read_bytes().startswith(b'\xef\xbb\xbf'), 'PowerShell needs UTF-8 BOM on Windows PowerShell'
    launcher_names = ['运行第四问候选.cmd', 'src/q4_model/运行第四问候选.cmd', 'src/q4_model/run_share25.ps1']
    for name in launcher_names:
        raw = (PROJECT/name).read_bytes()
        assert raw.count(b'\n') == raw.count(b'\r\n'), f'Expected CRLF: {name}'
    root_command = (PROJECT/launcher_names[0]).read_text(encoding='ascii')
    local_command = (PROJECT/launcher_names[1]).read_text(encoding='ascii')
    assert 'src\\q4_model\\run_share25.ps1' in root_command
    assert 'run_share25.ps1' in local_command
    for command in (root_command, local_command):
        assert 'exit /b %errorlevel%' in command.lower()
    script = ps1.read_text(encoding='utf-8-sig')
    assert 'verify_share25_release.py' in script and 'q4_share25_client.py' in script
    assert "'official_runs'" in script and "'share25'" in script
    assert script.index('verify_share25_release.py') < script.index('q4_share25_client.py')
    assert 'Read-Host' in script and 'exit $runExitCode' in script

    recorded = datetime.now(timezone.utc).isoformat()
    readiness = dict(recorded_at_utc=recorded, problem=4, variant='share25', ready_for_user_practice=True,
        local_only=True, official_simulator_contacted=False, model_unchanged_since_holdout_freeze=True,
        old_release_unchanged=True, protocol_tests=dict(passed=True, total=20, reused=13, candidate_specific=7),
        full_cli_sessions=full['rows'], actual_cli_actions_traces_and_certificates_match_direct=True,
        runtime_source_sha256=protocol['source_sha256'],
        launcher_validation=dict(powershell_parser=parser_result, cmd_ascii_crlf=True,
            powershell_utf8_bom_crlf=True, interactive_launcher_executed=False,
            actual_python_cli_executed_against_owned_ephemeral_fixtures=True),
        prior_model_evidence=dict(holdout_conditions=20, independent_holdout_layouts=10,
            mean_total_reduction_percent=assessment['paired_comparison']['relative_reduction_percent'],
            correctness_stress_conditions=assessment['stress']['runs'],
            correctness_stress_passed=assessment['stress']['passed']),
        environment_note='Initial sandbox invocation could not create test logs (WinError 5); the same local-only protocol command then passed with approved file access. Runtime code was not changed.',
        official_runtime_available=False, default_old_launcher_unchanged=True,
        user_entry='运行第四问候选.cmd', log_directory='src/q4_model/official_runs/share25',
        verification_limit='Interface equivalence uses self-built fixtures, not the official executable. Interactive PowerShell was parsed, not run against the official UI.')
    write_new(ready_path, readiness)
    sources = {'src/'+name: value for name,value in protocol['source_sha256'].items()}
    for name in launcher_names+['src/q4_model/requirements.txt', 'src/q4_model/verify_share25_release.py']:
        sources[name] = digest(PROJECT/name)
    evidence_names = ['cell_selection.json', 'task_candidate_assessment.json',
        'task_holdout_23000/analysis.json', 'task_stress_share25/summary.json',
        'share25_http_validation.json', 'share25_http_full/manifest.json',
        'share25_http_full/summary.json', 'share25_readiness.json']
    evidence = {'src/q4_model/results/'+name: digest(RESULTS/name) for name in evidence_names}
    release = dict(recorded_at_utc=recorded, problem=4, variant='share25',
        source_sha256=sources, evidence_sha256=evidence, official_simulator_used=False,
        official_evidence_available=False, default_entry_replaced=False,
        user_entry='运行第四问候选.cmd',
        launcher_validation='PowerShell parser and CMD/PowerShell encoding/path checks only; interactive launcher not executed.',
        interface_validation='20 protocol/session tests plus two full real Python CLI sessions on owned ephemeral loopback fixtures; 10/16 source paths exactly match direct model actions, trace and certificate.')
    write_new(release_path, release)
    print(f'Created candidate release: {len(sources)} files, {len(evidence)} evidence records; local only.')


if __name__ == '__main__': main()
