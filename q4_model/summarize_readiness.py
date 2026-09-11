"""Collect local readiness evidence and freeze the user-operated client release."""
import argparse
import hashlib
import json
import unittest
from datetime import datetime, timezone
from pathlib import Path

from benchmark_joint import hashes
from q4_official_client import source_hashes
from shared import ROOT
from test_thresholds import ThresholdChecks
from validation import validate_run
from polar_cover import PolarCover


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--http-evidence-dir', required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    output = here/'results'
    release_path = output/'client_release.json'
    readiness_path = output/'readiness_analysis.json'
    if release_path.exists() or readiness_path.exists():
        raise FileExistsError('已有发布记录；不覆盖历史验证。')
    stress_manifest = read(output/'stress-endpoints/manifest.json')
    stress = read(output/'stress-endpoints/summary.json')
    http = read(output/'q4_http_validation.json')
    assert hashes() == stress_manifest['source_sha256']
    assert source_hashes() == http['source_sha256']
    assert http['passed'] and http['source_snapshot_stable']
    assert stress['passed'] == stress['all_cleared'] == stress['runs'] == 30
    assert stress['source_snapshot_stable']
    assert hashlib.sha256((here/'stress_check.py').read_bytes()).hexdigest() == stress_manifest['test_sha256']
    for name, sha in http['test_sha256'].items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest() == sha
    original = read(output/'joint_selection.json')
    assert original['code_hashes'] == hashes(), 'selected model changed'
    thresholds = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ThresholdChecks))
    assert thresholds.wasSuccessful()
    integration = []
    case_dir = output/'http-local'
    case_dir.mkdir(exist_ok=False)
    mesh = PolarCover()
    evidence_dir = Path(args.http_evidence_dir)
    for index in (0, 9):
        evidence = read(evidence_dir/f'full-{index}-evidence.json')
        trace = read(evidence_dir/f'full-{index}.jsonl.beliefs.json')
        audit = validate_run(evidence['sources'], evidence['events'], trace,
                             evidence['summary'], mesh.stations, mesh.indices)
        assert audit == evidence['audit']
        assert evidence['direct_http_actions_equal'] and evidence['retry_executed_once']
        assert evidence['summary']['source_sha256'] == source_hashes()
        evidence['trace'] = trace
        path = case_dir/f'{evidence["case"]}.json'
        write_new(path, evidence)
        integration.append(dict(case=evidence['case'], source_count=len(evidence['sources']),
            complete=evidence['summary']['complete'], basis=evidence['summary']['certificate']['basis'],
            total_virtual_s=evidence['summary']['total_virtual_s'], audit=audit,
            direct_http_actions_equal=True, retry_executed_once=True,
            record=path.relative_to(ROOT).as_posix()))
    report = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), variant='shared',
        local_only=True, official_simulator_used=False, solver_unchanged_from_holdout=True,
        stress=dict(constructed_layouts=len(stress_manifest['cases']), runs=stress['runs'],
            passed=stress['passed'], all_cleared=stress['all_cleared'],
            source_instances=stress['source_instances'], audit_counts=stress['audit_counts'],
            maximum_total_s=stress['maximum_total_s'], maximum_runtime_s=stress['maximum_runtime_s'],
            layout_note='Ten deterministic constructions; outward_min_radius and one_omni also provide a tiny-rotation repeat. No statistical independence or representative mean is claimed.'),
        http_tests=dict(tests_run=http['tests_run'], passed=http['passed'], integration=integration),
        threshold_tests=dict(tests_run=thresholds.testsRun, passed=thresholds.wasSuccessful(),
            sha256=hashlib.sha256((here/'test_thresholds.py').read_bytes()).hexdigest()),
        rounding_scope='Nearest 0.01 degree after fixed errors attaining +/-1 degree; truncation is not covered.',
        third_question_changed=False)
    write_new(readiness_path, report)
    release_hashes = source_hashes()
    for name in ('requirements.txt', 'run_official.ps1', '运行第四问.cmd', 'verify_release.py'):
        p = here/name
        release_hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    evidence_paths = [readiness_path, output/'q4_http_validation.json', output/'stress-endpoints/manifest.json',
                      output/'stress-endpoints/summary.json']
    release = dict(recorded_at_utc=report['recorded_at_utc'], problem=4, variant='shared',
        official_simulator_used=False, source_sha256=release_hashes,
        evidence_sha256={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence_paths},
        launcher_validation='PowerShell parser checked separately; interactive launcher not executed.',
        interface_validation='Full Python CLI and Q4 session runner exercised only against ephemeral local HTTP fixtures.')
    write_new(release_path, release)
    print(json.dumps(dict(stress_runs=stress['runs'], http_tests=http['tests_run'],
        threshold_tests=thresholds.testsRun, release_files=len(release_hashes), local_only=True)))


if __name__ == '__main__':
    main()
