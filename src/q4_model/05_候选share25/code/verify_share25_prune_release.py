"""Read-only verification of the independent share25_prune release; no HTTP."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
Q4 = HERE.parents[1]
PROJECT = Q4.parent.parent
sys.path.insert(0, str(Q4))

from q4_share25_prune_client import VARIANT, source_hashes


def verify(manifest_path):
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    failures = []
    if manifest.get('variant') != VARIANT or manifest.get('local_validation_passed') is not True:
        failures.append('Candidate identity or local validation status does not match.')
    actual = source_hashes()
    expected = manifest.get('source_sha256', {})
    if actual != expected:
        failures.extend('Runtime mismatch: ' + name for name in sorted(set(actual) | set(expected))
                        if actual.get(name) != expected.get(name))
    evidence = manifest.get('evidence_sha256', {})
    if not evidence:
        failures.append('Release has no validation evidence.')
    for relative, digest in evidence.items():
        path = (PROJECT / relative).resolve()
        if not path.is_relative_to(PROJECT.resolve()):
            failures.append('Evidence path is outside the repository: ' + relative)
        elif not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append('Evidence mismatch: ' + relative)
    return dict(passed=not failures, variant=VARIANT, runtime_files=len(actual),
                evidence_files=len(evidence), local_only=True,
                official_simulator_contacted=False, failures=failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path,
                        default=HERE.parent / 'results/share25_prune_client_release.json')
    args = parser.parse_args()
    result = verify(args.manifest)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
